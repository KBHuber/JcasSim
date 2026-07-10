"""
FMCW radar sensing driven by a Sionna ray-traced scene, with a real
directional 8x8 array so angle discriminates a target from near-field clutter
(ground/wall bounces) -- the same mechanism real automotive radar relies on
(antenna directivity), not post-processing.

Sionna solves the monostatic backscatter paths (delay, per-element complex
gain, Doppler) once per snapshot; HermesPy's FMCW class (ping()/estimate(),
reused not duplicated) generates the chirp waveform and does the actual
dechirp. We do NOT use HermesPy's built-in SionnaRTChannel for propagation:
its _propagate() computes a full per-(tx-antenna, rx-antenna, sample) CIR
tensor, which is fine for a SISO link but blows GPU memory for a 64-element
array -- confirmed directly, it fails even at 1 chirp (out-of-memory trying
to allocate multiple GB). Instead we solve Sionna's paths once (no time axis)
and hand-replay the delay/Doppler across the chirp train ourselves, coherently
combining the real array elements -- which is what real FMCW hardware does
physically anyway (mix each chirp against a reference, track Doppler as phase
evolution across the chirp train).

    scene = make_scene(rt.scene.simple_street_canyon, look_at=[-75, 0, 20])
    sphere = add_target_sphere(scene, position=[-75, 0, 20])

    cube = sense_snapshot(scene, bs_position=[-100, 0, 1.5], look_at=[-75, 0, 20])
    plot_range_doppler(cube)

TX uses uniform, in-phase excitation across all 64 elements -- since the
array's real orientation (via look_at) already points the fixed beam at the
target, no separate steering-vector computation is needed for TX. RX scans a
grid of (azimuth, zenith) directions with a matched-filter combiner, using the
array's true WORLD-frame element positions (rotated via
sionna.rt.utils.rotation_matrix(receiver.orientation) -- PlanarArray.positions()
returns local, unrotated coordinates, confirmed directly: they don't change
when a Transmitter/Receiver's orientation does), so scan angles are real
compass directions.

Also confirmed directly: a purely specular sphere (default
scattering_coefficient=0) only backscatters from one infinitesimal point that
must exactly align with a mesh facet, which the default coarse tessellation
never hits, so add_target_sphere() gives it a nonzero scattering coefficient
-- without it the target was bit-for-bit invisible (present vs. absent gave
identical results).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light


def make_scene(scene_path, num_array_rows=8, num_array_cols=8):
    """Load a Sionna scene and give it a directional (tr38901) 8x8 tx/rx array."""
    import sionna.rt as rt

    scene = rt.load_scene(scene_path, merge_shapes=False)
    scene.tx_array = rt.PlanarArray(num_rows=num_array_rows, num_cols=num_array_cols, pattern="tr38901", polarization="V")
    scene.rx_array = scene.tx_array
    return scene


def add_target_sphere(scene, position, radius_m=0.25, scattering_coefficient=0.5, name="target"):
    """
    Add a metal sphere to the scene, so Sionna's ray tracer treats it as a
    physical radar target. A nonzero scattering_coefficient is required (see
    module docstring) for a monostatic solve to see any backscatter off a
    curved surface at all.

    Returns the SceneObject; move it between sense_snapshot() calls
    (`sphere.position = ...`) to animate a moving target.
    """
    import sionna.rt as rt

    mat_name = "_target_metal"
    if mat_name not in scene.radio_materials:
        mat = rt.ITURadioMaterial(
            name=mat_name, itu_type="metal", thickness=0.002,
            scattering_coefficient=scattering_coefficient,
        )
        scene.add(mat)
    else:
        mat = scene.radio_materials[mat_name]

    sphere = rt.SceneObject(fname=rt.scene.sphere, name=name, radio_material=mat)
    scene.edit(add=[sphere])
    sphere.scaling = radius_m
    sphere.position = position
    return sphere


class _RadarState:
    """Minimal duck-typed stand-in for HermesPy's TransmitState/ReceiveState --
    FMCW.ping()/estimate() only ever touch these three attributes."""

    def __init__(self, bandwidth, sampling_rate, oversampling_factor):
        self.bandwidth = bandwidth
        self.sampling_rate = sampling_rate
        self.oversampling_factor = oversampling_factor


def _steering_vector(element_positions_world, wavelength, azimuth_rad, zenith_rad):
    """Unit-norm planar-wave phase vector toward (azimuth, zenith), world frame."""
    direction = np.array([
        np.sin(zenith_rad) * np.cos(azimuth_rad),
        np.sin(zenith_rad) * np.sin(azimuth_rad),
        np.cos(zenith_rad),
    ])
    phase = 2 * np.pi / wavelength * (element_positions_world @ direction)
    w = np.exp(1j * phase)
    return w / np.sqrt(len(w))


def sense_snapshot(
    scene,                        # from make_scene()
    bs_position,                  # [3] radar position, world-frame meters
    look_at=None,                 # [3] point the array's boresight at (e.g. drone start)
    carrier_frequency=28e9,       # Hz
    bandwidth=100e6,              # Hz -- sets range resolution (c/2B)
    num_chirps=128,
    chirp_duration=10e-6,         # s
    pulse_rep_interval=11e-6,     # s -- must be >= chirp_duration
    tx_power_w=1.0,
    detector_min_power=0.5,       # relative peak-power threshold (0-1)
    azimuth_range_deg=(-180.0, 180.0),
    num_azimuth_bins=13,
    zenith_range_deg=(0.0, 90.0),
    num_zenith_bins=5,
    max_paths=300,                # cap on strongest paths kept, for tractability
):
    """
    Solve Sionna's monostatic backscatter paths once (no time axis) against
    `scene`'s real 8x8 directional array, then synthesize one FMCW frame by
    hand-replaying delay + Doppler per path and dechirping with HermesPy's
    FMCW (reused, not duplicated). Returns an angle-resolved RadarCube.
    """
    import sionna.rt as rt
    from sionna.rt.utils import rotation_matrix
    from hermespy.radar import FMCW, RadarCube
    from hermespy.core import Signal

    bs_position = np.asarray(bs_position, dtype=float)
    scene.frequency = carrier_frequency
    scene._transmitters.clear()
    scene._receivers.clear()
    tx = rt.Transmitter("bs_tx", bs_position.tolist())
    rx = rt.Receiver("bs_rx", bs_position.tolist())
    scene.add(tx)
    scene.add(rx)
    if look_at is not None:
        tx.look_at(look_at)
        rx.look_at(look_at)

    p_solver = rt.PathSolver()
    paths = p_solver(
        scene=scene, max_depth=5, los=True, specular_reflection=True,
        diffuse_reflection=True, refraction=True, synthetic_array=True, seed=41,
    )

    tau = np.array(paths.tau)[0, 0]           # [P] -- single rx, single tx
    doppler = np.array(paths.doppler)[0, 0]   # [P]
    a_np = np.array(paths.a)                  # [2, 1, rx_ant, 1, tx_ant, P]
    a = (a_np[0, 0] + 1j * a_np[1, 0])[:, 0]   # [rx_ant, tx_ant, P]

    wavelength = speed_of_light / carrier_frequency
    tx_pos_local = np.array(scene.tx_array.positions(wavelength)).T  # [tx_ant, 3]
    rx_pos_local = np.array(scene.rx_array.positions(wavelength)).T  # [rx_ant, 3]

    # PlanarArray.positions() returns LOCAL, unrotated element coordinates --
    # confirmed directly they don't change with a Transmitter's orientation --
    # but Sionna's ray tracer uses the TRUE (rotated) world positions to
    # compute `a`'s phase. Rotate into world frame so our own RX steering
    # vectors (built from these positions) are consistent with `a`.
    R_rx = np.array(rotation_matrix(rx.orientation))[:, :, 0]  # drjit keeps a trailing batch dim
    rx_pos_world = rx_pos_local @ R_rx.T

    # --- TX: uniform, in-phase excitation across all 64 elements. The array's
    # real orientation (look_at) already points the resulting fixed beam at
    # the target -- no separate TX steering-vector computation needed.
    a_eff = a.sum(axis=1)  # [rx_ant, P]

    # --- Filter: valid delay, non-negligible amplitude, cap to strongest paths ---
    tau_min = 2 * 3.0 / speed_of_light  # skip paths shorter than 3 m round-trip (self-coupling)
    strength = np.abs(a_eff).max(axis=0)
    valid = (tau >= tau_min) & (strength > 0)
    indices = np.where(valid)[0]
    if len(indices) > max_paths:
        indices = indices[np.argsort(strength[indices])[-max_paths:]]

    # --- HermesPy's FMCW derives its guard interval as
    # int((pulse_rep_interval - chirp_duration) * bandwidth), prone to
    # floating-point truncation (e.g. 11e-6 - 10e-6 == 9.999999999999989e-07,
    # one sample short) while FMCW.estimate()'s reshape uses
    # int(pulse_rep_interval * bandwidth) directly -- mismatched by a sample,
    # which accumulates across the chirp train into a spurious Doppler smear.
    guard_samples = round((pulse_rep_interval - chirp_duration) * bandwidth)
    pulse_rep_interval = chirp_duration + guard_samples / bandwidth

    fmcw = FMCW(num_chirps=num_chirps, chirp_duration=chirp_duration, pulse_rep_interval=pulse_rep_interval)
    state = _RadarState(bandwidth=bandwidth, sampling_rate=bandwidth, oversampling_factor=1)

    tx_frame = np.asarray(fmcw.ping(state)[0:1, :].view(np.ndarray)).reshape(-1)
    num_samples = len(tx_frame)
    t_axis = np.arange(num_samples) / bandwidth
    freqs = np.fft.fftfreq(num_samples, d=1.0 / bandwidth)
    tx_spectrum = np.fft.fft(tx_frame)

    # --- Delay-Doppler replay of each kept path, summed coherently across rx elements ---
    num_rx_ant = rx_pos_world.shape[0]
    received = np.zeros((num_rx_ant, num_samples), dtype=complex)
    for i in indices:
        delayed = np.fft.ifft(tx_spectrum * np.exp(-2j * np.pi * freqs * tau[i]))
        path_signal = delayed * np.exp(2j * np.pi * doppler[i] * t_axis)
        received += np.outer(a_eff[:, i], path_signal)
    received *= np.sqrt(tx_power_w)

    # --- Thermal noise, physically scaled (received now carries real Watts-scale amplitude) ---
    kTB = 1.38e-23 * 290 * bandwidth
    noise_w = kTB * 10 ** (7.0 / 10)  # 7 dB noise figure
    received += np.sqrt(noise_w / 2) * (
        np.random.normal(size=received.shape) + 1j * np.random.normal(size=received.shape)
    )

    # --- RX beamform-scan the azimuth/zenith grid, dechirp each beam with HermesPy's FMCW.estimate ---
    azimuths = np.deg2rad(np.linspace(*azimuth_range_deg, num_azimuth_bins))
    zeniths = np.deg2rad(np.linspace(*zenith_range_deg, num_zenith_bins))
    az_grid, ze_grid = np.meshgrid(azimuths, zeniths, indexing="ij")
    angle_grid = np.stack([az_grid.ravel(), ze_grid.ravel()], axis=-1)

    slabs = []
    for az, ze in angle_grid:
        w = _steering_vector(rx_pos_world, wavelength, az, ze)
        combined = np.conj(w) @ received
        signal = Signal.Create(combined[np.newaxis, :], sampling_rate=bandwidth)
        slabs.append(fmcw.estimate(signal, state))
    cube_data = np.stack(slabs, axis=0)  # [angle, doppler, range]

    return RadarCube(
        data=cube_data,
        angle_bins=angle_grid,
        doppler_bins=fmcw.relative_doppler_bins,
        range_bins=fmcw.range_bins(bandwidth),
        carrier_frequency=carrier_frequency,
    )


def plot_range_doppler(cube, max_range_m=None, title=None, ax=None):
    """Range-Doppler heatmap -- straight from HermesPy's own RadarCube.plot_range_velocity()."""
    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=(10, 5))

    result = cube.plot_range_velocity(axes=ax, scale="velocity")
    plt.colorbar(result.mesh, ax=ax, label="Normalized Power")

    if max_range_m is not None:
        ax.set_xlim(0, max_range_m)

    ax.set_title(title or "Range-Doppler Map")

    if create_fig:
        plt.tight_layout()
        plt.show()
