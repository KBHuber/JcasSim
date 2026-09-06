"""
OFDM JCAS radar sensing driven by a Sionna ray-traced scene.

Sionna solves the monostatic backscatter paths (delay, per-element complex gain,
Doppler) once per snapshot; sense_snapshot(waveform=...) then estimates the
range-Doppler cube with one of two waveforms, on the identical solve:

  - "ofdm": HermesPy's OFDMRadar (hermespy.jcas), using the same numerology as
    the comms link (ofdm_config) -- radar and comms are one joint waveform.
  - "fmcw": HermesPy's FMCW chirp train, for accuracy/resolution comparison.

Both scan the same angle grid with the same ConventionalBeamformer, so the
waveform is the only thing that differs between them.

Propagation goes through GeometricChannelResponse (geometric_sionna_channel.py),
which keeps Sionna's compact per-path (gain, delay, Doppler) and replays them
across the frame in closed form instead of materializing a per-antenna-pair CIR.

    cube = sense_snapshot(scene, bs_position=[-100, 0, 1.5], look_at=[-75, 0, 20])
    plot_range_doppler(cube)

Callers build the scene (comms TX/RX + setup_drone_meshes() cube per drone, see
sim.py) and pass it in, so comms and sensing share one scene, antenna position
and boresight.

TX excitation is Hann-tapered over element position, for a wider main lobe with
low sidelobes instead of a uniformly-lit array's pencil beam. RX scans an
(azimuth, zenith) grid with a matched-filter combiner built from the array's
world-frame element positions, so scan angles are real compass directions.

Targets need a nonzero scattering coefficient to backscatter usefully; purely
specular meshes read as invisible. setup_drone_meshes() (sim.py) gives the drone
bodies one.
"""

from dataclasses import dataclass

import numpy as np
from numpy.fft import fft, ifft, ifftshift
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light
from scipy.signal import convolve

import mitsuba as mi
import drjit as dr
import sionna.rt as rt
from sionna.rt.utils import rotation_matrix

from hermespy.core import Signal, Transformation
from hermespy.simulation import SimulatedDevice
from hermespy.simulation.antennas import SimulatedCustomArray, SimulatedIdealAntenna
from hermespy.beamforming import ConventionalBeamformer
from hermespy.jcas import OFDMRadar
from hermespy.radar import RadarCube, FMCW, RadarPointCloud, PointDetection
from hermespy.modem.waveforms.orthogonal.waveform import ElementType

from geometric_sionna_channel import GeometricChannelResponse
from ofdm_config import make_ofdm_waveform
import ofdm_config


def _range_window(n):
    """Window across subcarriers, applied to the division matrix before the range IDFT.
    Costs 1.76 dB of SNR and ~1.62x mainlobe width; bin spacing is unchanged."""
    return np.hanning(n)


def _doppler_window(n):
    """Window across slow-time symbols, applied to the division matrix before the Doppler
    DFT. Same cost as _range_window."""
    return np.hanning(n)


def _hann_axis(p):
    """Hann amplitude weight per element, from that element's position along one array
    axis. Falls to exactly zero at the edge elements."""
    half = np.max(np.abs(p))
    return 0.5 * (1.0 + np.cos(np.pi * p / half)) if half > 0 else np.ones_like(p)


def _hamming_axis(p):
    """Hamming amplitude weight per element, from its position along one array axis.
    Hann is wrong here: it zeroes the edge elements of a short axis. Shared with
    jcas_drop.py's radar receive."""
    half = np.max(np.abs(p))
    return 0.54 + 0.46 * np.cos(np.pi * p / half) if half > 0 else np.ones_like(p)


def _axis_hpbw_deg(axis_pos, weights, k):
    """Half-power beamwidth (deg) along one array axis, swept numerically from the array
    factor. `axis_pos` are element positions along that axis, `weights` their amplitude
    weights, `k` the wavenumber."""
    # sweep off broadside, find where array factor power halves; HPBW = 2x that
    if axis_pos.max() - axis_pos.min() <= 0:
        return 360.0
    theta = np.linspace(0.0, np.pi / 2, 20001)
    af = np.abs(weights @ np.exp(1j * k * np.outer(axis_pos, np.sin(theta)))) ** 2
    af /= af[0]
    below = np.argmax(af <= 0.5)
    if below == 0:
        return 360.0  # never halves within [0, 90deg): effectively omni
    # interpolate the exact -3 dB crossing
    t0, t1, a0, a1 = theta[below - 1], theta[below], af[below - 1], af[below]
    theta_half = t0 + (0.5 - a0) * (t1 - t0) / (a1 - a0)
    return np.degrees(2 * theta_half)


def _center_rad(az, boresight_azimuth_rad):
    """Wrap an azimuth (rad) into (-pi, pi] relative to the array boresight."""
    return np.arctan2(np.sin(az - boresight_azimuth_rad), np.cos(az - boresight_azimuth_rad))


def _array_beamwidth_deg(pos_local, wavelength, weights=None):
    """
    Half-power (-3 dB) beamwidth (deg) of the array along its local horizontal
    (y) and vertical (z) axes, measured numerically from the actual array factor
    (so element geometry and amplitude taper are accounted for). Used to size
    sense_snapshot()'s RX scan grid at the array's own angular resolution.

    weights: per-element amplitude weights, default uniform to match the RX
    steering combiner; pass the TX taper to get that wider beam's width.
    """
    k = 2 * np.pi / wavelength
    n = pos_local.shape[0]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    return (_axis_hpbw_deg(pos_local[:, 1], w, k),
            _axis_hpbw_deg(pos_local[:, 2], w, k))


class _SensingOFDMRadar:
    """Mixed into an OFDMRadar instance (see _build_ofdm_radar) to add
    estimate_cube(): OFDMRadar's own beamforming and Sturm & Wiesbeck range-Doppler
    estimation, minus the communication symbol decode _receive would also run.
    Shared by sense_snapshot() and jcas_drop.py's monostatic radar link."""

    def estimate_cube(self, signal, device):
        transmitted = self.waveform.place(self._OFDMRadar__last_transmission.symbols)
        angles, beamformed = self._receive_beamform(signal, device)
        rr = self.range_resolution(device.bandwidth)

        # Sense on the dedicated pilot comb if the waveform carries one
        # (ofdm_config SENSING_SUBCARRIER_MODE == "comb"), else reciprocal-filter
        # the whole frame (Sturm & Wiesbeck).
        ref_mask = self.waveform.resource_mask[ElementType.REFERENCE.value][0]  # [num_subcarriers]
        if ref_mask.any():
            return self._estimate_comb(transmitted, beamformed, device, ref_mask, rr)

        # Filled in place: a list plus the np.stack that copies it holds two full blocks
        # at once. float32 because the values are magnitudes.
        range_bins = np.arange(self.waveform.num_subcarriers) * rr
        cube = np.empty((len(beamformed), self.waveform.words_per_frame,
                         self.waveform.num_subcarriers), dtype=np.float32)
        for i, line in enumerate(beamformed):
            cube[i] = self._estimate_range(transmitted, line, device)

        # undo the ifft's 1/N so the OFDM and FMCW cubes share an absolute scale
        cube = cube * self.waveform.num_subcarriers
        return range_bins, cube

    def _estimate_range(self, transmitted, line, device):
        """OFDMRadar.__estimate_range with the division matrix windowed in range and
        Doppler before the two transforms."""
        received = self.waveform.demodulate(line, device.bandwidth, device.oversampling_factor)
        tx = transmitted.raw
        normalized = np.divide(
            received.raw, tx, np.zeros_like(received.raw), where=np.abs(tx) != 0.0
        )
        d = normalized[0]  # [words, subcarriers]
        d = d * _doppler_window(d.shape[0])[:, np.newaxis] * _range_window(d.shape[1])[np.newaxis, :]
        return np.abs(ifftshift(fft(ifft(d, axis=1), axis=0), axes=0))

    def _estimate_comb(self, transmitted, beamformed, device, ref_mask, rr):
        """Range-Doppler estimate from the radar pilot comb alone -- the
        subcarrier-interleaved joint radar/comms approach (Sturm, Zwick & Wiesbeck,
        VTC 2009). Dividing by the known constant-modulus pilots avoids the
        data-dependent self-noise of dividing by random 16-QAM data.

        The comb is K = N/M uniformly-spaced subcarriers: range resolution stays
        c/2B, but unambiguous range shrinks by M to c/(2*M*df).
        """
        tx_ref = transmitted.raw[0][:, ref_mask]  # [words, K] known pilots
        K = tx_ref.shape[1]
        slabs = []
        for line in beamformed:
            rx = self.waveform.demodulate(line, device.bandwidth, device.oversampling_factor)
            rx_ref = rx.raw[0][:, ref_mask]  # [words, K]
            channel = np.divide(
                rx_ref, tx_ref, np.zeros_like(rx_ref), where=np.abs(tx_ref) != 0.0
            )
            channel = (channel * _doppler_window(channel.shape[0])[:, np.newaxis]
                               * _range_window(channel.shape[1])[np.newaxis, :])
            # range = IDFT over the comb subcarriers, doppler = DFT over slow-time words
            profile = ifftshift(fft(ifft(channel, axis=1), axis=0), axes=0)
            slabs.append(np.abs(profile) * K)  # *K undoes the ifft's 1/K
        cube = np.stack(slabs, axis=0)  # [angle, doppler, range]
        range_bins = np.arange(K) * rr
        return range_bins, cube


def _build_ofdm_radar(rx_pos_world, bs_position, carrier_frequency, bandwidth):
    """SimulatedDevice + OFDM radar operator for monostatic sensing. The array is
    built from Sionna's world-frame element positions, so the beamform scan stays
    phase-consistent with the ray-traced channel coefficients.

    Returns (radar_device, radar, device_state) -- device_state.transmit_state()
    feeds radar.transmit(), receive_state() feeds the scan.
    """

    radar_device = SimulatedDevice(
        antennas=SimulatedCustomArray(ports=[
            SimulatedIdealAntenna(pose=Transformation.From_Translation(p)) for p in rx_pos_world
        ]),
        carrier_frequency=carrier_frequency, bandwidth=bandwidth, oversampling_factor=1,
        pose=Transformation.From_Translation(np.asarray(bs_position, dtype=float)),
    )
    radar_cls = type("_BoundSensingOFDMRadar", (_SensingOFDMRadar, OFDMRadar), {})
    radar = radar_cls(waveform=make_ofdm_waveform(), receive_beamformer=ConventionalBeamformer(),
                      selected_transmit_ports=[0])
    radar_device.add_dsp(radar)
    device_state = radar_device.state(0.0)
    return radar_device, radar, device_state


def _ofdm_radar_cube_from_received(
    radar, rx_state, received, angle_grid, element_gain, bandwidth, carrier_frequency,
    angle_chunk=8, zero_doppler_guard=0, max_range_m=None,
):
    """Beamform-scan `received` [rx_ant, T] over `angle_grid` and Sturm & Wiesbeck-estimate
    a range-Doppler-angle RadarCube. Shared by sense_snapshot() (waveform="ofdm") and
    jcas_drop.py's monostatic radar link.

    The grid is probed in `angle_chunk`-sized chunks; the chunk is
    [angle_chunk, frame_samples] complex128, so angle_chunk trades peak memory against
    per-chunk overhead. `max_range_m` is applied per slab, not to the assembled cube --
    holding the untrimmed [all_angles, doppler, subcarriers] block is what runs out of
    memory.
    """

    received_signal = Signal.Create(received, sampling_rate=bandwidth, carrier_frequency=carrier_frequency)
    slabs = []
    cube_range_bins = None
    range_keep = None
    num_rx_ant = received.shape[0]
    for start in range(0, angle_grid.shape[0], angle_chunk):
        chunk = angle_grid[start:start + angle_chunk]
        radar.receive_beamformer.probe_focus_points = chunk[:, np.newaxis, :]
        cube_range_bins, slab = radar.estimate_cube(received_signal, rx_state)

        # probe() normalizes by 1/N where this codebase uses the power-preserving
        # 1/sqrt(N); element_gain is a power gain applied to a complex amplitude,
        # hence the sqrt. The FMCW branch of sense_snapshot() does the same pair.
        if range_keep is None:
            range_keep = (cube_range_bins <= max_range_m if max_range_m is not None
                          else slice(None))
        slab = slab[:, :, range_keep]
        slab = slab * np.sqrt(num_rx_ant)
        slab = slab * np.sqrt(element_gain[start:start + chunk.shape[0]])[:, np.newaxis, np.newaxis]
        # float32: the cube is real magnitudes, consumed only as |.|^2
        slabs.append(slab.astype(np.float32))
    if max_range_m is not None:
        cube_range_bins = cube_range_bins[range_keep]
    cube_data = np.concatenate(slabs, axis=0)  # [angle, doppler, range]
    del slabs

    # Centred, negated Doppler axis spaced at 1/(2T), replacing OFDMRadar's own:
    # the Sturm estimator puts zero Doppler at bin N/2 and measures the round-trip
    # shift (negative when receding), while RadarCube converts to velocity with the
    # one-way v = doppler * c / f_c. Negating matches ground_truth_range_velocity's
    # sign (receding = +v).
    n_sym = radar.waveform.words_per_frame
    doppler_bins = -(np.arange(n_sym) - n_sym // 2) * radar.relative_doppler_resolution(bandwidth)

    # Null the zero-velocity clutter band. A static scatterer sits at exactly zero
    # Doppler, so one bin is the right width; leakage past it is the window's job. The
    # nulled width is (guard + 0.5) * velocity resolution, and drones slower than that
    # share the clutter cell and are lost either way.
    zero_doppler_idx = np.argmin(np.abs(doppler_bins))
    lo = max(0, zero_doppler_idx - zero_doppler_guard)
    hi = min(len(doppler_bins), zero_doppler_idx + zero_doppler_guard + 1)
    cube_data[:, lo:hi, :] = 0

    return RadarCube(
        data=cube_data, angle_bins=angle_grid, doppler_bins=doppler_bins,
        range_bins=cube_range_bins, carrier_frequency=carrier_frequency,
    )


def sense_snapshot(
    scene,                        # Sionna scene with tx_array/rx_array configured
    bs_position,                  # [3] radar position, world-frame meters
    look_at=None,                 # [3] point the array's boresight at (e.g. drone start)
    waveform="fmcw",              # "ofdm" or "fmcw"
    carrier_frequency=None,       # None -> ofdm_config.CARRIER_FREQUENCY (Hz)
    bandwidth=None,               # None -> ofdm_config.BANDWIDTH, shared by both waveforms
    num_chirps=128,               # FMCW only
    chirp_duration=10e-6,         # FMCW only, s
    pulse_rep_interval=11e-6,     # FMCW only, s -- must be >= chirp_duration
    tx_power_w=1.0,
    azimuth_range_deg=(-180.0, 180.0),
    num_azimuth_bins=None,        # None -> auto-sample at HPBW/angular_oversample
    zenith_range_deg=(0.0, 90.0),
    num_zenith_bins=None,         # None -> auto-sample at HPBW/angular_oversample
    angular_oversample=3,         # scan-grid samples per HPBW when num_*_bins is None.
                                   # Interpolates the beam pattern more finely; true
                                   # resolution is still capped by the aperture.
    max_range_m=None,             # trim the range axis to the instrumented range; see
                                   # _ofdm_radar_cube_from_received
    max_num_paths_per_src=5000,   # cap on candidate paths the solver keeps. Sionna's
                                   # default (1e6) can exhaust memory once walls scatter
                                   # diffusely; every path returned is then replayed,
                                   # since weak diffuse returns are the clutter floor.
):
    """
    Solve Sionna's monostatic backscatter paths once (no time axis) against
    `scene`'s 8x8 directional array, replay delay + Doppler per path through the
    geometric channel, then estimate a range-Doppler-angle cube with either
    HermesPy's OFDM JCAS radar (shares the comms waveform, see ofdm_config.py) or
    HermesPy's FMCW. Returns an angle-resolved RadarCube.

    Range bins are spaced at c/2B (~1.46 m at 102.4 MHz) for either waveform. Angle
    bins default to the array's HPBW (_array_beamwidth_deg) / angular_oversample.
    """
    if waveform not in ("ofdm", "fmcw"):
        raise ValueError(f"waveform must be 'ofdm' or 'fmcw', got {waveform!r}")
    if carrier_frequency is None:
        carrier_frequency = ofdm_config.CARRIER_FREQUENCY
    if bandwidth is None:
        bandwidth = ofdm_config.BANDWIDTH

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
    # max_depth matches run_jcas_drop()'s solve, so the two cubes stay comparable.
    paths = p_solver(
        scene=scene, max_depth=3, los=True, specular_reflection=True,
        diffuse_reflection=True, refraction=True, synthetic_array=True, seed=41,
        max_num_paths_per_src=max_num_paths_per_src,
    )

    # Parametric geometric channel: each path's (complex gain, delay, Doppler) from
    # the synthetic-array solve, no time axis and no per-antenna-pair densification.
    channel = GeometricChannelResponse.from_sionna_paths(paths, carrier_frequency)

    wavelength = speed_of_light / carrier_frequency
    tx_pos_local = np.array(scene.tx_array.positions(wavelength)).T  # [tx_ant, 3]
    rx_pos_local = np.array(scene.rx_array.positions(wavelength)).T  # [rx_ant, 3]

    # Size the scan grid by the array's own HPBW -- the angular analogue of c/2B.
    az_hpbw_deg, ze_hpbw_deg = _array_beamwidth_deg(rx_pos_local, wavelength)
    az_bin_deg = az_hpbw_deg / angular_oversample
    ze_bin_deg = ze_hpbw_deg / angular_oversample
    if num_azimuth_bins is None:
        num_azimuth_bins = max(2, round((azimuth_range_deg[1] - azimuth_range_deg[0]) / az_bin_deg) + 1)
    if num_zenith_bins is None:
        num_zenith_bins = max(2, round((zenith_range_deg[1] - zenith_range_deg[0]) / ze_bin_deg) + 1)

    # positions() is local and unrotated; the ray tracer phases `a` on rotated world
    # positions, so rotate to keep the RX steering vectors consistent with it.
    R_rx = np.array(rotation_matrix(rx.orientation))[:, :, 0]  # drjit keeps a batch dim
    rx_pos_world = rx_pos_local @ R_rx.T

    # TX: in-phase excitation across all elements (look_at already aims the fixed
    # beam), Hann-tapered by element position for a broader main lobe with ~-31 dB
    # sidelobes. Hann is separable over the two aperture axes.
    tx_weights = _hann_axis(tx_pos_local[:, 1]) * _hann_axis(tx_pos_local[:, 2])
    # sum(|w|^2) == N: same total radiated power as uniform excitation
    tx_weights *= np.sqrt(len(tx_weights) / np.sum(tx_weights ** 2))

    # The radar as a HermesPy device, its array built from Sionna's world-frame element
    # positions so the ConventionalBeamformer scan stays phase-consistent with the
    # ray-traced coefficients. Transmit is single-stream (the illumination beam is
    # applied via the channel's tx_weights); reception keeps all elements for the scan.
    if waveform == "ofdm":
        radar_device, radar, device_state = _build_ofdm_radar(rx_pos_world, bs_position, carrier_frequency, bandwidth)
        rx_state = device_state.receive_state()
        # OFDMRadar stashes the transmitted symbols it divides out during range estimation
        tx_frame = radar.transmit(device_state).signal.view(np.ndarray)[0]

    else:  # waveform == "fmcw"
        radar_device = SimulatedDevice(
            antennas=SimulatedCustomArray(ports=[
                SimulatedIdealAntenna(pose=Transformation.From_Translation(p)) for p in rx_pos_world
            ]),
            carrier_frequency=carrier_frequency, bandwidth=bandwidth, oversampling_factor=1,
            pose=Transformation.From_Translation(bs_position),
        )
        # FMCW truncates its guard interval where estimate()'s reshape rounds; round
        # here too, or the two drift a sample apart into a spurious Doppler smear.
        guard_samples = round((pulse_rep_interval - chirp_duration) * bandwidth)
        pulse_rep_interval = chirp_duration + guard_samples / bandwidth
        fmcw = FMCW(num_chirps=num_chirps, chirp_duration=chirp_duration, pulse_rep_interval=pulse_rep_interval)
        beamformer = ConventionalBeamformer()
        device_state = radar_device.state(0.0)
        tx_state = device_state.transmit_state()
        rx_state = device_state.receive_state()

        tx_frame = np.asarray(fmcw.ping(tx_state)[0:1, :].view(np.ndarray)).reshape(-1)

    # Illuminate and propagate: the channel combines the tapered tx excitation per path,
    # replays delay + Doppler across the frame, and sums coherently across rx elements.
    received = channel.propagate(
        tx_frame, sampling_rate=bandwidth, tx_weights=tx_weights,
        power_w=tx_power_w, min_range_m=3.0,
    )

    # Thermal noise, physically scaled (received carries real Watts-scale amplitude)
    kTB = 1.38e-23 * 290 * bandwidth
    noise_w = kTB * 10 ** (7.0 / 10)  # 7 dB noise figure
    received += np.sqrt(noise_w / 2) * (
        np.random.normal(size=received.shape) + 1j * np.random.normal(size=received.shape)
    )

    # RX aperture taper (Hamming over both array axes), applied per element before the
    # scan. Pre-scaling element i by w_i is exactly a tapered beamformer, dropping the
    # uniform combiner's ~-13 dB angular sidelobes to ~-34 dB at ~3.5 dB aperture loss.
    # After the noise, so the taper weights it as real hardware would. run_jcas_drop()
    # applies the identical taper; keep the two in step.
    rx_taper = _hamming_axis(rx_pos_local[:, 1]) * _hamming_axis(rx_pos_local[:, 2])
    rx_taper *= len(rx_taper) / rx_taper.sum()  # sum(w) = N: preserve boresight gain
    received *= rx_taper[:, np.newaxis]

    # Scan grid, kept half an HPBW clear of zenith 0: straight up has no azimuth
    # dependence, so sampling the pole smears one beam across a whole azimuth row.
    azimuths = np.deg2rad(np.linspace(*azimuth_range_deg, num_azimuth_bins))
    zeniths = np.deg2rad(np.linspace(max(zenith_range_deg[0], ze_hpbw_deg / 2), zenith_range_deg[1], num_zenith_bins))
    az_grid, ze_grid = np.meshgrid(azimuths, zeniths, indexing="ij")
    angle_grid = np.stack([az_grid.ravel(), ze_grid.ravel()], axis=-1)

    # Real per-element pattern (tr38901) as a scalar gain per scan direction: for
    # identical co-pointed elements, array response = element pattern * array factor.
    # ConventionalBeamformer's codebook is pure isotropic phase, so without this the
    # element's ~30 dB front/back ratio is lost and every detection ghosts behind the array.
    directions_world = np.stack([
        np.sin(angle_grid[:, 1]) * np.cos(angle_grid[:, 0]),
        np.sin(angle_grid[:, 1]) * np.sin(angle_grid[:, 0]),
        np.cos(angle_grid[:, 1]),
    ], axis=-1)
    directions_local = directions_world @ R_rx  # world -> local (R_rx is orthogonal)
    theta_local = np.arccos(np.clip(directions_local[:, 2], -1.0, 1.0))
    phi_local = np.arctan2(directions_local[:, 1], directions_local[:, 0])
    c_theta, c_phi = scene.rx_array.antenna_pattern.patterns[0](mi.Float(theta_local), mi.Float(phi_local))
    element_gain = np.array(dr.abs(c_theta) ** 2 + dr.abs(c_phi) ** 2)  # [angle] linear power gain

    angle_chunk = 64  # scanning the full grid at once would materialize an
                       # [all_angles, sample] block of ~4 GB

    if waveform == "ofdm":
        return _ofdm_radar_cube_from_received(
            radar, rx_state, received, angle_grid, element_gain, bandwidth, carrier_frequency,
            angle_chunk=angle_chunk, max_range_m=max_range_m,
        )

    else:  # waveform == "fmcw"
        # Beamform-scan with the same ConventionalBeamformer, dechirp each beam with
        # FMCW.estimate, rescaling probe()'s 1/N to this codebase's 1/sqrt(N).
        #
        # NOT WINDOWED, unlike the OFDM branch: FMCW.estimate dechirps internally and
        # exposes no window. Fix before comparing the two waveforms.
        slabs = []
        for start in range(0, angle_grid.shape[0], angle_chunk):
            chunk = angle_grid[start:start + angle_chunk]
            beamformed = beamformer.probe(received, rx_state, chunk[:, np.newaxis, :])[:, 0, :]
            beamformed *= np.sqrt(rx_pos_world.shape[0])
            beamformed *= np.sqrt(element_gain[start:start + chunk.shape[0]])[:, np.newaxis]
            for combined in beamformed:
                signal = Signal.Create(combined[np.newaxis, :], sampling_rate=bandwidth)
                slabs.append(fmcw.estimate(signal, rx_state))
        cube_data = np.stack(slabs, axis=0).astype(np.complex64)  # [angle, doppler, range]
        cube_range_bins = fmcw.range_bins(bandwidth)

        # Sionna's Doppler is the round-trip shift, but RadarCube converts to velocity
        # with the one-way v = doppler * c / f_c; halve the bins so speeds come out right.
        doppler_bins = fmcw.relative_doppler_bins / 2

        # Clutter guard: the zero-Doppler return spills a few bins wide (windowing and
        # finite chirp count), so null a band rather than the single closest bin.
        zero_doppler_guard = 3

    zero_doppler_idx = np.argmin(np.abs(doppler_bins))
    lo = max(0, zero_doppler_idx - zero_doppler_guard)
    hi = min(len(doppler_bins), zero_doppler_idx + zero_doppler_guard + 1)
    cube_data[:, lo:hi, :] = 0

    # Trim to the instrumented range. The axis spans one bin per subcarrier, i.e. the
    # full unambiguous range c/(2*df) -- ~5 km at 30 kHz SCS -- so a 200 m scene leaves
    # ~95% of it empty. Those bins hold nothing but noise and far clutter, and every one
    # is another CFAR hypothesis test. HermesPy's OFDMRadar trims the near end the same
    # way (min_range); this is the far end.
    if max_range_m is not None:
        keep = cube_range_bins <= max_range_m
        cube_data = cube_data[:, :, keep]
        cube_range_bins = cube_range_bins[keep]

    return RadarCube(
        data=cube_data,
        angle_bins=angle_grid,
        doppler_bins=doppler_bins,
        range_bins=cube_range_bins,
        carrier_frequency=carrier_frequency,
    )


def ground_truth_range_velocity(bs_position, target_position, target_velocity):
    """True (range, radial velocity) of a target relative to bs_position -- the two
    quantities a Range-Doppler map's axes represent, so a peak can be checked
    against ground truth. Positive velocity = receding, matching
    RadarCube.velocity_bins.
    """
    bs_position = np.asarray(bs_position, dtype=float)
    delta = np.asarray(target_position, dtype=float) - bs_position
    rng = np.linalg.norm(delta)
    radial_velocity = np.dot(np.asarray(target_velocity, dtype=float), delta) / rng
    return float(rng), float(radial_velocity)


def peak_range_velocity(cube, max_range_m=None):
    """(range_m, velocity_mps) of the brightest point in the angle-summed map
    RadarCube.plot_range_velocity() displays.

    max_range_m: restrict the search to range bins below this. The unambiguous
    range runs far past the scene, so without it a frame whose return is below the
    noise floor reports whichever noise bin peaks furthest out. Pass the same value
    used for the plot so the print and the picture agree.
    """
    range_velocity_profile = np.abs(np.sum(cube.data, axis=0))
    if max_range_m is not None:
        range_velocity_profile = range_velocity_profile[:, cube.range_bins <= max_range_m]
    vel_idx, range_idx = np.unravel_index(np.argmax(range_velocity_profile), range_velocity_profile.shape)
    range_bins = cube.range_bins if max_range_m is None else cube.range_bins[cube.range_bins <= max_range_m]
    return float(range_bins[range_idx]), float(cube.velocity_bins[vel_idx])


def world_to_spherical(bs_position, target_position):
    """(range_m, azimuth_rad, zenith_rad) of target_position relative to bs_position,
    in the world-frame spherical convention sense_snapshot()'s RX scan uses (azimuth
    from +x in the xy-plane, zenith from +z), so ground truth can be overlaid on a
    range-angle plot.
    """
    delta = np.asarray(target_position, dtype=float) - np.asarray(bs_position, dtype=float)
    rng = np.linalg.norm(delta)
    zenith = np.arccos(delta[2] / rng)
    azimuth = np.arctan2(delta[1], delta[0])
    return float(rng), float(azimuth), float(zenith)


def _local_maxima(cube, hits, n_angle, n_doppler, n_range):
    """Keep only threshold crossings that are a local maximum over their 3x3x3
    (angle, doppler, range) neighbourhood -- CFAR's peak-grouping stage.

    Only flagged cells are tested, so this costs ~27 lookups per hit rather than a pass
    over the cube. The angle axis is a raveled (azimuth, zenith) meshgrid with
    indexing="ij", so neighbours are taken on that 2D grid; the flat index would wrap one
    azimuth column into the next.
    """
    n_ze = len(np.unique(cube.angle_bins[:, 1]))
    n_az = n_angle // n_ze if n_ze else 0
    product_grid = n_ze > 0 and n_az * n_ze == n_angle

    def angle_neighbours(a):
        if not product_grid:                      # not a meshgrid: fall back to 1D
            return [x for x in (a - 1, a, a + 1) if 0 <= x < n_angle]
        i_az, i_ze = divmod(a, n_ze)
        return [j_az * n_ze + j_ze
                for j_az in range(max(i_az - 1, 0), min(i_az + 2, n_az))
                for j_ze in range(max(i_ze - 1, 0), min(i_ze + 2, n_ze))]

    kept = []
    for a, d, r, power in hits:
        for a2 in angle_neighbours(a):
            for d2 in range(max(d - 1, 0), min(d + 2, n_doppler)):
                for r2 in range(max(r - 1, 0), min(r + 2, n_range)):
                    if (a2, d2, r2) != (a, d, r) and abs(cube.data[a2, d2, r2]) ** 2 > power:
                        break
                else:
                    continue
                break
            else:
                continue
            break
        else:
            kept.append((a, d, r, power))
    return kept


def detect_targets(cube, num_training_cells=(8, 16), num_guard_cells=(2, 4), pfa=5e-8,
                   peak_extraction=True):
    """CFAR-detect targets in every angle bin separately, so the decision is made in the
    full (azimuth, zenith, range, Doppler) resolution cell rather than in a range-Doppler
    plane with the angle axis collapsed away. `pfa` is per cell and the cube holds ~2e8
    cells, so the default targets ~10 false alarms per frame -- scale it with the cube.
    """
    n_angle, n_doppler, n_range = cube.data.shape

    window_size_x = 2 * num_training_cells[0] + 2 * num_guard_cells[0] + 1
    window_size_y = 2 * num_training_cells[1] + 2 * num_guard_cells[1] + 1
    if n_doppler < window_size_x or n_range < window_size_y:
        raise ValueError(
            "Radar cube doppler and range dimensions must be bigger than num_training_cells + num_guard_cells + 1"
        )

    kernel = np.ones((window_size_x, window_size_y))
    kernel[num_training_cells[0]:-num_training_cells[0], num_training_cells[1]:-num_training_cells[1]] = 0

    # Training-cell count per cell under test, reduced at the edges by mode="same".
    # Identical for every angle slab, so convolve the ones-plane once.
    threshold_normalization = np.round(
        convolve(np.ones((n_doppler, n_range)), kernel, mode="same", method="fft")
    )
    threshold_factor = pfa ** (-1 / threshold_normalization) - 1

    hits = []
    for angle_idx in range(n_angle):
        # float64 for the FFT convolution's precision, even though the cube is complex64
        power = np.abs(cube.data[angle_idx]).astype(np.float64) ** 2   # [doppler, range]
        noise_threshold = convolve(power, kernel, mode="same", method="fft")
        for doppler_idx, range_idx in np.argwhere(power > noise_threshold * threshold_factor):
            hits.append((angle_idx, doppler_idx, range_idx, power[doppler_idx, range_idx]))

    if peak_extraction:
        hits = _local_maxima(cube, hits, n_angle, n_doppler, n_range)

    cloud = RadarPointCloud(cube.range_bins.max())
    for angle_idx, doppler_idx, range_idx, power in hits:
        azimuth, zenith = cube.angle_bins[angle_idx]
        cloud.add_point(
            PointDetection.FromSpherical(
                zenith, azimuth, cube.range_bins[range_idx],
                cube.velocity_bins[doppler_idx], power,
            )
        )
    return cloud


def _window(centre, halfwidth, n):
    """Index range of halfwidth cells either side of centre, clipped to [0, n)."""
    return slice(max(0, centre - halfwidth), min(n, centre + halfwidth + 1))


def _beam_noise_power(beam):
    """Thermal noise power per cell in one beam's [doppler, range] plane.

    Every cell of a beam sees the same noise power -- the injected noise is white and
    everything from there to the cube is linear -- but not every beam does, since each
    is scaled by its element gain, so this is measured per beam.

    |x|^2 of complex Gaussian noise is exponential, whose median is N*ln2. The median
    over the whole plane therefore reads the noise floor while ignoring the targets and
    clutter in a minority of cells. Cells nulled by the zero-Doppler clutter guard are
    exactly zero and would drag it down, so they are dropped.
    """
    power = np.abs(beam).astype(np.float64) ** 2
    occupied = power[power > 0]
    if occupied.size == 0:
        return np.nan
    return float(np.median(occupied) / np.log(2))


def per_target_sensing_snr(
    cube,
    truths_raz,                       # [(range_m, azimuth_rad, zenith_rad)] per target
    truths_radial_velocity_mps,       # [v_r] per target, +ve = receding
    search_halfwidth=(1, 1),          # (doppler, range) cells searched around the truth cell
):
    """Two-way sensing SNR (BS -> drone -> BS) in dB per ground-truth target: the target's
    signal power in its own (angle, range, Doppler) resolution cell over the thermal
    noise power in that cell. This is the measured realisation of the radar-equation SNR,
    so it obeys the monostatic 1/R^4 and is the sensing counterpart of the comms link's
    one-way SNR.

    The cell holds signal PLUS noise, so the noise power is subtracted from the peak to
    get the signal power; both are measured off the cube, in the same units, at the same
    point in the chain. Measured per beam rather than after the detector's MAX over
    angle, so it is defined for missed targets too -- but a target can read positive here
    and still be missed, since detect_targets() collapses angle first.

    Single-look, so it is only trustworthy where the target is: unbiased to ~0.3 dB
    above 15 dB, +/-3 dB by 10 dB, and biased high below ~3 dB, where the search window
    maxes over 9 cells and mostly finds the luckiest noise cell. Read low values as
    "at the floor", not as a number.

    Returns dB per target, aligned with `truths_raz`. -inf means the peak sits at or
    below the noise floor, i.e. no signal power is measurable; nan means the beam gave no
    usable noise estimate.
    """
    range_bins = np.asarray(cube.range_bins, dtype=float)
    velocity_bins = np.asarray(cube.velocity_bins, dtype=float)
    angle_bins = np.asarray(cube.angle_bins, dtype=float)          # [n_angle, 2] (az, zenith)
    n_doppler, n_range = cube.data.shape[1:]

    def direction(azimuth, zenith):
        return np.array([np.sin(zenith) * np.cos(azimuth),
                         np.sin(zenith) * np.sin(azimuth),
                         np.cos(zenith)])

    # Unit vector per scan direction, so the nearest beam is a max dot product and
    # azimuth wraparound can't pick a bin 359 degrees away.
    beam_dirs = np.stack([direction(az, ze) for az, ze in angle_bins])   # [n_angle, 3]

    noise_cache = {}
    snrs = []
    for (range_t, azimuth_t, zenith_t), velocity_t in zip(truths_raz, truths_radial_velocity_mps):
        angle_idx = int(np.argmax(beam_dirs @ direction(azimuth_t, zenith_t)))
        beam = cube.data[angle_idx]
        d_idx = int(np.argmin(np.abs(velocity_bins - velocity_t)))
        r_idx = int(np.argmin(np.abs(range_bins - range_t)))

        # Peak over a small neighbourhood: a truth straddling a bin edge smears into
        # its neighbour.
        search = beam[_window(d_idx, search_halfwidth[0], n_doppler),
                      _window(r_idx, search_halfwidth[1], n_range)]
        peak = float((np.abs(search).astype(np.float64) ** 2).max())

        if angle_idx not in noise_cache:
            noise_cache[angle_idx] = _beam_noise_power(beam)
        noise = noise_cache[angle_idx]

        if not np.isfinite(noise) or noise <= 0:
            snrs.append(float("nan"))
        elif peak <= noise:
            snrs.append(float("-inf"))
        else:
            snrs.append(float(10 * np.log10((peak - noise) / noise)))
    return snrs


@dataclass
class SensingMetrics:
    """Aggregate sensing accuracy over a run of frames -- the counterpart to the comms
    SNR/BER/throughput printed per drop. Computed by :func:`score_sensing`; its
    ``__str__`` is the printable summary."""
    num_frames: int
    num_truth: int              # ground-truth targets summed over all frames
    num_detections: int         # CFAR detections summed over all frames
    num_detected: int           # truths with >=1 detection inside their gate (true positives)
    num_false_alarms: int       # detections not inside ANY truth's gate
    prob_detection: float       # num_detected / num_truth -- "detection accuracy"
    false_alarms_per_frame: float
    false_alarm_fraction: float  # num_false_alarms / num_detections (share of hits that are spurious)
    mean_position_error_m: float   # mean over detected truths of the horizontal (x,y) miss distance
    rmse_position_error_m: float
    mean_range_error_m: float
    mean_azimuth_error_deg: float

    def __str__(self) -> str:
        pd = self.prob_detection
        return (
            "Sensing accuracy (over {nf} frame(s), {nt} target(s)):\n"
            "  Detection accuracy (P_d)   : {pd:.1%}  ({ndet}/{nt} targets detected)\n"
            "  False alarm rate           : {fapf:.2f}/frame  ({faf:.1%} of {ndets} detections spurious)\n"
            "  Position accuracy (RMSE)   : {rmse:.2f} m   (mean {mean:.2f} m horizontal miss)\n"
            "    range error (mean)       : {rerr:.2f} m\n"
            "    azimuth error (mean)     : {aerr:.2f} deg"
        ).format(
            nf=self.num_frames, nt=self.num_truth, pd=pd, ndet=self.num_detected,
            fapf=self.false_alarms_per_frame, faf=self.false_alarm_fraction, ndets=self.num_detections,
            rmse=self.rmse_position_error_m, mean=self.mean_position_error_m,
            rerr=self.mean_range_error_m, aerr=self.mean_azimuth_error_deg,
        )


def _detection_range_azimuth(detections):
    """[(range_m, azimuth_rad)] per point in a RadarPointCloud (or a plain list of its
    .points), read off the Cartesian position as plot_range_angle() does, so scores and
    plots agree."""
    points = detections.points if hasattr(detections, "points") else detections
    return [
        (float(np.linalg.norm(p.position)), float(np.arctan2(p.position[1], p.position[0])))
        for p in points
    ]


def score_sensing(
    cubes,
    ground_truth_positions,
    bs_position,
    detections_per_frame=None,
    range_gate_m=8.0,
    azimuth_gate_deg=8.0,
    detect_kwargs=None,
):
    """Score a run of sensing frames against ground truth: detection accuracy (P_d),
    false alarm rate, and position accuracy. Returns a :class:`SensingMetrics`.

    cubes: list [T] of RadarCube, one per timestep.
    ground_truth_positions: list [T], each a list of that frame's target world
        positions [3], converted to (range, azimuth) via world_to_spherical().
    bs_position: [3] radar world position.
    detections_per_frame: optional list [T] of already-computed RadarPointCloud. If
        None, detect_targets(cube, **detect_kwargs) runs per frame, so scoring does not
        depend on any plotting having run.

    Association is per frame and non-exclusive, tolerating CFAR reporting a cluster of
    adjacent cells for one target:
      * a target counts as detected if >=1 detection falls within range_gate_m AND
        azimuth_gate_deg of it; P_d = detected targets / total targets.
      * a detection in no target's gate is a false alarm, reported per frame and as a
        fraction of all detections.
      * position error uses each detected target's nearest in-gate detection; the miss
        distance is horizontal (x, y) Euclidean, with range and azimuth broken out.

    The gates are association windows, not the sensor's resolution. The defaults are a
    few range bins and about one beamwidth, so a hit has to land in roughly the right
    resolution cell; widen them if you only care about gross localization.
    """
    if detect_kwargs is None:
        detect_kwargs = {}
    az_gate = np.deg2rad(azimuth_gate_deg)

    num_frames = len(cubes)
    num_truth = num_detections = num_detected = num_false_alarms = 0
    pos_errs, range_errs, az_errs = [], [], []

    for f, cube in enumerate(cubes):
        truths = [world_to_spherical(bs_position, pos)[:2] for pos in ground_truth_positions[f]]  # (r, az)
        dets = detections_per_frame[f] if detections_per_frame is not None else detect_targets(cube, **detect_kwargs)
        det_ra = _detection_range_azimuth(dets)

        num_truth += len(truths)
        num_detections += len(det_ra)

        det_matched = [False] * len(det_ra)   # fell in some target's gate?
        for (r_t, az_t) in truths:
            best = None  # (normalized_dist, r_d, az_d)
            for j, (r_d, az_d) in enumerate(det_ra):
                d_r = r_d - r_t
                d_az = np.arctan2(np.sin(az_d - az_t), np.cos(az_d - az_t))  # wrapped
                if abs(d_r) <= range_gate_m and abs(d_az) <= az_gate:
                    det_matched[j] = True
                    nd = np.hypot(d_r / range_gate_m, d_az / az_gate)
                    if best is None or nd < best[0]:
                        best = (nd, r_d, az_d)
            if best is not None:
                num_detected += 1
                _, r_d, az_d = best
                # horizontal (x, y) miss distance -- range and cross-range together
                dx = r_d * np.cos(az_d) - r_t * np.cos(az_t)
                dy = r_d * np.sin(az_d) - r_t * np.sin(az_t)
                pos_errs.append(float(np.hypot(dx, dy)))
                range_errs.append(abs(r_d - r_t))
                az_errs.append(abs(np.rad2deg(np.arctan2(np.sin(az_d - az_t), np.cos(az_d - az_t)))))

        num_false_alarms += det_matched.count(False)

    pos_errs = np.asarray(pos_errs)
    return SensingMetrics(
        num_frames=num_frames,
        num_truth=num_truth,
        num_detections=num_detections,
        num_detected=num_detected,
        num_false_alarms=num_false_alarms,
        prob_detection=(num_detected / num_truth) if num_truth else float("nan"),
        false_alarms_per_frame=(num_false_alarms / num_frames) if num_frames else float("nan"),
        false_alarm_fraction=(num_false_alarms / num_detections) if num_detections else float("nan"),
        mean_position_error_m=float(pos_errs.mean()) if pos_errs.size else float("nan"),
        rmse_position_error_m=float(np.sqrt((pos_errs ** 2).mean())) if pos_errs.size else float("nan"),
        mean_range_error_m=float(np.mean(range_errs)) if range_errs else float("nan"),
        mean_azimuth_error_deg=float(np.mean(az_errs)) if az_errs else float("nan"),
    )


def plot_range_angle(cube, ground_truth=None, detections=None, max_range_m=None, title=None, ax=None,
                      boresight_azimuth_rad=0.0):
    """
    Range-azimuth heatmap (power reduced by MAX over zenith and doppler bins), with
    optional ground-truth (range_m, azimuth_rad) points and/or CFAR detections
    overlaid -- a top-down view of what the array sees, complementary to
    plot_range_doppler()'s range-velocity view.

    ground_truth: (range_m, azimuth_rad) tuple or list of them, typically the first
    two elements of world_to_spherical(bs_position, target_position).
    detections: a RadarPointCloud from detect_targets(cube), or a list of its .points.

    boresight_azimuth_rad: world-frame azimuth the array's boresight points at (e.g.
    monte_carlo.py's _boresight_azimuth_rad). Subtracted from every plotted azimuth, so
    0 deg on the axis means dead ahead. Defaults to 0, i.e. world-frame azimuth.
    """
    azimuths_world = np.unique(cube.angle_bins[:, 0])
    zeniths = np.unique(cube.angle_bins[:, 1])

    # Collapse Doppler and zenith by MAX, not sum: a target occupies a single Doppler
    # bin, so summing buries it under ~10*log10(N_doppler) dB of noise from the rest.
    # sense_snapshot() already zeroes the clutter band, so those bins never win.
    power = np.abs(cube.data) ** 2  # [angle, doppler, range]
    range_azimuth = power.max(axis=1).reshape(len(azimuths_world), len(zeniths), -1).max(axis=1)  # [az, range]

    # Re-centre, then re-sort: the shift can move the wrap point into the middle of the
    # azimuth grid, so range_azimuth's rows must follow the same permutation.
    azimuths = _center_rad(azimuths_world, boresight_azimuth_rad)
    az_order = np.argsort(azimuths)
    azimuths = azimuths[az_order]
    range_azimuth = range_azimuth[az_order, :]

    range_mask = cube.range_bins <= max_range_m if max_range_m is not None else np.ones_like(cube.range_bins, dtype=bool)
    range_bins = cube.range_bins[range_mask]
    range_azimuth = range_azimuth[:, range_mask]

    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=(10, 5))

    mesh = ax.pcolormesh(range_bins, np.rad2deg(azimuths), 10 * np.log10(range_azimuth + 1e-30), shading="auto")
    plt.colorbar(mesh, ax=ax, label="Power (dB)")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Azimuth relative to boresight (deg)" if boresight_azimuth_rad else "Azimuth (deg)")
    ax.set_title(title or "Range-Azimuth Map")

    if ground_truth is not None:
        truths = ground_truth if isinstance(ground_truth, list) else [ground_truth]
        gt_az_deg = [np.rad2deg(_center_rad(t[1], boresight_azimuth_rad)) for t in truths]
        ax.scatter([t[0] for t in truths], gt_az_deg,
                   marker="+", color="lime", s=100, label="Ground truth")

    if detections is not None:
        points = detections.points if hasattr(detections, "points") else detections
        det_range = [float(np.linalg.norm(p.position)) for p in points]
        det_az = [np.rad2deg(_center_rad(float(np.arctan2(p.position[1], p.position[0])), boresight_azimuth_rad)) for p in points]
        ax.scatter(det_range, det_az,
                   marker="x", color="red", s=60, label="CFAR detections")

    if ground_truth is not None or detections is not None:
        ax.legend(loc="upper right")

    if create_fig:
        plt.tight_layout()
        plt.show()

def plot_range_doppler(cube, max_range_m=None, title=None, ax=None, ground_truth=None):
    """Range-Doppler heatmap, straight from HermesPy's RadarCube.plot_range_velocity().
    Prints the detected peak and, if given, the ground-truth range/velocity beside it.

    ground_truth: optional (range_m, velocity_mps) tuple, or a list of them (one per
    target) from ground_truth_range_velocity().
    """
    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=(10, 5))

    result = cube.plot_range_velocity(axes=ax, scale="velocity")
    plt.colorbar(result.mesh, ax=ax, label="Normalized Power")

    if max_range_m is not None:
        ax.set_xlim(0, max_range_m)

    ax.set_title(title or "Range-Doppler Map")

    label = f"{title}: " if title else ""
    peak_range, peak_velocity = peak_range_velocity(cube, max_range_m=max_range_m)
    print(f"{label}peak  R={peak_range:.1f} m, v={peak_velocity:+.1f} m/s")
    if ground_truth is not None:
        truths = ground_truth if isinstance(ground_truth, list) else [ground_truth]
        for gt_range, gt_velocity in truths:
            print(f"{label}truth R={gt_range:.1f} m, v={gt_velocity:+.1f} m/s")

    if create_fig:
        plt.tight_layout()
        plt.show()
