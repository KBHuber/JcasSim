"""
A unified JCAS drop: one Sionna ray-trace solve, N transmitted OFDM frames (one
per served drone) precoded and summed onto the same 64-element array, propagated
through one geometric channel to both a monostatic radar receiver and every
served drone's comms receiver. Yields a RadarCube (Sturm & Wiesbeck) and, per
drone, a decoded comms link -- bit errors measured from the decode, not from a
closed-form SINR.

Unlike sense_snapshot() (sensing.py), which senses off its own solve, the radar
illumination here IS the downlink: one PathSolver call covers the radar link and
every drone's comms link (the scene shares rx_array = tx_array, so every drone
Receiver and the radar Receiver use the identical array).

MULTI-USER
----------
GeometricChannelResponse.propagate() is linear, so propagating a sum of N
independently-precoded streams is the same as propagating each and summing. A
composite ``X = sum_i W_i (x) s_i`` (compute_zf_precoder returns a jointly-nulled
multi-user W) therefore produces true multi-user interference at every receiver
with no separate interference model.

The radar's Sturm & Wiesbeck estimator divides by one known reference frame's
symbols -- rx_names[0]'s. Every other drone's stream shows up in the radar's
received signal as additive self-interference, the usual JCAS noise-floor
elevation.

WHAT'S SIMPLIFIED
-----------------
- Each drone combines with one frequency-flat matched-filter vector steered at
  the known BS direction, not per-subcarrier digital MRC across 8 chains. This
  matches a one-RF-combiner drone architecture.
- Power is split equally across served drones (1/sqrt(N) per unit-norm stream),
  so tx_power_w means "total array budget" for any N. No channel-inversion-style
  unequal allocation.
- Equalizer channel knowledge comes from the ray-traced paths (optionally
  corrupted by csi_error_std), not from OFDM pilot symbols.

    drop = run_jcas_drop(scene, bs_position=[-100,0,20],
                          look_at=[-40,5,12], rx_names=["d0", "d1"])
    drop.radar_cube                     # RadarCube, as from sense_snapshot()
    drop.comms["d0"].ber, drop.comms["d0"].throughput_mbps
    drop.comms["d0"].snr_db             # measured EVM-based effective SINR
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light

import mitsuba as mi
import drjit as dr
import sionna.rt as rt
from sionna.rt.utils import rotation_matrix

from hermespy.core import Signal, Transformation
from hermespy.simulation import SimulatedDevice
from hermespy.simulation.antennas import SimulatedCustomArray, SimulatedIdealAntenna
from hermespy.beamforming import ConventionalBeamformer
from hermespy.modem import StatedSymbols, ZeroForcingChannelEqualization

from geometric_sionna_channel import GeometricSionnaChannel
from sensing import (
    _build_ofdm_radar, _ofdm_radar_cube_from_received, _array_beamwidth_deg,
    _hamming_axis, world_to_spherical,
)
from metrics import compute_zf_precoder, estimate_channel
import ofdm_config


# Raw (pre-decoding) BER below which a rate-1/2 LDPC/turbo code recovers the frame,
# i.e. the BLER 0/1 boundary in run_jcas_drop()'s throughput. No FEC is simulated, so
# this stands in for a real decoder's waterfall region; 1e-2 is the conventional
# operating point. Raise it to model a stronger code, lower it to be conservative.
FEC_BER_THRESHOLD = 1e-2


class CommsResult:
    """One served drone's measured comms outcome: bits in, bits out, after noise
    and multi-user interference."""

    def __init__(self, tx_bits, rx_bits, ber, throughput_mbps, snr_db, precoding_vector):
        self.tx_bits = tx_bits
        self.rx_bits = rx_bits
        self.ber = ber
        self.throughput_mbps = throughput_mbps
        self.snr_db = snr_db
        self.precoding_vector = precoding_vector


class JCASDropResult:
    """The two outputs of one JCAS drop: the monostatic RadarCube (which sees every
    served drone's stream as illumination) and each drone's decoded comms link."""

    def __init__(self, radar_cube, comms):
        self.radar_cube = radar_cube
        self.comms = comms   # dict[str, CommsResult], keyed by drone name


def _add_thermal_noise(received, noise_w):
    """Add circularly-symmetric complex Gaussian noise of total power `noise_w`, in
    place, to a [rx_ant, T] receive buffer carrying Watts-scale amplitude."""
    received += np.sqrt(noise_w / 2) * (
        np.random.normal(size=received.shape) + 1j * np.random.normal(size=received.shape)
    )
    return received


def run_jcas_drop(
    scene,                        # scene already containing every served drone Receiver
                                   # (+ mesh, if radar should see it as a target)
    bs_position,                  # [3] BS position, world-frame meters
    look_at,                      # [3] BS array boresight, radar and transmit alike
    rx_names,                     # drone Receiver names in `scene` to serve; rx_names[0]
                                   # doubles as the radar's Sturm & Wiesbeck reference frame
    precoding_vectors=None,       # [64, N] ZF weights, columns in rx_names order;
                                   # None -> metrics.compute_zf_precoder, jointly nulled
    tx_power_w=1.0,                # total array tx power, split equally across drones
    csi_error_std=0.0,            # stddev of CSI error for the RX equalizer's channel
                                   # knowledge; propagation itself is always exact
    noise_figure_db=7.0,
    max_num_paths_per_src=5000,
    samples_per_src=1_000_000,    # SBR rays launched per transmitter (Sionna's default).
                                   # Sets whether the radar sees a target at all: expected
                                   # hits on a body of radius r at range R go as
                                   # samples_per_src * r^2 / (4 R^2), so a 0.25 m drone
                                   # gets ~25 hits at 25 m but ~0.4 at 200 m. Cost is
                                   # roughly linear; drone radius is the cheaper knob.
    azimuth_range_deg=(-180.0, 180.0),
    num_azimuth_bins=None,
    zenith_range_deg=(0.0, 90.0),
    num_zenith_bins=None,
    angular_oversample=3,
    max_range_m=None,             # trim the radar cube's range axis to the instrumented
                                   # range; see sensing._ofdm_radar_cube_from_received
    seed=41,
) -> JCASDropResult:
    """Solve once, transmit once per served drone (summed onto one composite array
    excitation), propagate once per receiver; returns a radar cube and a decoded
    comms link per drone. See the module docstring for what is simplified.
    """

    bs_position = np.asarray(bs_position, dtype=float)
    carrier_frequency = ofdm_config.CARRIER_FREQUENCY
    bandwidth = ofdm_config.BANDWIDTH
    wavelength = speed_of_light / carrier_frequency
    num_users = len(rx_names)

    scene.frequency = carrier_frequency
    # Drop a previous call's bs_tx/bs_rx without touching the drone Receivers: unlike
    # sense_snapshot()'s scene, this one is shared with the comms drones.
    if "bs_tx" in scene.transmitters:
        del scene._transmitters["bs_tx"]
    if "bs_rx" in scene.receivers:
        del scene._receivers["bs_rx"]
    bs_tx = rt.Transmitter("bs_tx", bs_position.tolist())
    bs_rx = rt.Receiver("bs_rx", bs_position.tolist())
    scene.add(bs_tx)
    scene.add(bs_rx)
    bs_tx.look_at(look_at)
    bs_rx.look_at(look_at)

    # One channel: channel.realize() runs the single PathSolver solve covering the
    # monostatic radar link and every drone's comms link. Each link below is then a
    # cheap realization.sample() against that cached solve, not a fresh trace.
    channel = GeometricSionnaChannel(
        scene, gain=tx_power_w, seed=seed, max_depth=3, max_num_paths_per_src=max_num_paths_per_src,
        monostatic_min_range_m=3.0, samples_per_src=samples_per_src,
    )
    realization = channel.realize()
    paths = realization.paths

    scene_rx_names = list(scene.receivers.keys())
    drone_indices = [scene_rx_names.index(name) for name in rx_names]

    if precoding_vectors is None:
        # Multi-user ZF: compute_zf_precoder inverts H H^H over all rows at once when
        # given several rx_indices, so this is jointly nulled, not N single-user precoders.
        precoding_vectors = compute_zf_precoder(paths, rx_indices=drone_indices)   # [64, N]

    # BS array geometry, for the radar operator and the beamform scan. rx_array =
    # tx_array, so every receiver in the scene shares it.
    rx_pos_local = np.array(scene.rx_array.positions(wavelength)).T  # [rx_ant, 3]
    R_bs_rx = np.array(rotation_matrix(bs_rx.orientation))[:, :, 0]
    rx_pos_world_bs = rx_pos_local @ R_bs_rx.T

    # One independent OFDM frame per served drone, generated via radar.transmit() as
    # sense_snapshot() does, so it is exactly the frame Sturm & Wiesbeck normalization
    # expects. rx_names[0]'s (radar, device_state) is kept for the estimate below; the
    # others exist only for their one .transmit() call.
    radar_device, radar, device_state = _build_ofdm_radar(rx_pos_world_bs, bs_position, carrier_frequency, bandwidth)
    tx_transmission = radar.transmit(device_state)
    waveform = radar.waveform
    tx_iqs = [tx_transmission.signal.view(np.ndarray)[0]]
    tx_bits_list = [radar._OFDMRadar__last_transmission.bits]
    for _ in rx_names[1:]:
        _, extra_radar, extra_device_state = _build_ofdm_radar(rx_pos_world_bs, bs_position, carrier_frequency, bandwidth)
        extra_transmission = extra_radar.transmit(extra_device_state)
        tx_iqs.append(extra_transmission.signal.view(np.ndarray)[0])
        tx_bits_list.append(extra_radar._OFDMRadar__last_transmission.bits)

    # Composite per-antenna signal: N precoded frames summed onto the BS's elements.
    # The 1/sqrt(N) split keeps tx_power_w meaning "total array budget" for any N.
    tx_signal = sum(
        (precoding_vectors[:, i:i + 1] / np.sqrt(num_users)) * tx_iqs[i][np.newaxis, :]
        for i in range(num_users)
    )   # [64, T]

    # Drone RX devices: each comms link's receiver for realization.sample() below, and
    # reused for the RF combine in the COMMS section.
    drone_objs = {name: scene.receivers[name] for name in rx_names}
    drone_positions = {name: np.asarray(obj.position, dtype=float).reshape(3) for name, obj in drone_objs.items()}
    rx_pos_world_drones = {}
    combiner_devices = {}
    for name, obj in drone_objs.items():
        R_drone = np.array(rotation_matrix(obj.orientation))[:, :, 0]
        rx_pos_world_drone = rx_pos_local @ R_drone.T
        rx_pos_world_drones[name] = rx_pos_world_drone
        combiner_devices[name] = SimulatedDevice(
            antennas=SimulatedCustomArray(ports=[
                SimulatedIdealAntenna(pose=Transformation.From_Translation(p)) for p in rx_pos_world_drone
            ]),
            carrier_frequency=carrier_frequency, bandwidth=bandwidth, oversampling_factor=1,
            pose=Transformation.From_Translation(drone_positions[name]),
        )

    # N+1 links off one cached solve: realization.sample() matches each device's position
    # back to its Sionna tx/rx and applies the monostatic self-coupling guard only to the
    # bs_tx<-bs_rx link. Only the radar link is materialized here; drone links are sampled
    # one at a time in the COMMS loop and dropped after use, since each costs ~775 MB and
    # the Monte Carlo grid reaches ~126 drones.
    sample_radar = realization.sample(radar_device, radar_device, 0.0, carrier_frequency, bandwidth)

    tx_signal_model = Signal.Create(tx_signal, sampling_rate=bandwidth, carrier_frequency=carrier_frequency)
    received_radar = np.asarray(sample_radar.propagate(tx_signal_model).view(np.ndarray))

    # Thermal noise, physically scaled (received carries Watts-scale amplitude); applied
    # identically to the drone links as they are streamed in below.
    kTB = 1.38e-23 * 290 * bandwidth
    noise_w = kTB * 10 ** (noise_figure_db / 10)

    # A monostatic link can legitimately return zero paths -- nothing sent energy straight
    # back to the co-located tx/rx -- and HermesPy then hands back an empty Signal the
    # Sturm & Wiesbeck estimator cannot consume. Substitute a zero receive buffer so noise
    # is added as on any other link and the radar sees a noise-only cube.
    if received_radar.shape[1] == 0:
        num_radar_rx_ant = rx_pos_world_bs.shape[0]
        received_radar = np.zeros((num_radar_rx_ant, tx_signal.shape[1]), dtype=complex)
    _add_thermal_noise(received_radar, noise_w)

    # RX aperture taper (Hamming over both array axes), applied per element before the
    # scan and after the noise, so the taper weights it as real hardware would. Drops the
    # uniform combiner's ~-13 dB angular sidelobes to ~-34 dB at ~3.5 dB aperture loss.
    # Receive only: the transmit excitation is the ZF precoder, which cannot be tapered
    # without perturbing its nulls. Identical to sense_snapshot()'s taper.
    rx_taper = _hamming_axis(rx_pos_local[:, 1]) * _hamming_axis(rx_pos_local[:, 2])
    rx_taper *= len(rx_taper) / rx_taper.sum()  # sum(w) = N: preserve boresight gain
    received_radar *= rx_taper[:, np.newaxis]

    # ==================== RADAR ====================
    az_hpbw_deg, ze_hpbw_deg = _array_beamwidth_deg(rx_pos_local, wavelength)
    if num_azimuth_bins is None:
        num_azimuth_bins = max(2, round((azimuth_range_deg[1] - azimuth_range_deg[0]) / (az_hpbw_deg / angular_oversample)) + 1)
    if num_zenith_bins is None:
        num_zenith_bins = max(2, round((zenith_range_deg[1] - zenith_range_deg[0]) / (ze_hpbw_deg / angular_oversample)) + 1)
    azimuths = np.deg2rad(np.linspace(*azimuth_range_deg, num_azimuth_bins))
    zeniths = np.deg2rad(np.linspace(max(zenith_range_deg[0], ze_hpbw_deg / 2), zenith_range_deg[1], num_zenith_bins))
    az_grid, ze_grid = np.meshgrid(azimuths, zeniths, indexing="ij")
    angle_grid = np.stack([az_grid.ravel(), ze_grid.ravel()], axis=-1)

    directions_world = np.stack([
        np.sin(angle_grid[:, 1]) * np.cos(angle_grid[:, 0]),
        np.sin(angle_grid[:, 1]) * np.sin(angle_grid[:, 0]),
        np.cos(angle_grid[:, 1]),
    ], axis=-1)
    directions_local = directions_world @ R_bs_rx
    theta_local = np.arccos(np.clip(directions_local[:, 2], -1.0, 1.0))
    phi_local = np.arctan2(directions_local[:, 1], directions_local[:, 0])
    c_theta, c_phi = scene.rx_array.antenna_pattern.patterns[0](mi.Float(theta_local), mi.Float(phi_local))
    element_gain = np.array(dr.abs(c_theta) ** 2 + dr.abs(c_phi) ** 2)

    # radar.transmit() above already generated rx_names[0]'s frame, so
    # __last_transmission is set for Sturm normalization; the other drones' streams are
    # additive self-interference in received_radar (see module docstring).
    radar_cube = _ofdm_radar_cube_from_received(
        radar, device_state.receive_state(), received_radar, angle_grid, element_gain, bandwidth, carrier_frequency,
        max_range_m=max_range_m,
    )

    # ==================== COMMS ====================
    # One decode chain per served drone. Interference needs no explicit term: tx_signal is
    # the composite of every drone's precoded stream, so it is already in the receive.
    n_sym = waveform.words_per_frame
    symbol_period = (1 + ofdm_config.PREFIX_RATIO) / ofdm_config.SUBCARRIER_SPACING
    t_symbols = np.arange(n_sym) * symbol_period                              # [n_sym]
    freqs_k = ofdm_config.subcarrier_frequencies()                            # [K]
    frame_duration = n_sym * (1 + ofdm_config.PREFIX_RATIO) / ofdm_config.SUBCARRIER_SPACING

    comms = {}
    for i, name in enumerate(rx_names):
        precoding_vector = precoding_vectors[:, i]
        tx_bits = tx_bits_list[i]
        drone_pos = drone_positions[name]
        rx_pos_world_drone = rx_pos_world_drones[name]
        combiner_device = combiner_devices[name]

        # Sample and propagate this drone's link alone, released at the end of the
        # iteration -- see the sample_radar comment above.
        sample_drone = realization.sample(
            radar_device, combiner_device, 0.0, carrier_frequency, bandwidth
        )
        received_drone = _add_thermal_noise(
            np.asarray(sample_drone.propagate(tx_signal_model).view(np.ndarray)), noise_w
        )

        # A fully shadowed drone (no path within max_depth) gets an empty signal from
        # propagate() rather than a frame of zeros, which demodulate() cannot consume.
        # Report it as a failed link: no signal arrived, which is a real outcome.
        if received_drone.shape[1] == 0:
            comms[name] = CommsResult(
                tx_bits=tx_bits, rx_bits=np.zeros_like(tx_bits), ber=0.5,
                throughput_mbps=0.0, snr_db=float("-inf"), precoding_vector=precoding_vector,
            )
            continue

        # Frequency-flat matched-filter combine toward the BS (one RF chain, not 8 digital
        # receivers), via the same ConventionalBeamformer the radar scan uses.
        _, az_bs, ze_bs = world_to_spherical(drone_pos, bs_position)
        combiner_rx_state = combiner_device.state(0.0).receive_state()
        combiner = ConventionalBeamformer()
        focus = np.array([[az_bs, ze_bs]])
        combined_drone = combiner.probe(received_drone, combiner_rx_state, focus[:, np.newaxis, :])[0, 0, :]
        num_rx_ant = rx_pos_world_drone.shape[0]
        combined_drone = combined_drone * np.sqrt(num_rx_ant)  # probe()'s 1/N -> 1/sqrt(N)

        rx_placed = waveform.demodulate(combined_drone.reshape(1, -1), bandwidth, 1)
        raw_full = np.asarray(rx_placed.raw)          # [1, num_ofdm_symbols, num_subcarriers]

        # The exact weight vector applied above, so the equalizer's channel knowledge is
        # self-consistent with what was combined.
        codebook_row = combiner._codebook(carrier_frequency, focus, combiner_rx_state.antennas)[0]  # 1/N-normalized
        w_eff = codebook_row * num_rx_ant / np.sqrt(num_rx_ant)                                     # -> 1/sqrt(N)

        # Equalizer channel knowledge: this drone's true per-(subcarrier, OFDM-symbol)
        # channel with precoding and RX combining folded in, optionally corrupted by
        # csi_error_std. It is an a priori estimate independent of this frame's bits, so
        # imperfect CSI causes real decode errors.
        #
        # It varies along the slow-time axis, not just frequency: at these drone speeds
        # Doppler rotates the channel by several radians over one ~1.4 ms frame, so a
        # frequency-only estimate scrambles the constellation for later symbols.
        channel_drone = sample_drone.response
        # Two amplitude factors act on the signal but are not in the raw path gains `a`:
        # sqrt(tx_power_w), applied inside propagate() since the channel was built with
        # gain=tx_power_w, and the 1/sqrt(num_users) power split baked into tx_signal.
        # 16-QAM decides magnitude bits by amplitude, so a scale error costs every symbol
        # one of its four bits.
        scale = np.sqrt(tx_power_w / num_users)
        g_p = np.einsum("a,atp,t->p", w_eff, channel_drone.a, precoding_vector) * scale   # [P] per-path gain
        # Snap tau as propagate() did: it delays by whole samples with no fractional ramp,
        # and the exact geometric tau would be up to a sample off, a phase error growing
        # to +/-pi at the band edges.
        tau_quantized = np.floor(channel_drone.tau * bandwidth) / bandwidth        # [P]
        phase_delay = np.exp(-2j * np.pi * np.outer(freqs_k, tau_quantized))       # [K, P]
        phase_doppler = np.exp(2j * np.pi * np.outer(t_symbols, channel_drone.doppler))  # [n_sym, P]
        h_true_kn = np.einsum("kp,np,p->nk", phase_delay, phase_doppler, g_p)      # [n_sym, K]
        h_est_kn = estimate_channel(h_true_kn, csi_error_std)

        states_full = h_est_kn[np.newaxis, np.newaxis, :, :]                       # [1, 1, n_sym, K]
        stated = StatedSymbols(raw_full, np.ascontiguousarray(states_full))
        picked = waveform.pick(stated)
        equalized = ZeroForcingChannelEqualization().equalize_channel(picked)
        rx_bits = waveform.unmap(equalized)

        ber = float(np.mean(rx_bits != tx_bits))

        # Throughput T = R * (1 - BLER), the standard link-level form:
        #   R    = payload bits / frame duration (data subcarriers only, over the full
        #          frame, so the pilot comb and CP overhead are both accounted for)
        #   BLER = 0 or 1 for one frame, decided by whether the raw BER is inside what
        #          standard channel coding closes.
        # Not R * (1 - BER): correct bits at unknown positions carry no information, and
        # BER here is strongly bimodal, so the linear form would credit drones delivering
        # nothing with most of the link rate.
        bler = 0.0 if ber < FEC_BER_THRESHOLD else 1.0
        throughput_mbps = (len(tx_bits) / frame_duration / 1e6) * (1.0 - bler)

        # Post-equalization EVM-based SINR, off the same equalized symbols that produced
        # rx_bits, so it is consistent with the BER above. Signal power comes from the true
        # tx symbols, making the ratio independent of constellation normalization.
        #
        # It is a SINR, not an SNR: the error vector lumps in thermal noise, residual
        # multi-user interference and CSI mismatch. The attribute is named `snr_db` for
        # CSV-schema compatibility, but label it SINR on plots.
        tx_symbols = waveform.map(tx_bits).raw.flatten()
        rx_symbols = np.asarray(equalized.raw).flatten()
        err_power = float(np.mean(np.abs(rx_symbols - tx_symbols) ** 2))
        sig_power = float(np.mean(np.abs(tx_symbols) ** 2))
        snr_db = 10 * np.log10(sig_power / err_power) if err_power > 0 else float("inf")

        comms[name] = CommsResult(
            tx_bits=tx_bits, rx_bits=rx_bits, ber=ber,
            throughput_mbps=throughput_mbps, snr_db=snr_db, precoding_vector=precoding_vector,
        )

    return JCASDropResult(radar_cube=radar_cube, comms=comms)


def _ber_for_log(c):
    """A CommsResult's BER, floored for a log axis: zero bit errors only tells us
    BER < 1/num_bits, so plot that bound rather than dropping the point."""
    return c.ber if c.ber > 0 else 0.5 / len(c.tx_bits)


def plot_jcas_kpis(jcas_results, scene=None):
    """Plot measured effective SINR, BER, and throughput over time from the per-timestep
    list of JCASDropResult returned by run_jcas_drop(). One line per served drone; if
    scene is given, each line takes that drone's color from the scene.
    """

    drone_names = list(jcas_results[0].comms.keys())
    steps = np.arange(len(jcas_results))
    colors = {name: scene.receivers[name].color for name in drone_names} if scene else {}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for name in drone_names:
        color = colors.get(name)
        axes[0].plot(steps, [r.comms[name].snr_db for r in jcas_results], marker="o", label=name, color=color)
        axes[1].plot(steps, [_ber_for_log(r.comms[name]) for r in jcas_results], marker="o", label=name, color=color)
        axes[2].plot(steps, [r.comms[name].throughput_mbps for r in jcas_results], marker="o", label=name, color=color)

    axes[0].set_title("Measured effective SINR (dB, EVM-based)")
    axes[1].set_title("Measured BER (16-QAM, uncoded)")
    axes[1].set_yscale("log")
    axes[2].set_title("Measured throughput (Mbps)")

    for ax in axes:
        ax.set_xlabel("Time step")
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    plt.show()
