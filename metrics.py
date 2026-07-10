def estimate_channel(H, #true per-user channel matrix [num_rx, num_tx_ant]
                     csi_error_std = 0.0 # stddev of per-entry complex Gaussian CSI error
                     ):
    """Stand-in CSI estimator - placeholder for a future sensing-based estimate
    (e.g. ISAC position sensing -> re-derived H_est with structured, geometry-driven
    error instead of this i.i.d. noise). rzf_precoder() doesn't care which one fed it.
    """
    import numpy as np

    if csi_error_std <= 0:
        return H
    noise = (np.random.normal(size=H.shape) + 1j * np.random.normal(size=H.shape)) / np.sqrt(2)
    return H + csi_error_std * noise


def rzf_precoder(H_est, #estimated per-user channel matrix [num_rx, num_tx_ant]
                 alpha = 0.0 # regularization (0 = pure ZF, >0 = RZF/MMSE, interpolates toward MRT as alpha grows)
                 ):
    """Regularized zero-forcing / MMSE precoder, built only from an (estimated) channel
    matrix - agnostic to how H_est was obtained (synthetic noise today, sensing-derived
    later). alpha=0 reduces to pinv(H_est) (pure ZF); larger alpha trades interference
    nulling for robustness to errors in H_est, approaching MRT as alpha -> inf.
    """
    import numpy as np

    num_rx = H_est.shape[0]
    W = H_est.conj().T @ np.linalg.inv(H_est @ H_est.conj().T + alpha * np.eye(num_rx))  # [num_tx_ant, num_rx]
    return W / np.linalg.norm(W, axis=0, keepdims=True)  # unit-norm beamforming vector per user


def compute_zf_precoder(paths, #PathSolver()
                        t = 0, # which transmitter to compute precoding weights for
                        alpha = 0.0, # passed to rzf_precoder
                        csi_error_std = 0.0, # passed to estimate_channel
                        rx_indices = None, # subset of receiver indices to include; None = all
                        ):

    import numpy as np

    a_np = np.array(paths.a)           # [2, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
    a_complex = a_np[0] + 1j * a_np[1] # [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
    rows = rx_indices if rx_indices is not None else list(range(a_complex.shape[0]))

    # --- per-user channel vector: sum over rx antennas and paths, keep tx antennas ---
    # this folds in NLOS paths (reflections/diffraction) at whatever angle they
    # arrive from automatically - no manual look_at / angle reasoning needed
    # NOTE: collapsing each user's rx antennas into one row means ZF only nulls this
    # summed direction, not every individual rx-antenna channel - with per-antenna MRC
    # at the receiver (see compute_kpis), some residual inter-user interference can remain
    H = np.stack([
        np.sum(a_complex[r, :, t, :, :], axis=(0, -1))  # [num_tx_ant]
        for r in rows
    ], axis=0)  # [num_rx, num_tx_ant]

    # compute_kpis always scores W against this true H, so any gap introduced by
    # estimate_channel (synthetic now, sensing-derived later) shows up as real loss
    H_est = estimate_channel(H, csi_error_std)
    return rzf_precoder(H_est, alpha)


def compute_channel_gain(paths, #PathSolver()
                          r = 0, # which receiver to compute gain for
                          t = 0, # which transmitter to compute gain for
                          precoding_vector=None # zero-forcing weights for this receiver, shape [num_tx_ant]
                          ):
    """Effective channel gain g_r for receiver r, shared by compute_kpis (for SNR)
    and allocate_power_channel_inversion (for per-user power allocation).
    """
    import numpy as np

    a_np = np.array(paths.a)           # [2, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
    a_complex = a_np[0] + 1j * a_np[1] # [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]

    if precoding_vector is None:
        # --- fixed/uniform excitation: unit-norm equivalent of "ones" ---
        num_tx_ant = a_complex.shape[3]
        h = np.sum(a_complex[r, :, t, :, :], axis=(-2, -1))  # sum over tx_ant and paths -> scalar per rx_ant
        return np.sum(np.abs(h)**2) / num_tx_ant  # normalize for the unit-norm equivalent of "ones"
    else:
        # --- zero-forcing: per-rx-antenna channel projected onto precoding weights, then MRC across rx antennas ---
        H_r = np.sum(a_complex[r, :, t, :, :], axis=-1)  # sum over paths -> [num_rx_ant, num_tx_ant]
        h = H_r @ precoding_vector                       # [num_rx_ant]
        return np.sum(np.abs(h)**2)


def allocate_power_channel_inversion(gains, #per-user channel gains g_r, shape [num_rx]
                                      tx_power_dbm=30 #total transmit power to split across users
                                      ):
    """Channel-inversion power allocation: P_r = P_total * (1/g_r) / sum(1/g_j).
    Gives more power to users with weak channels and less to users with strong
    channels, equalizing SINR across users (noise power is the same for everyone here).
    """
    import numpy as np
    from sionna.rt.utils import dbm_to_watt

    inv_gains = 1.0 / np.asarray(gains)
    return dbm_to_watt(tx_power_dbm) * inv_gains / np.sum(inv_gains)  # [num_rx], watts


def compute_channel_gain_matrix(paths, #PathSolver()
                                 t = 0, # which transmitter
                                 precoding_vectors=None, # [num_tx_ant, num_rx] columns = per-user precoding vectors; None = no per-user beams
                                 rx_indices=None, # subset of receiver indices; None = all
                                 ):
    """Cross-gain matrix G[r, j]: signal power receiver r picks up from the beam
    intended for user j (MRC-combined over rx antennas). The diagonal G[r, r] is
    each user's own signal gain (same as compute_channel_gain); off-diagonals
    G[r, j] for j != r are inter-user interference - ~0 under ideal ZF, non-zero
    once CSI error makes the nulling imperfect. Needed for true SINR
    (signal / (noise + interference)).

    With precoding_vectors=None there are no per-user beams (single shared
    broadcast), so G is diagonal and there is no inter-user interference.
    """
    import numpy as np

    a_np = np.array(paths.a)           # [2, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
    a_complex = a_np[0] + 1j * a_np[1] # [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
    rows = rx_indices if rx_indices is not None else list(range(a_complex.shape[0]))

    if precoding_vectors is None:
        num_tx_ant = a_complex.shape[3]
        h = np.sum(a_complex[np.ix_(rows), :, t, :, :].squeeze(0), axis=(-2, -1))  # [|rows|, num_rx_ant]
        gains = np.sum(np.abs(h)**2, axis=-1) / num_tx_ant
        return np.diag(gains)

    H = np.sum(a_complex[np.ix_(rows), :, t, :, :].squeeze(0), axis=-1)  # [|rows|, num_rx_ant, num_tx_ant]
    HW = np.einsum('rat,tj->raj', H, precoding_vectors)                   # [|rows|, num_rx_ant, num_rx]
    return np.sum(np.abs(HW)**2, axis=1)                                   # [|rows|, num_rx]


def compute_kpis(paths, #PathSolver()
                 radio_map, #RadioMap()
                 scene,
                 bandwidth=100e6, #100mhz
                 tx_power_dbm=30, #1W, used only if tx_power_w is not given
                 tx_power_w=None, #per-user tx power in watts, e.g. from allocate_power_channel_inversion; overrides tx_power_dbm
                 noise_figure_db=7, #receiver noise
                 coverage_threshold_dbm=-80, #what is considered covered
                 r = 0, # which receiver do we want metrics for
                 t = 0, # which transmitter do we want metrics for
                 precoding_vector=None, # zero-forcing weights for this receiver, shape [num_tx_ant]
                 g=None, # own-signal channel gain for receiver r; if None, computed from paths/precoding_vector
                 interference_w=0.0 # power (W) receiver r picks up from OTHER users' beams, e.g. from compute_channel_gain_matrix
                 ):

    import numpy as np
    from sionna.rt.utils import dbm_to_watt, watt_to_dbm
    from scipy.special import erfc

    if g is None:
        g = compute_channel_gain(paths, r=r, t=t, precoding_vector=precoding_vector)

    if tx_power_w is None:
        tx_power_w = dbm_to_watt(tx_power_dbm) / len(scene.receivers)  # equal power split across users

    thermal_noise_w = scene.thermal_noise_power * (bandwidth / scene.bandwidth)
    noise_figure_lin = 10 ** (noise_figure_db / 10)
    noise_power_w = thermal_noise_w * noise_figure_lin

    # SINR = own signal / (thermal noise + interference from other users' beams)
    sinr_linear = (np.squeeze(np.maximum((tx_power_w * g) / (noise_power_w + interference_w), 0)))
    sinr_db = (10 * np.log10(sinr_linear))

    # closed form BER for coherent bpsk over awgn - BER = 1/2 erfc (sqrt(eta))
    ber = 0.5 * erfc(np.sqrt(sinr_linear))

    # shannon capacity
    throughput_mbps = bandwidth * np.log2(1 + sinr_linear) / 1e6 #div by 1e6 to get mbps instead of bps

    rm_path_gain = radio_map.path_gain.numpy()
    rss_dbm = 10 * np.log10(np.maximum(rm_path_gain * tx_power_w, 1e-30) / 1e-3)
    coverage_fraction = float(np.mean(rss_dbm > coverage_threshold_dbm))

    return {
        "sinr_db":           sinr_db,
        "ber":               float(ber),
        "throughput_mbps":   round(float(throughput_mbps), 2),
        "coverage_fraction": round(coverage_fraction, 4),
        "tx_power_dbm":      round(float(watt_to_dbm(tx_power_w)), 2),
    }


def plot_kpis(kpi_log, scene=None):
    """Plot SNR, BER, throughput, and coverage over time from the kpi_log returned by simulate().

    If scene is given, each drone's line uses that drone's color as set in the scene.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    drone_names = list(kpi_log[0].keys())
    steps = np.arange(len(kpi_log))
    colors = {name: scene.receivers[name].color for name in drone_names} if scene else {}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for name in drone_names:
        color = colors.get(name)
        axes[0, 0].plot(steps, [step[name]["sinr_db"] for step in kpi_log], marker="o", label=name, color=color)
        axes[0, 1].plot(steps, [step[name]["ber"] for step in kpi_log], marker="o", label=name, color=color)
        axes[1, 0].plot(steps, [step[name]["throughput_mbps"] for step in kpi_log], marker="o", label=name, color=color)

    # coverage_fraction comes from the shared radio map, so it's the same for every drone at a given step
    coverage = [next(iter(step.values()))["coverage_fraction"] for step in kpi_log]
    axes[1, 1].plot(steps, coverage, marker="o", color="black")

    axes[0, 0].set_title("SINR (dB)")

    axes[0, 1].set_title("BER (BPSK)")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylim(1e-6, 0.6)  # below 1e-6 FEC handles it; 0.6 shows the noise floor

    axes[1, 0].set_title("Throughput (Mbps)")
    axes[1, 0].set_ylim(0, 1000)    # Shannon cap at ~30 dB SINR with 100 MHz BW

    axes[1, 1].set_title("Coverage Fraction")
    axes[1, 1].set_ylim(0, 1)

    for ax in axes.flat:
        ax.set_xlabel("Time step")
        ax.grid(True)

    for ax in axes.flat[:3]:
        ax.legend()

    fig.tight_layout()
    plt.show()

# use tensors - extract what is what position, use them correctly - check
# implement fancy position changing - shaboing
# parallelization - done mostly
# documentation -
# plotting


# downlink transmission
# beamforming - done (zero-forcing precoder, see compute_zf_precoder)
# track da drone - if sionna has strongest beam then go along
# power as a function of snr
# plot kpis
# ofdm, broadband

# make plots look better, box plots or other stuff

# figure out sionna's waveforms and hermespy waveforms, which to use, the integration between the two, add meshes

# prefix slide with all the settings and stuff
# should be able to see buildings
# plot the ground truth vs what is seen
# directional beam needs to be wider
# implement ofdm if possible
# try different waveforms to see the accuracy differences

# key metrics : accuracy, resolution, probability of detection