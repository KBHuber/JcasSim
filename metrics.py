def compute_kpis(paths, #PathSolver() 
                 radio_map, #RadioMap()
                 scene,
                 bandwidth=100e6, #100mhz
                 tx_power_dbm=30, #1W
                 noise_figure_db=7, #receiver noise
                 coverage_threshold_dbm=-50, #what is considered covered
                 r = 0, # which receiver do we want metrics for
                 t = 0 # which transmitter do we want metrics for
                 ):

    import drjit as dr
    import numpy as np
    from sionna.rt.utils import dbm_to_watt, watt_to_dbm
    from scipy.special import erfc



    a_np = np.array(paths.a)           # [2, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
    a_complex = a_np[0] + 1j * a_np[1] # [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]

    # Now indexing is correct
    h = np.sum(a_complex[r, :, t, :, :], axis=(-2, -1))  # sum over tx_ant and paths -> scalar per rx_ant
    g = np.sum(np.abs(h)**2)

    thermal_noise_w = scene.thermal_noise_power * (bandwidth / scene.bandwidth)
    noise_figure_lin = 10 ** (noise_figure_db / 10)
    noise_power_w = thermal_noise_w * noise_figure_lin

    tx_power_w = dbm_to_watt(tx_power_dbm)

    # SNR
    snr_linear = (np.squeeze(np.maximum((tx_power_w * g) / noise_power_w, 0)))
    snr_db = (10 * np.log10(snr_linear))

    # closed form BER for coherent bpsk over awgn - BER = 1/2 erfc (sqrt(eta))
    ber = 0.5 * erfc(np.sqrt(snr_linear))

    # shannon capacity
    throughput_mbps = bandwidth * np.log2(1 + snr_linear) / 1e6 #div by 1e6 to get mbps instead of bps

    rm_values = radio_map.path_gain.numpy()
    coverage_fraction = float(np.mean(rm_values > coverage_threshold_dbm))

    return {
        "snr_db":            snr_db,
        "ber":               float(ber),
        "throughput_mbps":   round(float(throughput_mbps), 2),
        "coverage_fraction": round(coverage_fraction, 4),
    }

# use tensors - extract what is what position, use them correctly - check
# implement fancy position changing - shaboing
# parallelization - done mostly
# documentation -
# plotting


# downlink transmission
# beamforming
# track da drone - if sionna has strongest beam then go along
# power as a function of snr
# plot kpis
# ofdm, broadband

