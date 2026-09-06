import numpy as np

from ofdm_config import CARRIER_FREQUENCY


def estimate_channel(H, #true per-user channel matrix [num_rx, num_tx_ant]
                     csi_error_std = 0.0 # stddev of per-entry complex Gaussian CSI error
                     ):
    """Stand-in CSI estimator - placeholder for a future sensing-based estimate
    (e.g. ISAC position sensing -> re-derived H_est with structured, geometry-driven
    error instead of this i.i.d. noise).
    """

    if csi_error_std <= 0:
        return H
    noise = (np.random.normal(size=H.shape) + 1j * np.random.normal(size=H.shape)) / np.sqrt(2)
    return H + csi_error_std * noise


def rzf_precoder(H_est, #estimated per-user channel matrix [num_rx, num_tx_ant]
                 alpha = 0.0 # regularization (0 = pure ZF, >0 = RZF/MMSE, interpolates toward MRT as alpha grows)
                 ):
    """Regularized zero-forcing / MMSE precoder, built only from an (estimated) channel
    matrix. alpha=0 reduces to pinv(H_est) (pure ZF); larger alpha trades interference
    nulling for robustness to errors in H_est, approaching MRT as alpha -> inf.

    This is G = H^H (H H^H + alpha I)^-1 with unit-norm columns -- exactly
    sionna.phy.mimo.rzf_precoding_matrix (Bjornson/Hoydis/Sanguinetti Eq. 4.37). Kept
    in numpy because sionna.phy is torch-backed, and importing torch alongside
    tensorflow + sionna.rt (drjit/mitsuba) is fragile and memory-hungry here.

    Uses pinv rather than inv for the Gram inverse: a fully blocked user's channel row
    is ~zero, making H H^H singular at alpha=0. pinv zeros that user's beam instead of
    raising, and the guarded normalization leaves the column at zero, so the user gets
    ~0 throughput.
    """

    num_rx = H_est.shape[0]
    W = H_est.conj().T @ np.linalg.pinv(H_est @ H_est.conj().T + alpha * np.eye(num_rx))  # [num_tx_ant, num_rx]
    norms = np.linalg.norm(W, axis=0, keepdims=True)
    return W / np.where(norms > 0, norms, 1.0)  # unit-norm per user; blocked (zero) user stays zero


def compute_zf_precoder(paths, #PathSolver()
                        t = 0, # which transmitter to compute precoding weights for
                        alpha = 0.0, # passed to rzf_precoder
                        csi_error_std = 0.0, # passed to estimate_channel
                        rx_indices = None, # subset of receiver indices to include; None = all
                        ):
    """Multi-user ZF/RZF precoder [num_tx_ant, num_rx] from a Sionna path solve.

    Each user's channel row sums over its rx antennas and paths, keeping tx antennas,
    so NLOS paths (reflections/diffraction) fold in automatically. Collapsing the rx
    antennas into one row means ZF nulls only this summed direction, not every
    individual rx-antenna channel, so some residual inter-user interference remains
    against a receiver doing per-antenna MRC.
    """

    a_np = np.array(paths.a)           # [2, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths], float32
    tau_np = np.array(paths.tau)       # [num_rx, num_tx, num_paths]
    rows = rx_indices if rx_indices is not None else list(range(a_np.shape[1]))

    # Raw paths.a omits the carrier delay phase exp(-j2*pi*f_c*tau_i), which Sionna folds
    # in only inside cir(). It sets the relative phase between paths, so without it a
    # multi-path sum combines at arbitrary phase.
    carrier_phase = np.exp(-2j * np.pi * CARRIER_FREQUENCY * tau_np[:, t, :])  # [num_rx, num_paths]
    # Promote to complex one receiver at a time: over the whole tensor this allocates
    # complex128 at 4x the float32 source (~13 GB at 64 receivers) for a result of only
    # [num_rx, num_tx_ant].
    H = np.stack([
        np.sum(
            (a_np[0, r, :, t, :, :] + 1j * a_np[1, r, :, t, :, :])
            * carrier_phase[r][np.newaxis, np.newaxis, :],
            axis=(0, -1),
        )  # [num_tx_ant]
        for r in rows
    ], axis=0)  # [num_rx, num_tx_ant]

    H_est = estimate_channel(H, csi_error_std)
    return rzf_precoder(H_est, alpha)
