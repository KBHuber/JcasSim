import sionna.rt as rt
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import tensorflow as tf
from metrics import compute_kpis, compute_zf_precoder
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera, \
    PathSolver, RadioMapSolver, subcarrier_frequencies


def simulate(rx_path,   # tf.Tensor [N, T, 3]
             cameras,
             scene,
             generate_metrics=True,
             render=False,
             beamforming_on=True,
             precoder_alpha=0.0,    # RZF/MMSE regularization (0 = pure ZF)
             csi_error_std=0.0):    # stddev of simulated CSI estimation error

    rm_solver = RadioMapSolver()
    p_solver  = PathSolver()

    num_steps = rx_path.shape[1]  # T dimension
    kpi_log = [] if generate_metrics else None

    for t in range(num_steps):

        # --- Move ALL receivers (drones) to their position at step t ---
        for i, rx in enumerate(scene.receivers.values()):
            rx.position = rx_path[i, t, :]  # tf.Tensor slice, shape [3]

        # --- Solve radio map and paths after repositioning ---
        rm = rm_solver(
            scene=scene,
            samples_per_tx=20**6,
            refraction=True,
            max_depth=5,
            center=[0, 0, 0.5],
            orientation=[0, 0, 0],
            size=[186, 121],
            cell_size=[2, 2]
        )

        paths = p_solver(
            scene=scene,
            max_depth=5,
            refraction=True,
            los=True,
            diffraction=True,
            max_num_paths_per_src=10000
        )

        # [num_tx_ant, num_rx] precoding weights, one column per drone (None if beamforming is off)
        W = compute_zf_precoder(paths, alpha=precoder_alpha, csi_error_std=csi_error_std) if beamforming_on else None

        # --- Metrics ---
        if generate_metrics:
            step_kpis = {}
            for i, (name, rx) in enumerate(scene.receivers.items()):
                precoding_vector = W[:, i] if beamforming_on else None
                kpis = compute_kpis(paths, radio_map=rm, r=i, scene=scene, precoding_vector=precoding_vector)
                kpis["position"] = rx.position.numpy().tolist()
                step_kpis[name] = kpis
            kpi_log.append(step_kpis)
            print(f"Step {t}: {step_kpis}")

        # --- Render ---
        if render:
            if beamforming_on:
                # --- per-drone radio map with that drone's ZF beam, combined incoherently ---
                # each beam carries 1/num_rx of the total tx power (matches compute_kpis), so
                # the combined RSS is just the mean of the per-beam path gains times tx_power
                num_rx = W.shape[1]
                pathgain_sum = None
                for i in range(num_rx):
                    precoding_vec = (
                        mi.TensorXf(W[:, i].real.astype(np.float32)),
                        mi.TensorXf(W[:, i].imag.astype(np.float32))
                    )
                    rm_beam = rm_solver(
                        scene=scene,
                        precoding_vec=precoding_vec,
                        samples_per_tx=10**6,
                        refraction=True,
                        max_depth=5,
                        center=[0, 0, 0.5],
                        orientation=[0, 0, 0],
                        size=[186, 121],
                        cell_size=[2, 2]
                    )
                    gain = rm_beam.path_gain.numpy()
                    pathgain_sum = gain if pathgain_sum is None else pathgain_sum + gain

                # path_gain has no public setter - _pathgain_map is the tensor it reads from,
                # overwrite it so rm.rss reflects the combined per-beam coverage at render time
                rm._pathgain_map = mi.TensorXf((pathgain_sum / num_rx).astype(np.float32))

            # When beamforming is off, rm already holds the un-precoded radio map.
            # Always pass radio_map; only pass paths when metrics are computed
            # (paths object is always available here regardless of generate_metrics)
            for cam in cameras:
                scene.render(
                    camera=cameras[cam],
                    paths=paths,
                    radio_map=rm,
                    num_samples=512,
                    rm_show_color_bar=True,
                    rm_vmax=-40,
                    rm_vmin=-150,
                    rm_metric="rss"
                )

    return kpi_log


# written by claude sonnet 4.6 low
# I didn't want to do this by hand but I havent been able to break it so probably right
def build_rx_path(
    start_points: tf.Tensor,
    base_velocities: tf.Tensor,
    num_steps: int,
    dt: float,
    base_dt: float = 1,
    on_mismatch: str = "error",
) -> tf.Tensor:
    """
    Subdivide or resample drone velocity segments into a higher-resolution path.

    Args:
        start_points:    Starting positions, shape [N, 3].
        base_velocities: Coarse velocity segments, shape [N, T, 3].
        num_steps:       Number of fine-grained steps to produce.
        dt:              Timestep for the fine-grained path.
        base_dt:         Timestep that the base_velocities were defined at.
        on_mismatch:     What to do if num_steps * dt != total trajectory time.
                         One of "error", "warn", or "snap".

    Returns:
        rx_path: Shape [N, num_steps + 1, 3], including the start point.

    Raises:
        ValueError: If on_mismatch="error" and num_steps * dt != total time.
        ValueError: If on_mismatch is not one of "error", "warn", "snap".
    """
    base_steps = base_velocities.shape[1]  # reads T from [N, T, 3]
    total_time = base_dt * base_steps
    requested_time = num_steps * dt

    # --- handle time mismatch ---
    if abs(requested_time - total_time) > 1e-6:
        msg = (
            f"num_steps * dt = {num_steps} * {dt} = {requested_time:.4f}s "
            f"but total trajectory time is {total_time}s."
        )
        if on_mismatch == "error":
            raise ValueError(msg + " Adjust num_steps or dt to match.")
        elif on_mismatch == "warn":
            print(f"Warning: {msg} Endpoint positions will drift.")
        elif on_mismatch == "snap":
            dt = total_time / num_steps
            print(f"Warning: {msg} Snapping dt to {dt:.6f}s.")
        else:
            raise ValueError(f"on_mismatch must be 'error', 'warn', or 'snap'. Got '{on_mismatch}'.")

    # --- check for clean subdivision ---
    subdivisions = num_steps / base_steps
    is_clean = abs(subdivisions - round(subdivisions)) < 1e-6

    if is_clean:
        subdivisions = int(round(subdivisions))
        velocities = tf.repeat(base_velocities, repeats=subdivisions, axis=1)
    else:
        # Resample via linear interpolation along the time axis
        try:
            import tensorflow_probability as tfp
        except ImportError:
            raise ImportError(
                "tensorflow_probability is required for non-integer subdivisions. "
                "Install it with: pip install tensorflow-probability"
            )

        base_t = tf.linspace(0.0, 1.0, base_steps)
        new_t  = tf.linspace(0.0, 1.0, num_steps)
        num_drones = base_velocities.shape[0]

        velocities = tf.stack([
            tf.stack([
                tfp.math.interp_regular_1d_grid(new_t, 0.0, 1.0, base_velocities[n, :, c])
                for c in range(3)
            ], axis=-1)
            for n in range(num_drones)
        ], axis=0)  # [N, num_steps, 3]

    # --- build path ---
    displacements = tf.cumsum(velocities * dt, axis=1, exclusive=False)
    rx_path = tf.concat([
        start_points[:, tf.newaxis, :],
        start_points[:, tf.newaxis, :] + displacements,
    ], axis=1)  # [N, num_steps + 1, 3]

    return rx_path