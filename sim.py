import mitsuba as mi
import numpy as np
import tensorflow as tf
from sionna.rt import SceneObject, ITURadioMaterial

try:
    import tensorflow_probability as tfp
except ImportError:
    # Optional: only build_rx_path()'s non-integer-subdivision branch needs it, which
    # raises with an install hint rather than failing this import.
    tfp = None


# Vertical gap between a drone's comms antenna (its Receiver point) and the bottom of
# its body, as a fraction of the body's half-extent -- the body is a solid metal cube,
# so one sitting flush on that point occludes the drone's own antenna.
#
# A ratio, not a fixed distance, because the clearance that matters is angular: for a
# cube of half-extent r centred r + c above the antenna, an arrival at elevation theta
# clears it iff tan(theta) < c/r. At 0.6 that cone is ~31 deg for any radius. Arrivals
# from below are never blocked; the body sits entirely above the antenna.
DRONE_ANTENNA_CLEARANCE_RATIO = 0.6


def drone_mesh_z_offset(drone_radius_m, antenna_clearance_m=None):
    """Height of a drone body's centre above its Receiver point, so the body's
    underside clears the antenna. ``antenna_clearance_m=None`` scales the gap with the
    body via DRONE_ANTENNA_CLEARANCE_RATIO, keeping the unoccluded arrival cone
    constant as the body grows; pass a number to override. A caller that moves a drone
    applies the same lift setup_drone_meshes() used at init.
    """
    if antenna_clearance_m is None:
        antenna_clearance_m = DRONE_ANTENNA_CLEARANCE_RATIO * drone_radius_m
    return drone_radius_m + antenna_clearance_m


def setup_drone_meshes(scene, drone_radius_m=0.5, scattering_coefficient=0.5,
                        antenna_clearance_m=None):
    """
    Add one metal cube SceneObject per drone receiver already in the scene.

    Each cube is the drone's one physical body, shared by both solves that run
    against this scene: the ray tracer finds comms multipath off it, and the
    monostatic radar finds backscatter off the same object.

    The body is Mitsuba's unit cube, [-1,1]^3, so `scaling=drone_radius_m` gives a
    half-extent of drone_radius_m -- the 0.5 default is a 1 m cube, a Matrice-class
    UAS. Its flat bottom face is why the clearance above exists.

    Returns
    -------
    drone_meshes : dict  {receiver_name: SceneObject}
    """
    # one shared material for all drone bodies, a thin metal skin
    mat_name = "_uas_metal"
    if mat_name not in scene.radio_materials:
        mat = ITURadioMaterial(name=mat_name, itu_type="metal", thickness=0.002,
                                scattering_coefficient=scattering_coefficient)
        scene.add(mat)
    else:
        mat = scene.radio_materials[mat_name]

    mesh_objs = []
    drone_meshes = {}
    for rx_name in scene.receivers:
        obj = SceneObject(mi_mesh=mi.load_dict({"type": "cube"}), name=f"_mesh_{rx_name}", radio_material=mat)
        mesh_objs.append(obj)
        drone_meshes[rx_name] = obj

    # scene.edit() binds each obj to the scene (sets obj.scene); scale/position
    # setters require self.scene to be set, so they must come after.
    scene.edit(add=mesh_objs)

    z_offset = drone_mesh_z_offset(drone_radius_m, antenna_clearance_m)
    for rx_name, rx in scene.receivers.items():
        if rx_name in drone_meshes:
            drone_meshes[rx_name].scaling = drone_radius_m
            # lift the body so its underside clears the antenna
            pos = np.asarray(rx.position, dtype=float).reshape(3)
            pos[2] += z_offset
            drone_meshes[rx_name].position = pos.tolist()

    return drone_meshes

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
        if tfp is None:
            raise ImportError(
                "tensorflow_probability is required for non-integer subdivisions. "
                "Install it with: pip install tensorflow-probability"
            )

        new_t = tf.linspace(0.0, 1.0, num_steps)
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