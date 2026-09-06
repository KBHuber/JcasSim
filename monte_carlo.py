"""
Monte Carlo JCAS scene generator.

The drone notebooks (1_drone.ipynb, 3_drone.ipynb) fly a hand-placed drone
(or few) along a hand-verified street corridor, timestep by timestep -- a
qualitative sanity check, not something you can sweep quantitatively. This
module instead uses randomly (but seeded, reproducibly) sampled STATIC
drone layouts -- one Sionna scene, one run_jcas_drop() call, no timesteps --
so a notebook can sweep five real experimental knobs:

SCENE axes (each value needs its own sampled layout):
  - `range_m`   : radius of the region drones can appear in (explores wide vs.
                  narrow beam effects -- a wide range puts targets at angles
                  and distances a narrow beam can miss).
  - `density_per_km2` : drones per km^2 within that region (explores angular/
                  range resolution -- how close together targets can be and
                  still be resolved, e.g. wavelength-dependent effects).
                  Airspace density is always quoted per km^2; the per-m^2
                  conversion happens once, inside sample_drone_scene.

WAVEFORM axes (each value needs its own ray-traced drop, but the SAME seed
gives the identical drone layout, so bands/combs are compared like-for-like):
  - `band`         : which carrier/array/numerology (ofdm_config.select_band).
  - `comb_spacing` : radar pilot comb M (ofdm_config.select_comb_spacing) --
                     the sensing/comms split: 1/M of the subcarriers become
                     radar pilots and the unambiguous range shrinks by M.

DETECTOR axis (post-processing -- costs no ray trace at all):
  - `detect_kwargs`: CFAR detector settings, chiefly `pfa`. Pass a LIST of
                     settings to score one finished radar cube at all of them,
                     which is how detector sensitivity should be swept -- it
                     turns an N-fold sweep into N cheap CFAR passes instead of
                     N full simulations, and holds the physics exactly fixed
                     across the sweep (same cube, same noise), so the resulting
                     P_d / false-alarm curve is a clean ROC.

SAMPLING REGION
---------------
The BS is a directional array (~30 dB front/back ratio) fixed on one boresight,
not an omnidirectional sensor, so the sampling region is a forward-facing SECTOR
rather than a disc: an annulus [near_range_m, range_m] restricted in azimuth to
boresight_azimuth +/- sector_half_width_deg (capped below 90 deg, beyond which is
behind the array's face). The default half-width is wider than the array's HPBW,
so plenty of drones land outside the main beam and beamwidth effects show up.

Altitude is sampled independently within z_range, a flight-altitude band below
typical rooflines and above ground clutter, not tied to the BS's own height.

BUILDING AVOIDANCE AND LOS
--------------------------
points_in_building() is a point-in-solid parity test against the scene's own
geometry (a ray to +z crosses a closed solid an odd number of times iff it
started inside), batched one ray_intersect() per bounce over the whole candidate
batch, since Dr.Jit's per-call dispatch overhead dominates otherwise.

Not-in-a-building alone is not enough: the BS is facade-mounted in a street grid,
so a wide sector at long range sweeps into blocks it cannot see through. At
range_m=100 with a 60 deg half-width in Munich, 25 of 34 sampled drones sat behind
another building and came back as dead links. points_los_blocked() rejects those
with the same batched ray cast. Real drones are served over NLOS paths too, but at
mmWave those run 20-40 dB weaker, so LOS is the floor of what is worth sampling.

NEAR-FIELD EXCLUSION
--------------------
near_range_m defaults to the array's Fraunhofer distance 2*D^2/lambda, computed
from Sionna's real element geometry (_max_fraunhofer_distance_m). Inside it the
plane-wave-across-the-aperture assumption GeometricChannelResponse relies on
breaks down, distorting that link's channel estimate and, through the joint ZF
precoder, every other drone's nulling. It uses the MAX over every band in
ofdm_config.BANDS (~0.53 m at 28 GHz, ~6.75 m at 10 GHz), so drone positions stay
identical across a band switch for the same seed.

DRONE SPACING AND COUNT
-----------------------
Drones may sit close together, down to just past their body radius -- that is what
demonstrates angular/range resolution -- but never overlapping. Count is
n_drones ~ Poisson(density * sector_area), so it is random per trial.

    rows = run_monte_carlo_scene(scene, bs_position=[-69,-64,12],
                                  look_at=[-55,-40,15], range_m=150,
                                  density_per_km2=2000, seed=0, tx_power_w=1.0)
"""

from __future__ import annotations

import gc
import inspect

import numpy as np
import mitsuba as mi
import drjit as dr
import sionna.rt as rt
from scipy.constants import speed_of_light

from sim import setup_drone_meshes
from jcas_drop import run_jcas_drop
from sensing import (
    score_sensing, detect_targets, world_to_spherical, ground_truth_range_velocity,
    plot_range_angle, _detection_range_azimuth, per_target_sensing_snr,
)
import ofdm_config


# detect_targets()' own signature defaults, so a row can record the settings actually in
# force without this module hardcoding sensing.py's numbers.
_DETECT_DEFAULTS = {
    name: param.default
    for name, param in inspect.signature(detect_targets).parameters.items()
    if param.default is not inspect.Parameter.empty
}


def create_munich_scene():
    """Load a fresh Munich scene with the active band's array configured, as the
    notebook's "create scene" cell does.

    A long sweep should call this periodically and replace its `scene` variable,
    on top of the per-trial cleanup. Repeatedly adding/removing drone Receivers and
    solving on the same long-lived Scene accumulates GPU memory that Dr.Jit's
    allocator has no record of and the per-trial flush does not release; only
    dropping every reference to the old Scene reclaims it.

    Anything the notebook's scene cell does beyond band/array setup must be
    reapplied after each reload.
    """
    scene = rt.load_scene(rt.scene.munich, merge_shapes=False)
    return apply_band_to_scene(scene)


def apply_band_to_scene(scene):
    """Point an already-loaded `scene` at the currently ACTIVE band: carrier
    frequency plus that band's array geometry. Returns the same scene.

    create_munich_scene() calls this at load time; a sweep that switches bands
    between trials must re-apply it, or the scene keeps the old carrier and element
    count while everything downstream reads the new band's constants.
    """
    scene.frequency = ofdm_config.CARRIER_FREQUENCY
    scene.tx_array = rt.PlanarArray(num_rows=ofdm_config.ARRAY_ROWS, num_cols=ofdm_config.ARRAY_COLS,
                                     pattern="tr38901", polarization="V")
    scene.rx_array = scene.tx_array
    return scene


def points_in_building(scene, positions, max_bounces=8, eps=1e-3):
    """Vectorized point-in-solid test: True where `positions[i]` is inside a
    closed building mesh in `scene`. positions: [N, 3] array-like.

    Casts one ray per point straight up and counts intersections with the scene
    geometry; an odd count means the ray started inside a closed solid. Batched
    across all N points per bounce, one ray_intersect() call per bounce.

    max_bounces bounds how many stacked solids one ray can pass through; these
    buildings rarely exceed 1-2, so 8 is a generous margin.
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    n = positions.shape[0]
    ox = mi.Float(positions[:, 0].copy())
    oy = mi.Float(positions[:, 1].copy())
    oz = mi.Float(positions[:, 2].copy())
    d = mi.Vector3f(0.0, 0.0, 1.0)
    hits = np.zeros(n, dtype=np.int64)
    active = np.ones(n, dtype=bool)

    for _ in range(max_bounces):
        if not active.any():
            break
        ray = mi.Ray3f(mi.Point3f(ox, oy, oz), d)
        si = scene.mi_scene.ray_intersect(ray)
        valid = np.asarray(si.is_valid()).astype(bool) & active
        hits += valid.astype(np.int64)
        new_oz = np.asarray(si.p.z) + eps
        oz = mi.Float(np.where(valid, new_oz, np.asarray(oz)))
        active = valid

    return (hits % 2) == 1


def points_los_blocked(scene, origin, positions, eps=0.05):
    """Vectorized straight-line LOS test: True where ANY scene geometry sits
    between `origin` (e.g. the BS position) and `positions[i]`. One ray per
    point, cast from `origin` toward it; blocked if the first hit is closer
    than the point itself (minus `eps`, so the point's own near-arrival isn't
    misread as self-blocking). Batched in a single ray_intersect() call, same
    reasoning as points_in_building().

    Straight-line only: it says nothing about the reflection/diffraction paths the
    PathSolver in run_jcas_drop() does trace. A cheap proxy for "worth simulating".
    """
    origin = np.asarray(origin, dtype=float)
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    n = positions.shape[0]
    delta = positions - origin[np.newaxis, :]
    dist = np.linalg.norm(delta, axis=1)
    direction = delta / dist[:, np.newaxis]

    ox = mi.Float(np.full(n, origin[0]))
    oy = mi.Float(np.full(n, origin[1]))
    oz = mi.Float(np.full(n, origin[2]))
    dx = mi.Float(direction[:, 0].copy())
    dy = mi.Float(direction[:, 1].copy())
    dz = mi.Float(direction[:, 2].copy())
    ray = mi.Ray3f(mi.Point3f(ox, oy, oz), mi.Vector3f(dx, dy, dz))
    si = scene.mi_scene.ray_intersect(ray)
    hit_t = np.asarray(si.t)
    valid = np.asarray(si.is_valid()).astype(bool)
    return valid & (hit_t < (dist - eps))


def _max_fraunhofer_distance_m(margin_m=0.5):
    """Max Fraunhofer (far-field) distance 2*D^2/lambda across every band in
    ofdm_config.BANDS, plus a flat safety margin -- see the module docstring's
    near-field section. D is each band's aperture diagonal, read from Sionna's own
    PlanarArray geometry. Used as sample_drone_scene()'s default near_range_m.
    """

    max_dist = 0.0
    for band in ofdm_config.BANDS.values():
        wavelength = speed_of_light / band.carrier_frequency
        arr = rt.PlanarArray(num_rows=band.array_rows, num_cols=band.array_cols,
                              pattern="tr38901", polarization="V")
        pos = np.array(arr.positions(wavelength)).T   # [num_ant, 3], meters
        D = float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0)))  # aperture diagonal
        max_dist = max(max_dist, 2 * D ** 2 / wavelength)
    return max_dist + margin_m


def _boresight_azimuth_rad(bs_position, look_at):
    bs_position = np.asarray(bs_position, dtype=float)
    look_at = np.asarray(look_at, dtype=float)
    delta = look_at - bs_position
    return float(np.arctan2(delta[1], delta[0]))


def sample_drone_scene(
    scene,
    bs_position,
    look_at,
    range_m,               # outer radius of the sampling sector, meters
    density_per_km2,       # drones per km^2 within the sector (converted below)
    seed,
    near_range_m=None,     # inner radius; None -> _max_fraunhofer_distance_m()
    sector_half_width_deg=60.0,   # azimuth extent either side of boresight; hard-capped < 90
    z_range=(8.0, 18.0),   # world-frame altitude band drones are sampled from
    drone_radius_m=0.5,    # physical body half-extent (sim.py setup_drone_meshes default)
    min_separation_m=0.1,  # extra clearance beyond touching bodies
    speed_range_mps=(0.0, 10.0),
    require_los=True,      # reject candidates with no straight-line path to bs_position
    max_batches=20,
    batch_size=64,
):
    """Sample a random (but seeded) drone layout: positions inside a forward-facing
    sector of the BS's boresight, outside every building, with a clear line of sight
    to bs_position, not overlapping, each with a random horizontal velocity.

    Returns a list of dicts: {"position": [x,y,z], "velocity": [vx,vy,vz]}. May
    return fewer than the Poisson-sampled count if valid non-overlapping spots run
    out within max_batches, warning rather than looping forever.
    """
    if sector_half_width_deg >= 90.0:
        raise ValueError(
            f"sector_half_width_deg must be < 90 (a directional array can't see "
            f"behind its own face); got {sector_half_width_deg}"
        )
    if near_range_m is None:
        near_range_m = _max_fraunhofer_distance_m()
    rng = np.random.default_rng(seed)
    bs_position = np.asarray(bs_position, dtype=float)
    half_width_rad = np.deg2rad(sector_half_width_deg)
    boresight_az = _boresight_azimuth_rad(bs_position, look_at)

    area = half_width_rad * (range_m ** 2 - near_range_m ** 2)  # sector area, m^2
    n_target = int(rng.poisson(density_per_km2 / 1e6 * area))

    min_sep = 2 * drone_radius_m + min_separation_m
    accepted_xyz = []

    for _ in range(max_batches):
        if len(accepted_xyz) >= n_target:
            break
        n_needed = n_target - len(accepted_xyz)
        n_gen = max(batch_size, 4 * n_needed)

        r = np.sqrt(rng.uniform(near_range_m ** 2, range_m ** 2, size=n_gen))
        az = rng.uniform(boresight_az - half_width_rad, boresight_az + half_width_rad, size=n_gen)
        z = rng.uniform(z_range[0], z_range[1], size=n_gen)
        x = bs_position[0] + r * np.cos(az)
        y = bs_position[1] + r * np.sin(az)
        candidates = np.stack([x, y, z], axis=1)

        clear_of_buildings = ~points_in_building(scene, candidates)
        candidates = candidates[clear_of_buildings]

        if require_los and candidates.shape[0] > 0:
            visible = ~points_los_blocked(scene, bs_position, candidates)
            candidates = candidates[visible]

        for cand in candidates:
            if len(accepted_xyz) >= n_target:
                break
            if accepted_xyz:
                dists = np.linalg.norm(np.asarray(accepted_xyz) - cand, axis=1)
                if dists.min() < min_sep:
                    continue
            accepted_xyz.append(cand)

    if len(accepted_xyz) < n_target:
        print(
            f"sample_drone_scene: only placed {len(accepted_xyz)}/{n_target} drones "
            f"after {max_batches} batches (region too small/dense for this separation) -- "
            f"continuing with fewer drones."
        )

    drones = []
    for pos in accepted_xyz:
        speed = rng.uniform(*speed_range_mps)
        heading = rng.uniform(0.0, 2 * np.pi)
        vel = [float(speed * np.cos(heading)), float(speed * np.sin(heading)), 0.0]
        drones.append({"position": pos.tolist(), "velocity": vel})
    return drones


def _associate_per_target(truths_ra, dets, range_gate_m, azimuth_gate_deg):
    """Per-truth-target (detected, position_error_m, range_error_m,
    azimuth_error_deg): the same nearest-in-gate association score_sensing() does
    internally, surfaced per target instead of aggregated, so each drone gets its
    own CSV row. truths_ra: list of (range_m, azimuth_rad)."""
    az_gate = np.deg2rad(azimuth_gate_deg)
    det_ra = _detection_range_azimuth(dets)

    results = []
    for (r_t, az_t) in truths_ra:
        best = None
        for (r_d, az_d) in det_ra:
            d_r = r_d - r_t
            d_az = np.arctan2(np.sin(az_d - az_t), np.cos(az_d - az_t))
            if abs(d_r) <= range_gate_m and abs(d_az) <= az_gate:
                nd = np.hypot(d_r / range_gate_m, d_az / az_gate)
                if best is None or nd < best[0]:
                    best = (nd, r_d, az_d)
        if best is None:
            results.append({"detected": False, "position_error_m": float("nan"),
                             "range_error_m": float("nan"), "azimuth_error_deg": float("nan")})
        else:
            _, r_d, az_d = best
            dx = r_d * np.cos(az_d) - r_t * np.cos(az_t)
            dy = r_d * np.sin(az_d) - r_t * np.sin(az_t)
            results.append({
                "detected": True,
                "position_error_m": float(np.hypot(dx, dy)),
                "range_error_m": float(abs(r_d - r_t)),
                "azimuth_error_deg": float(abs(np.rad2deg(np.arctan2(np.sin(az_d - az_t), np.cos(az_d - az_t))))),
            })
    return results


def run_monte_carlo_scene(
    scene,
    bs_position,
    look_at,
    range_m,
    density_per_km2,              # drones per km^2 in the sampling sector (standard unit)
    seed,
    tx_power_w=1.0,
    csi_error_std=0.0,
    scene_id=None,
    range_gate_m=4.0,
    azimuth_gate_deg=4.0,
    range_azimuth_plot=False,
    sample_kwargs=None,
    cleanup=True,
    scattering_coefficient=0.5,   # drone body diffuse-scattering fraction -- see sim.py
    samples_per_src=1_000_000,    # SBR rays per transmitter -- raise if drones at range
                                   # are missed entirely (see run_jcas_drop)
    band=None,                    # ofdm_config band key to run this trial at, or None for
                                   # the active one; also re-points `scene` at it
    comb_spacing=None,            # radar pilot comb spacing M; None leaves the active one
    detect_kwargs=None,           # detect_targets() kwargs, e.g. {"pfa": 1e-4}, or a LIST
                                   # of them to score this one cube at several settings
):
    """One Monte Carlo trial: sample a random drone layout (sample_drone_scene), run
    one JCAS drop (run_jcas_drop -- decoded comms plus a radar cube, no timesteps),
    score sensing accuracy, and return one CSV row (dict) per drone. Returns [] if the
    sampled scene has zero drones.

    scene must already be load_scene()'d (create_munich_scene); this adds and removes
    its own drone Receivers and meshes per call, so it can be called repeatedly
    against the same scene.

    SWEEPABLE WAVEFORM/DETECTOR AXES
    --------------------------------
      * `band` and `comb_spacing` change the transmitted waveform, so each value needs
        its own ray-traced drop. Both are applied before anything solves, and `band`
        also re-points the scene's carrier/array. Drone positions are unaffected --
        sample_drone_scene reads no band constant -- so the same seed lays out the
        identical scene on every band.
      * `detect_kwargs` changes only the detector, which is post-processing on the
        finished cube. A list of settings scores all of them off the one drop, emitting
        a row per (drone, setting); sweep detector sensitivity this way rather than by
        re-running trials.

    Every row records the band, comb spacing and detector settings it was produced
    under, plus the derived resolutions, so a multi-axis sweep stays disentangleable.

    cleanup: True (the default, for sweeps) removes the drone Receivers and meshes
    from `scene` before returning. Pass False for a one-off scene you want to inspect
    afterward, e.g. scene.preview().
    """
    if sample_kwargs is None:
        sample_kwargs = {}

    # Waveform axes, applied before any solve so the drop, the sensing processor and
    # the scene agree on one band/comb for this trial.
    if band is not None:
        ofdm_config.select_band(band)
        apply_band_to_scene(scene)
    if comb_spacing is not None:
        ofdm_config.select_comb_spacing(comb_spacing)

    # Detector axis: one or many settings scored off the same cube.
    detect_variants = (list(detect_kwargs) if isinstance(detect_kwargs, (list, tuple))
                       else [detect_kwargs or {}])

    if scene_id is None:
        # Identifies the layout only; band/comb/pfa are columns of their own, so a
        # paired band-vs-band comparison can match rows on this id.
        scene_id = f"r{range_m}_d{density_per_km2:g}_s{seed}"

    # Everything below goes through the `finally` cleanup, the zero-drone early return
    # included: sample_drone_scene()'s ray casts allocate on the scene even when no
    # drone is accepted, so skipping the flush there leaks across a sweep.
    drone_names = []
    drone_meshes = None
    try:
        drones = sample_drone_scene(scene, bs_position, look_at, range_m, density_per_km2,
                                    seed, **sample_kwargs)
        if not drones:
            return []

        drone_names = [f"mc_uas_{i}" for i in range(len(drones))]
        for name, drone in zip(drone_names, drones):
            rx = rt.Receiver(name=name, position=drone["position"], orientation=[0.0, 0.0, 0.0])
            scene.add(rx)

        drone_meshes = setup_drone_meshes(
            scene,
            drone_radius_m=sample_kwargs.get("drone_radius_m", 0.5),
            scattering_coefficient=scattering_coefficient,
        )
        for name, drone in zip(drone_names, drones):
            drone_meshes[name].velocity = drone["velocity"]

        result = run_jcas_drop(
            scene, bs_position=bs_position, look_at=look_at, rx_names=drone_names,
            tx_power_w=tx_power_w, csi_error_std=csi_error_std,
            samples_per_src=samples_per_src,
            # 25% margin past the sampled sector, so a drone at the far edge keeps a
            # full CFAR training window beyond it
            max_range_m=1.25 * range_m,
        )
        cube = result.radar_cube

        positions = [d["position"] for d in drones]
        truths_raz = [world_to_spherical(bs_position, pos) for pos in positions]   # (range, azimuth, zenith)
        truths_ra = [t[:2] for t in truths_raz]                                    # (range, azimuth)
        radial_velocities = [
            ground_truth_range_velocity(bs_position, d["position"], d["velocity"])[1] for d in drones
        ]
        rows = []
        # One drop, scored once per detector setting: detection is post-processing on
        # `cube`, so each extra setting costs a CFAR pass, not another ray trace.
        for variant in detect_variants:
            # fill in detect_targets' defaults so the CSV records what was in force
            settings = {**_DETECT_DEFAULTS, **variant}
            dets = detect_targets(cube, **variant)
            scene_metrics = score_sensing(
                [cube], [positions], bs_position, detections_per_frame=[dets],
                range_gate_m=range_gate_m, azimuth_gate_deg=azimuth_gate_deg,
            )
            per_drone = _associate_per_target(truths_ra, dets, range_gate_m, azimuth_gate_deg)
            # Two-way SNR at each drone's true range/Doppler/bearing cell, measured off
            # the same cube the detector runs on. Recorded for every drone, detected or
            # not, so the CSV can explain misses.
            sensing_snrs = per_target_sensing_snr(cube, truths_raz, radial_velocities)

            if range_azimuth_plot:
                plot_range_angle(
                    cube, ground_truth=truths_ra, detections=dets, max_range_m=range_m,
                    title=f"Range-Azimuth — scene {scene_id} (pfa={settings['pfa']:.0e})",
                    boresight_azimuth_rad=_boresight_azimuth_rad(bs_position, look_at),
                )

            for i, (name, drone) in enumerate(zip(drone_names, drones)):
                pos = drone["position"]
                vel = drone["velocity"]
                true_range, true_az = truths_ra[i]
                true_radial_vel = radial_velocities[i]
                comms = result.comms[name]
                rows.append({
                    "scene_id": scene_id,
                    "seed": seed,
                    "band": ofdm_config.ACTIVE_BAND.key,
                    "comb_spacing": ofdm_config.RADAR_COMB_SPACING,
                    "pfa": settings["pfa"],
                    "num_training_cells": str(settings["num_training_cells"]),
                    "num_guard_cells": str(settings["num_guard_cells"]),
                    "range_resolution_m": ofdm_config.RANGE_RESOLUTION,
                    "max_unambiguous_range_m": ofdm_config.MAX_UNAMBIGUOUS_RANGE,
                    "range_m": range_m,
                    "density_per_km2": density_per_km2,
                    "num_drones": len(drones),
                    "prob_detection": scene_metrics.prob_detection,
                    "false_alarms_per_frame": scene_metrics.false_alarms_per_frame,
                    "false_alarm_fraction": scene_metrics.false_alarm_fraction,
                    "rmse_position_error_m": scene_metrics.rmse_position_error_m,
                    "drone_id": name,
                    "x": pos[0], "y": pos[1], "z": pos[2],
                    "vx": vel[0], "vy": vel[1], "vz": vel[2],
                    "true_range_m": true_range,
                    "true_azimuth_deg": np.rad2deg(true_az),
                    "true_radial_velocity_mps": true_radial_vel,
                    # `snr_db` for CSV compatibility, but the quantity is an EVM-based
                    # effective SINR -- label it SINR on plots.
                    "snr_db": comms.snr_db,                   # comms link, one-way BS -> drone
                    "sensing_snr_db": sensing_snrs[i],        # sensing link, two-way BS -> drone -> BS
                    "ber": comms.ber,
                    "throughput_mbps": comms.throughput_mbps,
                    **per_drone[i],
                })
        return rows
    finally:
        if cleanup:
            if drone_meshes is not None:
                scene.edit(remove=list(drone_meshes.values()))
            for name in drone_names:
                if name in scene.receivers:
                    scene.remove(name)
            # Each trial has a different drone count, so Dr.Jit JIT-compiles a fresh
            # kernel set and BVH per shape count and never releases the old ones -- over
            # dozens of trials that exhausts device memory. A re-JIT next trial is cheap
            # beside the ray-trace solve.
            #
            # gc.collect() must run BEFORE flush_malloc_cache(), which only releases
            # blocks already at refcount zero; the nanobind wrappers behind a scene edit
            # form Python cycles that plain refcounting never breaks.
            gc.collect()
            dr.flush_malloc_cache()
            dr.flush_kernel_cache()
