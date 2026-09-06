"""
A geometric (parametric) Sionna-ray-traced MIMO channel for HermesPy.

HermesPy's stock :class:`SionnaRTChannel` propagates via
``paths.cir(num_time_steps=...)``, materializing a dense per-(rx-ant, tx-ant, path,
time-sample) tensor -- tens of GB for a 64-element array over a long frame. This
module keeps the channel parametric instead: one synthetic-array solve gives
``paths.a`` of shape ``[rx_ant, tx_ant, P]`` plus per-path ``tau`` and ``doppler``,
and each path's delay and Doppler are replayed across the frame in closed form
(:meth:`GeometricChannelResponse.propagate`). Peak memory is the received signal
itself, independent of antenna count squared.

APPROXIMATIONS
--------------
1. Far-field / plane-wave across the aperture: each path arrives at one angle common
   to all elements, so the array response is a per-element phase. Breaks only inside
   the Fraunhofer distance (~0.26 m for an 8x8 at 28 GHz), and it is what
   ``synthetic_array=True`` already assumes.
2. Lazy Doppler (lossless): (gain, delay, Doppler) per path evaluated on demand
   rather than pre-expanded over every time sample.
3. Integer-sample delay snap: each path is delayed by floor(tau_i * W) whole samples
   (see ``_causal_delay``) rather than with a fractional-delay ramp, which would cost
   a per-path FFT/IFFT. Delay is quantized to <= half a sample, i.e. <= c/2W in range.
   This is what the stock SionnaRTChannel does too.

LAYERS
------
- :class:`GeometricChannelResponse` -- pure NumPy. Holds the parametric channel and
  does the propagation; ``sensing.py`` uses it directly, since it drives a live scene
  rather than a HermesPy ``Simulation``. Testable on CPU with synthetic a/tau/doppler.
- :class:`GeometricSionnaChannelSample` / :class:`GeometricSionnaChannelRealization`
  / :class:`GeometricSionnaChannel` -- a HermesPy ``Channel`` trio wrapping the
  response (``channel.realize()`` -> ``realization.sample(tx, rx, ...)`` ->
  ``sample.propagate(signal)``). ``jcas_drop.py`` drives this: one ``realize()`` runs
  the single PathSolver solve covering every tx/rx in the scene, and each link is a
  cheap ``.sample()`` against it.
"""

from __future__ import annotations

from typing import Any, Set
from typing_extensions import override

import numpy as np
from scipy.constants import speed_of_light

import sionna.rt as rt

from hermespy.channel.channel import (
    Channel,
    ChannelRealization,
    ChannelSample,
    ChannelSampleHook,
    InterpolationMode,
    LinkState,
)
from hermespy.core import (
    ChannelStateInformation,
    ChannelStateFormat,
    DeserializationProcess,
    SerializationProcess,
    SignalBlock,
)


def _causal_delay(x: np.ndarray, shift: int, axis: int = -1) -> np.ndarray:
    """Delay ``x`` by ``shift`` whole samples along ``axis``: zeros fill the front, the
    tail past the frame end is dropped, output length equals input. Causal, not
    circular -- an ``np.roll`` would wrap the tail to the front as a near-range artefact.
    """
    if shift <= 0:
        return np.array(x)              # shift 0 is a no-op copy
    if shift >= x.shape[axis]:
        return np.zeros_like(x)         # echo arrives after the frame ends
    out = np.zeros_like(x)
    dst = [slice(None)] * x.ndim
    src = [slice(None)] * x.ndim
    dst[axis] = slice(shift, None)
    src[axis] = slice(None, x.shape[axis] - shift)
    out[tuple(dst)] = x[tuple(src)]
    return out


class GeometricChannelResponse:
    """Parametric ray-traced channel: a fixed set of paths, each a (complex gain,
    delay, Doppler) triplet, with the array response already baked into the gains.

    ``a`` has shape ``[num_rx_ant, num_tx_ant, num_paths]`` and is the
    baseband-equivalent per-element coefficient a^b_i = a_i * exp(-j2*pi*f_c*tau_i):
    Sionna's raw ``paths.a`` (element pattern + array-steering phase) times the carrier
    delay phase, which raw ``paths.a`` omits -- :meth:`from_sionna_paths` applies it.
    ``tau`` (seconds) and ``doppler`` (Hz) are per path. No time axis is stored; the
    time evolution is synthesized in :meth:`propagate`, which delays each path by
    floor(tau_i * W) whole samples.
    """

    def __init__(
        self,
        a: np.ndarray,          # [num_rx_ant, num_tx_ant, num_paths] complex
        tau: np.ndarray,        # [num_paths] seconds
        doppler: np.ndarray,    # [num_paths] Hz
        carrier_frequency: float,
    ) -> None:
        a = np.asarray(a)
        if a.ndim != 3:
            raise ValueError(f"a must be [rx_ant, tx_ant, paths], got shape {a.shape}")
        self.a = a
        self.tau = np.asarray(tau, dtype=float).reshape(-1)
        self.doppler = np.asarray(doppler, dtype=float).reshape(-1)
        if self.tau.shape[0] != a.shape[2] or self.doppler.shape[0] != a.shape[2]:
            raise ValueError("tau/doppler length must equal a.shape[2] (num_paths)")
        self.carrier_frequency = float(carrier_frequency)

    @property
    def num_rx_ant(self) -> int:
        return self.a.shape[0]

    @property
    def num_tx_ant(self) -> int:
        return self.a.shape[1]

    @property
    def num_paths(self) -> int:
        return self.a.shape[2]

    # --- construction from a Sionna solve -------------------------------------

    @classmethod
    def from_sionna_paths(
        cls, paths: Any, carrier_frequency: float, rx_index: int = 0, tx_index: int = 0,
    ) -> "GeometricChannelResponse":
        """Extract one (rx, tx) link's parametric channel from a Sionna ``Paths`` object
        that may cover MULTIPLE transmitters/receivers in one solve.

        ``paths.a`` is ``[2, num_rx, rx_ant, num_tx, tx_ant, P]`` (real/imag, every link
        in the scene); this pulls out one ``(rx_index, tx_index)`` link's ``[rx_ant,
        tx_ant, P]`` slice. Defaults (0, 0) match the single-tx/single-rx scenes
        ``sense_snapshot`` builds; jcas_drop.py calls this repeatedly with different
        ``rx_index`` to get the radar link and every comms link from one solve.
        """
        a_np = np.array(paths.a)                                  # [2, num_rx, rx_ant, num_tx, tx_ant, P]
        a = (a_np[0, rx_index] + 1j * a_np[1, rx_index])[:, tx_index]  # -> [rx_ant, tx_ant, P], raw a_i
        tau = np.array(paths.tau)[rx_index, tx_index]             # [P]
        doppler = np.array(paths.doppler)[rx_index, tx_index]     # [P]

        # Fold in the carrier delay phase exp(-j2*pi*f_c*tau_i) that raw paths.a omits;
        # Sionna applies it only inside cir(), which this module never calls. tau<0 marks
        # invalid paths, dropped by select_paths()/state(), so their phase is never used.
        a = a * np.exp(-2j * np.pi * carrier_frequency * tau)[np.newaxis, np.newaxis, :]
        return cls(a=a, tau=tau, doppler=doppler, carrier_frequency=carrier_frequency)

    # --- path selection -------------------------------------------------------

    def select_paths(self, a_eff: np.ndarray, min_range_m: float) -> np.ndarray:
        """Indices of paths worth replaying: real (positive) delay beyond
        ``min_range_m`` round-trip, with non-zero amplitude. ``a_eff`` is the
        tx-combined per-rx gain ``[rx_ant, P]``.

        Only self-coupling clutter (inside the ``min_range_m`` round-trip gate) and
        zero-gain paths are dropped. Weak-but-nonzero paths are kept deliberately: the
        faint diffuse returns are the physical clutter floor, so there is no strength cap.
        """
        tau_min = 2.0 * min_range_m / speed_of_light  # round-trip
        strength = np.abs(a_eff).max(axis=0)          # [P]
        valid = (self.tau >= tau_min) & (strength > 0)
        return np.where(valid)[0]

    # --- propagation ----------------------------------------------------------

    def propagate(
        self,
        tx_signal: np.ndarray,      # [T] single stream, or [num_tx_ant, T] per-element
        sampling_rate: float,       # Hz
        tx_weights: np.ndarray | None = None,  # [num_tx_ant] excitation for the single-stream case
        start_time: float = 0.0,    # s -- absolute time of sample 0, for Doppler phase continuity
        power_w: float = 1.0,       # linear tx power scaling; output multiplied by sqrt(power_w)
        min_range_m: float = 0.0,   # drop paths shorter than this round-trip range
    ) -> np.ndarray:
        """Synthesize the received signal ``[num_rx_ant, T]`` by replaying each path.

        For each kept path p: delay the transmit signal by ``floor(tau[p] * W)`` whole
        samples (causal shift, no per-path FFT needed), multiply by the closed-form
        Doppler phase ``exp(2j*pi*doppler[p]*t)`` across the frame (``t`` anchored at
        ``start_time``, so consecutive frames stay phase-continuous), and accumulate
        through the path's per-element gain.

        Single-stream (``tx_signal`` is ``[T]`` + optional ``tx_weights``): the tx
        elements are combined once into ``a_eff = sum_tx a * w`` and the frame is
        delayed/Dopplered once per path. Multi-stream (``[num_tx_ant, T]``): the general
        MIMO contraction, one delay per tx element per path.
        """
        tx_signal = np.asarray(tx_signal)
        single_stream = tx_signal.ndim == 1
        if single_stream:
            num_samples = tx_signal.shape[0]
        else:
            if tx_signal.shape[0] != self.num_tx_ant:
                raise ValueError(
                    f"tx_signal has {tx_signal.shape[0]} streams, expected {self.num_tx_ant} tx antennas"
                )
            num_samples = tx_signal.shape[1]

        t_axis = np.arange(num_samples) / sampling_rate + start_time

        if single_stream:
            # tx excitation -> per-rx effective gain per path, then replay once per path.
            w = np.ones(self.num_tx_ant) if tx_weights is None else np.asarray(tx_weights)
            a_eff = (self.a * w[np.newaxis, :, np.newaxis]).sum(axis=1)   # [rx_ant, P]
            indices = self.select_paths(a_eff, min_range_m)

            received = np.zeros((self.num_rx_ant, num_samples), dtype=complex)
            for i in indices:
                shift = int(self.tau[i] * sampling_rate)   # integer-sample delay (floor)
                delayed = _causal_delay(tx_signal, shift)
                path_signal = delayed * np.exp(2j * np.pi * self.doppler[i] * t_axis)
                received += np.outer(a_eff[:, i], path_signal)
        else:
            # General MIMO: contract each tx element's delayed stream against a[:, :, p].
            # Path selection uses the tx-summed gain magnitude as a strength proxy.
            a_eff = self.a.sum(axis=1)   # [rx_ant, P]
            indices = self.select_paths(a_eff, min_range_m)

            received = np.zeros((self.num_rx_ant, num_samples), dtype=complex)
            for i in indices:
                shift = int(self.tau[i] * sampling_rate)              # integer-sample delay (floor)
                delayed = _causal_delay(tx_signal, shift, axis=1)     # [num_tx_ant, T]
                delayed = delayed * np.exp(2j * np.pi * self.doppler[i] * t_axis)[np.newaxis, :]

                # Under the plane-wave approximation each path's array response is a
                # per-element phase on each side, so a[:, :, i] is rank 1 and
                # A @ X == u (v^T X) exactly -- O((rx + tx) * T) instead of O(rx * tx * T).
                # Pivot on the largest entry for a well-conditioned division and verify
                # the reconstruction; anything not rank 1 (non-synthetic or dual-polarised
                # arrays) falls back to the exact matmul.
                A = self.a[:, :, i]
                r0, c0 = np.unravel_index(np.argmax(np.abs(A)), A.shape)
                pivot = A[r0, c0]
                if pivot != 0:
                    u = A[:, c0]                                      # [rx_ant]
                    v = A[r0, :]                                      # [tx_ant]
                    # atol scales with the pivot: entries far below the largest carry no
                    # energy, and a purely relative test would trip on numerical dust.
                    if np.allclose(A, np.outer(u, v) / pivot, rtol=1e-6,
                                    atol=1e-9 * abs(pivot)):
                        received += np.outer(u, (v @ delayed) / pivot)
                        continue
                received += A @ delayed                              # [rx_ant, T] exact fallback

        received *= np.sqrt(power_w)
        return received


# ---------------------------------------------------------------------------
# HermesPy Channel trio: wraps GeometricChannelResponse so the same physics plugs
# into a HermesPy Simulation/Device/JCAS pipeline. jcas_drop.py drives this layer;
# sensing.py uses GeometricChannelResponse directly.
# ---------------------------------------------------------------------------


class GeometricSionnaChannelSample(ChannelSample):
    """A HermesPy channel sample backed by a :class:`GeometricChannelResponse`."""

    def __init__(
        self,
        response: GeometricChannelResponse,
        gain: float,
        state: LinkState,
        min_range_m: float = 0.0,
    ) -> None:
        ChannelSample.__init__(self, state)
        self._response = response
        self._gain = gain
        self._min_range_m = min_range_m

    @property
    def response(self) -> GeometricChannelResponse:
        return self._response

    @property
    @override
    def expected_energy_scale(self) -> float:
        """Amplitude scale HermesPy uses for SNR bookkeeping, and gates on:
        ``ChannelRealization.propagate`` returns ``Signal.Empty`` when it is <= 0.

        Total path power rather than ``SionnaRTChannelSample``'s ``abs(sum(a))``: that
        form coherently sums complex gains, so thousands of ray-traced paths at
        arbitrary phases can cancel to near zero and gate off a good link. Summing
        |a|^2 over paths and tx antennas per rx antenna, averaging, and taking the root
        cannot cancel, so it is zero only when no path carries energy.
        """
        power_per_rx_ant = np.sum(np.abs(self._response.a) ** 2, axis=(1, 2))  # [rx_ant]
        return float(np.sqrt(np.mean(power_per_rx_ant)) * np.sqrt(self._gain))

    @override
    def _propagate(self, signal: SignalBlock, interpolation: InterpolationMode) -> SignalBlock:
        # signal is [num_tx_ant, num_samples]; Doppler is anchored at the block's
        # absolute start time so multi-block frames stay phase-continuous.
        x = np.asarray(signal)
        start_time = signal.offset / self.bandwidth
        received = self._response.propagate(
            x, sampling_rate=self.bandwidth, start_time=start_time, power_w=self._gain,
            min_range_m=self._min_range_m,
        )
        received = np.ascontiguousarray(received, dtype=np.complex128)
        # SignalBlock takes (num_streams, num_samples, offset, buffer), not the array.
        return SignalBlock(received.shape[0], received.shape[1], signal.offset, received.tobytes())

    @override
    def state(
        self,
        num_samples: int,
        max_num_taps: int,
        interpolation_mode: InterpolationMode = InterpolationMode.NEAREST,
    ) -> ChannelStateInformation:
        """Dense channel state information ``[rx, tx, num_samples, taps]``, for
        consumers that require CSI. This is the tensor whose size motivated the
        parametric design, so its footprint is the caller's to bound via
        ``num_samples`` and ``max_num_taps``. Radar sensing never calls it.
        """
        r = self._response
        bandwidth = self.bandwidth
        t_axis = np.arange(num_samples) / bandwidth
        max_delay_in_samples = 0 if r.num_paths == 0 else min(
            max_num_taps, int(np.ceil(np.max(r.tau) * bandwidth))
        )
        raw_state = np.zeros(
            (r.num_rx_ant, r.num_tx_ant, num_samples, 1 + max_delay_in_samples), dtype=np.complex128
        )
        for p in range(r.num_paths):
            if r.tau[p] < 0:
                continue
            tap = int(r.tau[p] * bandwidth)
            if tap > max_delay_in_samples:
                continue
            doppler_phase = np.exp(2j * np.pi * r.doppler[p] * t_axis)   # [num_samples]
            raw_state[:, :, :, tap] += r.a[:, :, p][:, :, np.newaxis] * doppler_phase[np.newaxis, np.newaxis, :]
        raw_state *= np.sqrt(self._gain)
        return ChannelStateInformation(ChannelStateFormat.IMPULSE_RESPONSE, raw_state)


class GeometricSionnaChannelRealization(ChannelRealization[GeometricSionnaChannelSample]):
    """One ray-trace solve, shared by every link sampled from it.

    A single ``PathSolver`` call at construction time covers every transmitter and
    receiver already present in ``scene``, and its ``paths`` are cached; ``_sample()``
    then slices out the (tx, rx) pair a ``LinkState`` asks for, with no further
    solving. The scene must already contain every tx/rx the caller needs -- this
    realization neither clears nor repopulates them.

    Keeps the scene's real (e.g. 8x8) array with ``synthetic_array=True`` rather than
    forcing 1x1 like :class:`SionnaRTChannelRealization`.
    """

    def __init__(
        self,
        scene: Any,
        scene_file: str,
        sample_hooks: Set[ChannelSampleHook] | None = None,
        gain: float = ChannelRealization._DEFAULT_GAIN,
        max_depth: int = 5,
        max_num_paths_per_src: int = 5000,
        monostatic_min_range_m: float = 3.0,
        path_solver_seed: int = 41,
        samples_per_src: int = 1000000,
    ) -> None:
        ChannelRealization.__init__(self, sample_hooks, gain)
        self._scene = scene
        self._scene_file = scene_file
        self._max_depth = max_depth
        self._max_num_paths_per_src = max_num_paths_per_src
        self._monostatic_min_range_m = monostatic_min_range_m
        self._samples_per_src = samples_per_src

        # One solve over every tx/rx already placed in `scene` by the caller.
        #
        # Sizing: max_num_paths_per_src bounds candidate paths per transmitter, but
        # paths.a is [2, num_rx, rx_ant, num_tx, tx_ant, P] and Sionna pads every
        # receiver to the same P, so device memory also scales with receiver count. At
        # P=5000, a 30-receiver scene with a 64-element array needs a ~4.9 GiB paths.a --
        # lower the cap for many-drone scenes.
        p_solver = rt.PathSolver()
        self.paths = p_solver(
            scene=self._scene,
            max_depth=self._max_depth,
            los=True,
            specular_reflection=True,
            diffuse_reflection=True,
            refraction=True,
            synthetic_array=True,
            seed=path_solver_seed,
            max_num_paths_per_src=self._max_num_paths_per_src,
            samples_per_src=self._samples_per_src,
        )
        # Snapshot tx/rx world positions at solve time, so `_sample()` can map a
        # LinkState's device positions back to Sionna tx/rx indices. Sionna positions
        # carry a trailing drjit batch dim, hence the reshape to a plain (3,).
        self._tx_names = list(self._scene.transmitters.keys())
        self._rx_names = list(self._scene.receivers.keys())
        self._tx_positions = np.array([
            np.asarray(self._scene.transmitters[name].position, dtype=float).reshape(3) for name in self._tx_names
        ])
        self._rx_positions = np.array([
            np.asarray(self._scene.receivers[name].position, dtype=float).reshape(3) for name in self._rx_names
        ])

    @property
    def scene(self) -> Any:
        return self._scene

    def _match_index(self, positions: np.ndarray, position: np.ndarray) -> int:
        """Index of the scene tx/rx closest to `position`. Devices passed to
        `.sample()` sit at their Sionna counterpart's exact coordinates, so in practice
        this is an exact match."""
        return int(np.argmin(np.linalg.norm(positions - position[np.newaxis, :], axis=1)))

    @override
    def _sample(self, state: LinkState) -> GeometricSionnaChannelSample:
        tx_index = self._match_index(self._tx_positions, np.asarray(state.transmitter.position, dtype=float))
        rx_index = self._match_index(self._rx_positions, np.asarray(state.receiver.position, dtype=float))
        response = GeometricChannelResponse.from_sionna_paths(
            self.paths, state.carrier_frequency, rx_index=rx_index, tx_index=tx_index,
        )
        # Co-located tx/rx need the self-coupling clutter guard; bistatic links don't.
        monostatic = np.allclose(self._tx_positions[tx_index], self._rx_positions[rx_index])
        min_range_m = self._monostatic_min_range_m if monostatic else 0.0
        return GeometricSionnaChannelSample(response, self.gain, state, min_range_m)

    @override
    def _reciprocal_sample(
        self, sample: GeometricSionnaChannelSample, state: LinkState
    ) -> GeometricSionnaChannelSample:
        return self._sample(state)

    @override
    def serialize(self, process: SerializationProcess) -> None:
        ChannelRealization.serialize(self, process)
        process.serialize_string(self._scene_file, "scene")

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> "GeometricSionnaChannelRealization":

        scene_file = process.deserialize_string("scene")
        return cls(
            rt.load_scene(scene_file),
            scene_file,
            **ChannelRealization._DeserializeParameters(process),  # type: ignore[arg-type]
        )


class GeometricSionnaChannel(Channel[GeometricSionnaChannelRealization, GeometricSionnaChannelSample]):
    """Geometric ray-traced MIMO channel: a parametric alternative to
    :class:`SionnaRTChannel` (no ``cir()`` time densification), so it survives large
    arrays. Takes a live ``scene`` object, array already configured, so it shares the
    exact scene a JCAS setup mutates.
    """

    def __init__(
        self,
        scene: Any,                      # live sionna.rt.Scene, array and tx/rx configured
        scene_file: str = "",
        gain: float = Channel._DEFAULT_GAIN,
        seed: int | None = None,
        max_depth: int = 5,
        max_num_paths_per_src: int = 5000,
        monostatic_min_range_m: float = 3.0,  # self-coupling clutter guard for co-located tx/rx links
        samples_per_src: int = 1000000,       # SBR rays launched per transmitter -- see _realize
    ) -> None:
        Channel.__init__(self, gain, seed)
        self._scene = scene
        self._scene_file = scene_file
        self._max_depth = max_depth
        self._max_num_paths_per_src = max_num_paths_per_src
        self._monostatic_min_range_m = monostatic_min_range_m
        self._samples_per_src = samples_per_src

    @property
    def scene(self) -> Any:
        return self._scene

    @override
    def _realize(self) -> GeometricSionnaChannelRealization:
        # The PathSolver call happens inside the Realization's constructor.
        return GeometricSionnaChannelRealization(
            self._scene,
            self._scene_file,
            self.sample_hooks,
            self.gain,
            self._max_depth,
            self._max_num_paths_per_src,
            self._monostatic_min_range_m,
            self.seed if self.seed is not None else 41,
            samples_per_src=self._samples_per_src,
        )
