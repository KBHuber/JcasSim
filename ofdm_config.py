"""OFDM numerology shared by the comms side (metrics.py) and the radar (sensing.py).

Both import from here, so the radar really is the comms waveform.

Three band presets, selected with select_band() or DEFAULT_BAND:

    key    carrier   array    SCS      N      occupied B
    "3.5"  3.5 GHz    4x4     30 kHz   3280   98.4 MHz     FR1 C-band (n78)
    "10"   10  GHz    8x8     60 kHz   1640   98.4 MHz     FR3 upper mid-band
    "28"   28  GHz    8x8    120 kHz    820   98.4 MHz     FR2 mmWave
"""

from dataclasses import dataclass

import numpy as np
from scipy.constants import speed_of_light

from hermespy.modem import (
    OFDMWaveform, GridResource, GridElement, SymbolSection, ElementType, PrefixType,
    UniformPilotSymbolSequence,
)


# --- Band presets -------------------------------------------------------------

@dataclass(frozen=True)
class BandConfig:
    """The band-dependent knobs. Slow-time and sensing policy are shared constants."""
    key: str                    # short selector, e.g. "3.5"
    label: str                  # for plots/logs
    carrier_frequency: float    # Hz
    subcarrier_spacing: float   # Hz, from the NR ladder 15*2^n kHz
    num_subcarriers: int
    array_rows: int
    array_cols: int
    channel_bandwidth: float    # Hz, nominal 3GPP width; occupied B = N * SCS


BANDS: dict[str, BandConfig] = {
    "3.5": BandConfig(
        key="3.5", label="FR1 C-band (n78), 3.5 GHz, 4x4, 98.4 MHz",
        carrier_frequency=3.5e9, subcarrier_spacing=30e3, num_subcarriers=3280,
        array_rows=4, array_cols=4, channel_bandwidth=100e6,
    ),
    "10": BandConfig(
        key="10", label="FR3 upper mid-band, 10 GHz, 8x8, 98.4 MHz",
        carrier_frequency=10e9, subcarrier_spacing=60e3, num_subcarriers=1640,
        array_rows=8, array_cols=8, channel_bandwidth=100e6,
    ),
    "28": BandConfig(
        key="28", label="FR2 mmWave, 28 GHz, 8x8, 98.4 MHz",
        carrier_frequency=28e9, subcarrier_spacing=120e3, num_subcarriers=820,
        array_rows=8, array_cols=8, channel_bandwidth=100e6,
    ),
}

DEFAULT_BAND = "28"


# --- Band-independent policy --------------------------------------------------

# Slow-time symbols per frame, which sets Doppler resolution. 128 was too coarse:
# at 3.5 GHz it gave 9.31 m/s per bin, so every drone landed in the zero-Doppler
# clutter bin. This is also the memory knob -- frame samples are CPI * bandwidth,
# and 512 exhausts a 10 GiB budget. Measured on a full 360x90 scan:
#
#   band     CPI      dv        peak RSS
#   3.5    9.2 ms   4.66 m/s     3.4 GiB
#   10     4.6 ms   3.26 m/s     3.8 GiB
#   28     2.3 ms   2.33 m/s     2.5 GiB
NUM_OFDM_SYMBOLS = 256

PREFIX_RATIO = 0.078           # CP as a fraction of the useful symbol (~NR normal CP)
DC_SUPPRESSION = True

# How the frame feeds the sensing processor:
#   "payload" -- reciprocal-filter the whole frame (Sturm & Wiesbeck 2011). Best
#      resolution and SNR, but random 16-QAM data raises a self-noise floor.
#   "comb" -- dedicate a 1-in-M comb to a constant-modulus pilot and sense on those
#      alone (Sturm/Zwick/Wiesbeck VTC 2009). No self-noise, at the cost of 1/M of
#      the comms subcarriers and an M-times smaller unambiguous range.
SENSING_SUBCARRIER_MODE = "comb"   # "payload" or "comb"
RADAR_COMB_SPACING = 4             # M; 0 forces payload mode so a sweep over M can
                                   # include "no comb" as a point
RADAR_PILOT_SYMBOL = 1.0 + 0.0j


# --- Active band --------------------------------------------------------------
# Rebound by select_band() via globals(); annotated here so static analysis sees them.
ACTIVE_BAND: BandConfig
CARRIER_FREQUENCY: float
SUBCARRIER_SPACING: float
NUM_SUBCARRIERS: int
ARRAY_ROWS: int
ARRAY_COLS: int
BANDWIDTH: float                  # occupied, = NUM_SUBCARRIERS * SUBCARRIER_SPACING
CP_OVERHEAD: float
RANGE_RESOLUTION: float           # m
MAX_UNAMBIGUOUS_RANGE: float      # m


def comb_enabled() -> bool:
    """True iff a radar pilot comb is in the waveform right now.

    The single place that decides, so the mask, the waveform grid and the
    unambiguous range agree. M = 0 is the off switch; M = 1 would leave no data
    subcarriers at all.
    """
    return SENSING_SUBCARRIER_MODE == "comb" and RADAR_COMB_SPACING >= 2


def select_band(name: str) -> BandConfig:
    """Make ``name`` the active band, rebinding the band-dependent globals.

    ``name`` is a key of BANDS: "3.5", "10" or "28". Consumers read the globals
    live, so calling this before a run switches carrier, array, bandwidth and
    numerology together.
    """
    if name not in BANDS:
        raise ValueError(f"unknown band {name!r}; choose one of {list(BANDS)}")
    band = BANDS[name]

    g = globals()
    g["ACTIVE_BAND"] = band
    g["CARRIER_FREQUENCY"] = band.carrier_frequency
    g["SUBCARRIER_SPACING"] = band.subcarrier_spacing
    g["NUM_SUBCARRIERS"] = band.num_subcarriers
    g["ARRAY_ROWS"] = band.array_rows
    g["ARRAY_COLS"] = band.array_cols

    bandwidth = band.num_subcarriers * band.subcarrier_spacing
    g["BANDWIDTH"] = bandwidth
    g["CP_OVERHEAD"] = 1.0 / (1.0 + PREFIX_RATIO)
    # Set by the spanned bandwidth, so the comb keeps it -- it still spans the band.
    g["RANGE_RESOLUTION"] = speed_of_light / (2 * bandwidth)
    # Set by the sample spacing in frequency: a 1-in-M comb samples every M*df and
    # aliases beyond c/(2*M*df).
    unamb_spacing = band.subcarrier_spacing * (RADAR_COMB_SPACING if comb_enabled() else 1)
    g["MAX_UNAMBIGUOUS_RANGE"] = speed_of_light / (2 * unamb_spacing)
    return band


def comb_spacing_is_valid(name: str, spacing: int) -> bool:
    """Can band ``name`` run a 1-in-``spacing`` comb? Lets a (band x M) sweep skip
    unrunnable points. The comb is laid down as mask[::M], so an M that doesn't
    divide the grid just ends one subcarrier short at the band edge -- everything
    but M = 1 is fine.
    """
    return name in BANDS and spacing != 1


def select_comb_spacing(spacing: int) -> int:
    """Set the comb spacing M and re-derive what depends on it.

    M is the sensing/comms tradeoff knob: raising it hands subcarriers back to the
    comms link (the pilot tax is 1/M) but shrinks the unambiguous range by the same
    factor. Range resolution is unaffected.
    """
    spacing = int(spacing)
    if spacing < 0:
        raise ValueError(f"comb spacing must be >= 0 (0 = comb off), got {spacing}")
    if spacing == 1:
        raise ValueError(
            "comb spacing 1 would put a pilot on every subcarrier, leaving no data "
            "subcarriers and therefore no comms link; use 0 to turn the comb off"
        )

    globals()["RADAR_COMB_SPACING"] = spacing
    select_band(ACTIVE_BAND.key)   # rebinds MAX_UNAMBIGUOUS_RANGE for the new M
    return spacing


select_band(DEFAULT_BAND)


def dc_subcarrier_index() -> int:
    return NUM_SUBCARRIERS // 2


def radar_comb_mask():
    """Subcarriers dedicated to the radar pilot comb, else all-False. Filled from
    index 0 in the same centred grid subcarrier_frequencies() uses, so a True here
    lines up with HermesPy's REFERENCE positions."""
    mask = np.zeros(NUM_SUBCARRIERS, dtype=bool)
    if comb_enabled():
        mask[::RADAR_COMB_SPACING] = True
    return mask


def data_subcarrier_mask():
    """Subcarriers that actually carry bits: all but the nulled DC and the pilot
    comb. Capacity sums over these."""
    mask = np.ones(NUM_SUBCARRIERS, dtype=bool)
    mask[radar_comb_mask()] = False
    if DC_SUPPRESSION:
        mask[dc_subcarrier_index()] = False
    return mask


def subcarrier_frequencies():
    """Baseband centre frequency of each subcarrier, (k - N/2) * SCS in Hz.

    Feeds the frequency-selective channel H(f_k) = sum_p a_p exp(-2j*pi*f_k*tau_p).
    Only frequency differences matter there, so the grid is centred to line its DC
    bin up with the subcarrier data_subcarrier_mask() nulls.
    """
    return (np.arange(NUM_SUBCARRIERS) - NUM_SUBCARRIERS // 2) * SUBCARRIER_SPACING


def make_ofdm_waveform():
    """The HermesPy OFDMWaveform the sensing OFDMRadar transmits. The comms side
    only needs the numerology, not the waveform object.

    In comb mode the grid is a tiled [1 REFERENCE, M-1 DATA] pattern. DC suppression
    is off there to keep the comb uniform in frequency, which sensing.py's sparse
    range IDFT assumes.
    """
    if comb_enabled():
        M = RADAR_COMB_SPACING
        resource = GridResource(
            repetitions=NUM_SUBCARRIERS // M,
            prefix_type=PrefixType.CYCLIC,
            prefix_ratio=PREFIX_RATIO,
            elements=[GridElement(ElementType.REFERENCE, 1), GridElement(ElementType.DATA, M - 1)],
        )
        return OFDMWaveform(
            grid_resources=[resource],
            grid_structure=[SymbolSection(NUM_OFDM_SYMBOLS, [0])],
            num_subcarriers=NUM_SUBCARRIERS,
            dc_suppression=False,
            pilot_sequence=UniformPilotSymbolSequence(RADAR_PILOT_SYMBOL),
        )

    resource = GridResource(
        prefix_type=PrefixType.CYCLIC,
        prefix_ratio=PREFIX_RATIO,
        elements=[GridElement(ElementType.DATA, NUM_SUBCARRIERS)],
    )
    return OFDMWaveform(
        grid_resources=[resource],
        grid_structure=[SymbolSection(NUM_OFDM_SYMBOLS, [0])],
        num_subcarriers=NUM_SUBCARRIERS,
        dc_suppression=DC_SUPPRESSION,
    )
