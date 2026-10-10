"""
Central configuration for the coal-quality pipeline.

Everything that is instrument- or site-specific lives here so that swapping
simulated data for real laboratory data touches exactly one file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
MODEL_DIR = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

for _d in (DATA_DIR, OUT_DIR, MODEL_DIR, FIG_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Spectral grid
# --------------------------------------------------------------------------
# The Geotek BoxScan spectrophotometer (ASD-class, 350-2500 nm) is exported by
# the BoxScan software on a 10 nm grid inside POINT_<hole>.csv.
WL_MIN, WL_MAX, WL_STEP = 350, 2500, 10
WAVELENGTHS = np.arange(WL_MIN, WL_MAX + WL_STEP, WL_STEP)  # 216 bands

# Regions to drop before modelling.
#   * < 420 nm  : silicon detector edge, low SNR on very dark targets
#   * > 2450 nm : InGaAs detector edge
#   * 990-1010  : VNIR/SWIR1 detector join
#   * 1790-1810 : SWIR1/SWIR2 detector join
# The screen_bands() helper additionally removes any band whose empirical
# noise level exceeds NOISE_LEVEL_MAX (Mao et al. 2024, eqs. 3-5).
EXCLUDED_REGIONS = [(350, 410), (985, 1015), (1785, 1815), (2455, 2500)]
NOISE_LEVEL_MAX = 0.01  # 1 % relative noise

# --------------------------------------------------------------------------
# Diagnostic absorption features (Cloutis 2003; Fysh 1985; Mao et al. 2024)
# --------------------------------------------------------------------------
# centre_nm -> (left shoulder, right shoulder, label, chemistry)
DIAGNOSTIC_FEATURES = {
    650:  (600, 720, "Fe3+_650", "Fe3+ crystal-field (goethite/hematite)"),
    900:  (820, 1000, "Fe_900", "Fe2+/Fe3+ electronic transition"),
    1150: (1100, 1220, "Fe2_1150", "Electron leaps in Fe2+-dominated metals"),
    1400: (1330, 1470, "OH_1400", "OH / H2O first overtone -> moisture + clay"),
    1700: (1650, 1760, "CH_1700", "CH2/CH3 first overtone -> volatile matter"),
    1900: (1830, 1970, "H2O_1900", "H2O combination band -> moisture"),
    2150: (2100, 2185, "arC_2150", "Aromatic C-H + C-C -> fixed carbon"),
    2200: (2160, 2245, "AlOH_2200", "Al-OH (kaolinite/illite) -> ash"),
    2300: (2260, 2360, "CH_2300", "CH2/CH3 asym stretch + bend -> volatiles"),
    2350: (2300, 2400, "CO3_2350", "Carbonate / aliphatic CH"),
}

# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------
TARGETS = ["MC", "Ash", "VM", "FC", "CV"]

TARGET_META = {
    "MC":  dict(label="Moisture",       unit="%",       standard="IS 1350 Pt.1 / ISO 11722"),
    "Ash": dict(label="Ash",            unit="%",       standard="IS 1350 Pt.1 / ISO 1171"),
    "VM":  dict(label="Volatile matter", unit="%",      standard="IS 1350 Pt.1 / ISO 562"),
    "FC":  dict(label="Fixed carbon",   unit="%",       standard="by difference"),
    "CV":  dict(label="Gross calorific value", unit="kcal/kg", standard="IS 1350 Pt.2"),
}

# CIL grade bands (Gross Calorific Value, kcal/kg) used for grade declaration.
CIL_GRADES = [
    ("G1", 7000, 99999), ("G2", 6700, 7000), ("G3", 6400, 6700),
    ("G4", 6100, 6400),  ("G5", 5800, 6100), ("G6", 5500, 5800),
    ("G7", 5200, 5500),  ("G8", 4900, 5200), ("G9", 4600, 4900),
    ("G10", 4300, 4600), ("G11", 4000, 4300), ("G12", 3700, 4000),
    ("G13", 3400, 3700), ("G14", 3100, 3400), ("G15", 2800, 3100),
    ("G16", 2500, 2800), ("G17", 2200, 2500),
]


def gcv_to_grade(gcv_kcal: float) -> str:
    """Map a gross calorific value (kcal/kg, GAR basis) to a CIL grade band."""
    if not np.isfinite(gcv_kcal):
        return "NA"
    for name, lo, hi in CIL_GRADES:
        if lo <= gcv_kcal < hi:
            return name
    return "UG" if gcv_kcal < 2200 else "G1"


# --------------------------------------------------------------------------
# Lab reference repeatability (used to inject realistic label noise and to
# define the "you cannot beat the lab" performance floor)
# --------------------------------------------------------------------------
LAB_REPEATABILITY = {
    "MC": 0.15,    # %, absolute
    "Ash": 0.30,   # %
    "VM": 0.35,    # %
    "FC": 0.50,    # % (propagated, by difference)
    "CV": 60.0,    # kcal/kg
}

# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_FRACTION = 0.25          # SPXY calibration / prediction split
N_MCCV_OUTLIER = 200          # Monte-Carlo CV repeats for outlier detection
OUTLIER_SIGMA = 3.0           # residual threshold in robust sigma units


@dataclass
class FeatureBlocks:
    """Which feature families to feed the models."""
    spectra: bool = True         # selected reflectance bands
    derivative: bool = True      # 1st-derivative at selected bands
    band_depths: bool = True     # continuum-removed absorption depths
    ratios: bool = True          # diagnostic band ratios
    xrf: bool = True             # pXRF elemental chemistry
    colour: bool = True          # CIE L*a*b* + greyscale
    texture: bool = False        # GLCM/Gabor from the RGB core image (optional)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


DEFAULT_BLOCKS = FeatureBlocks()
