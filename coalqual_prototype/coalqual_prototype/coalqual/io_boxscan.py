"""
Readers for Geotek BoxScan session folders.

A BoxScan session laid out on disk looks like::

    <session>/
        POINT_<HOLE>.csv         <- one row per scan location: spectra + XRF + colour
        ASD_<HOLE>.csv           <- SWIR/VNIR mineral unmixing per scan location
        LDBHD_<HOLE>.csv         <- linescan-derived borehole data
        RQD_<HOLE>.csv
        <HOLE>.xml
        01_<HOLE>/
            <box>.curation.csv   <- box geometry, track layout, depths
            <box>.curation.jpg   <- full colour image of the core box
            <box>.dryarea.jpg
            <box>.locations.csv  <- which sensors ran, and at which track positions
            <box>.discrete.xrf
            <box>.SP Spectra/
                *.sps                                <- raw spectrometer frames
                hole name unknown_vnir_reflectance   <- ENVI BSQ cube (+ .hdr)
                hole name unknown_swir_reflectance   <- ENVI BSQ cube (+ .hdr)

POINT_<HOLE>.csv is the analysis-ready table and is what the pipeline uses.
It carries a **two-row header**: row 0 holds sparse group names that apply to
all following columns until the next non-empty entry, row 1 holds the
sub-column name (wavelength, element, axis...).
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import WAVELENGTHS

# --------------------------------------------------------------------------
# POINT_<HOLE>.csv
# --------------------------------------------------------------------------
_META_GROUPS = {
    "HOLE_ID", "Project", "Borehole Depth", "Logged Depth", "Section",
    "Section Depth", "Box ID", "Track Number", "Measurement ID", "Time Stamp",
    "Magnetic Susceptibility", "Mag Sus in Air", "Greyscale Reflectance",
    "Munsell Colour", "CIE XYZ Colour Space", "CIE L*a*b* Colour Space",
    "XRF Total Live Time", "XRF Total Counts",
}


def _flatten_two_row_header(h0: list[str], h1: list[str]) -> tuple[list[str], list[str]]:
    """Forward-fill the sparse group row and join it with the sub-column row."""
    groups, current = [], ""
    for cell in h0:
        if cell.strip():
            current = cell.strip()
        groups.append(current)
    names = [g if not s.strip() else f"{g}|{s.strip()}" for g, s in zip(groups, h1)]
    return names, groups


def read_point_csv(path: str | Path) -> pd.DataFrame:
    """Read one POINT_<HOLE>.csv into a tidy, fully numeric-typed frame.

    Returns a DataFrame with one row per scan location and these column
    families (prefix -> meaning):

        meta_*   scan identity, depth, track, timestamp
        R<nm>    reflectance in percent, 350..2500 nm on a 10 nm grid
        xrf_*    pXRF element concentration in ppm (errors dropped)
        col_*    CIE XYZ / L*a*b* / greyscale / Munsell
        magsus   magnetic susceptibility, 1e-5 SI
    """
    path = Path(path)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 3:
        raise ValueError(f"{path} has no data rows")

    names, groups = _flatten_two_row_header(rows[0], rows[1])
    raw = pd.DataFrame(rows[2:], columns=names)
    raw = raw.replace(r"^\s*$", np.nan, regex=True)

    out = pd.DataFrame(index=raw.index)

    # ---- identity / geometry -------------------------------------------
    def _num(col):
        return pd.to_numeric(raw[col], errors="coerce") if col in raw else np.nan

    out["hole_id"] = raw.get("HOLE_ID")
    out["project"] = raw.get("Project")
    out["measurement_id"] = raw.get("Measurement ID")
    out["timestamp"] = raw.get("Time Stamp")
    out["box_id"] = _num("Box ID")
    out["section"] = _num("Section")
    out["track"] = _num("Track Number")
    out["borehole_depth_m"] = _num("Borehole Depth|m")
    out["logged_depth_m"] = _num("Logged Depth|m")
    out["section_depth_cm"] = _num("Section Depth|cm")

    # ---- physical / colour ---------------------------------------------
    out["magsus"] = _num("Magnetic Susceptibility|x10^-5 SI")
    out["col_grey"] = _num("Greyscale Reflectance")
    out["munsell"] = raw.get("Munsell Colour")
    for src, dst in [
        ("CIE XYZ Colour Space|X", "col_X"), ("CIE XYZ Colour Space|Y", "col_Y"),
        ("CIE XYZ Colour Space|Z", "col_Z"), ("CIE L*a*b* Colour Space|L*", "col_L"),
        ("CIE L*a*b* Colour Space|a*", "col_a"), ("CIE L*a*b* Colour Space|b*", "col_b"),
    ]:
        out[dst] = _num(src)

    # ---- reflectance ----------------------------------------------------
    refl_cols = [c for c in raw.columns if c.startswith("Reflectance (nm)|")]
    wl_found = np.array([int(c.split("|")[1]) for c in refl_cols])
    order = np.argsort(wl_found)
    refl_cols = [refl_cols[i] for i in order]
    wl_found = wl_found[order]
    R = raw[refl_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    # Interpolate onto the canonical grid so different firmware exports align.
    if not np.array_equal(wl_found, WAVELENGTHS):
        R = np.vstack([np.interp(WAVELENGTHS, wl_found, row) for row in R])
    refl_block = pd.DataFrame(R, columns=[f"R{w}" for w in WAVELENGTHS], index=raw.index)

    # ---- XRF ------------------------------------------------------------
    xrf_block = {}
    for c in raw.columns:
        if c.startswith("XRF Elements(ppm)|") and "(Value)" in c:
            el = c.split("|")[1].replace(" (Value)", "").strip()
            xrf_block[f"xrf_{el}"] = pd.to_numeric(raw[c], errors="coerce")
    xrf_block["xrf_live_time"] = _num("XRF Total Live Time|Seconds")
    xrf_block["source_file"] = str(path)

    return pd.concat(
        [out, refl_block, pd.DataFrame(xrf_block, index=raw.index)], axis=1
    ).copy()


def read_asd_csv(path: str | Path) -> pd.DataFrame:
    """Read ASD_<HOLE>.csv (SWIR/VNIR mineral unmixing) keyed by SPS filename."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "SPS Filename" in df.columns:
        df = df.rename(columns={"SPS Filename": "sps_name"})
    keep = ["sps_name", "swir-assemblage", "vnir-assemblage"]
    pct = [c for c in df.columns if c.endswith("%")]
    cols = [c for c in keep if c in df.columns] + pct
    out = df[cols].copy()
    out.columns = [
        c if c in keep else "min_" + c.replace(" %", "").replace(" ", "_")
        for c in out.columns
    ]
    return out


# --------------------------------------------------------------------------
# ENVI cubes
# --------------------------------------------------------------------------
_HDR_NUM = re.compile(r"^\s*([a-z ]+?)\s*=\s*([^\{].*?)\s*$", re.I)


def read_envi_header(path: str | Path) -> dict:
    """Minimal ENVI .hdr parser: scalar keys plus the wavelength vector."""
    text = Path(path).read_text(errors="ignore")
    hdr: dict = {}
    for line in text.splitlines():
        m = _HDR_NUM.match(line)
        if m:
            k, v = m.group(1).strip().lower(), m.group(2).strip()
            try:
                hdr[k] = int(v)
            except ValueError:
                try:
                    hdr[k] = float(v)
                except ValueError:
                    hdr[k] = v
    m = re.search(r"wavelength\s*=\s*\{(.*?)\}", text, re.S | re.I)
    if m:
        hdr["wavelength"] = np.array(
            [float(x) for x in m.group(1).replace("\n", "").split(",") if x.strip()]
        )
    return hdr


def read_envi_cube(bin_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an ENVI BSQ float32 cube, returning (cube[band, line, sample], wavelengths)."""
    bin_path = Path(bin_path)
    hdr_path = bin_path.with_suffix(bin_path.suffix + ".hdr")
    if not hdr_path.exists():
        hdr_path = Path(str(bin_path) + ".hdr")
    hdr = read_envi_header(hdr_path)

    dtype = {1: "u1", 2: "<i2", 3: "<i4", 4: "<f4", 5: "<f8", 12: "<u2"}[int(hdr["data type"])]
    nb, nl, ns = int(hdr["bands"]), int(hdr["lines"]), int(hdr["samples"])
    arr = np.fromfile(bin_path, dtype=dtype)
    interleave = str(hdr.get("interleave", "bsq")).lower()
    if interleave == "bsq":
        cube = arr.reshape(nb, nl, ns)
    elif interleave == "bil":
        cube = arr.reshape(nl, nb, ns).transpose(1, 0, 2)
    else:  # bip
        cube = arr.reshape(nl, ns, nb).transpose(2, 0, 1)
    return cube.astype(float), hdr.get("wavelength", np.arange(nb, dtype=float))


def envi_to_point_spectra(session_box_dir: str | Path) -> pd.DataFrame | None:
    """Merge the VNIR + SWIR ENVI cubes of one box into 350-2500 nm point spectra.

    The BoxScan spectrophotometer writes one *line* per scan location; the
    ``samples`` axis carries duplicated frames, so it is averaged out.
    """
    d = Path(session_box_dir)
    vnir = next(d.glob("*vnir_reflectance"), None)
    swir = next(d.glob("*swir_reflectance"), None)
    if vnir is None or swir is None:
        return None

    cv, wv = read_envi_cube(vnir)
    cs, ws = read_envi_cube(swir)
    sv = np.nanmean(cv, axis=2).T          # (line, band)
    ss = np.nanmean(cs, axis=2).T

    wl_all = np.concatenate([wv, ws])
    spec_all = np.hstack([sv, ss])
    order = np.argsort(wl_all)
    wl_all, spec_all = wl_all[order], spec_all[:, order]

    grid = np.vstack([np.interp(WAVELENGTHS, wl_all, row) for row in spec_all])
    out = pd.DataFrame(grid, columns=[f"R{w}" for w in WAVELENGTHS])
    out.insert(0, "line_index", np.arange(len(out)))
    out.insert(0, "box_dir", str(d))
    return out


# --------------------------------------------------------------------------
# Curation metadata
# --------------------------------------------------------------------------
def read_curation_csv(path: str | Path) -> dict:
    """Parse the key/value header block of <box>.curation.csv."""
    meta: dict = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.reader(fh):
            if len(row) < 2 or not row[0].strip():
                continue
            if row[0].strip().upper() in {"META KEY", "CURATION TYPE"}:
                continue
            key, val = row[0].strip(), row[1].strip()
            try:
                meta[key] = float(val)
            except ValueError:
                meta[key] = val
    return meta


def read_locations_csv(path: str | Path) -> pd.DataFrame:
    """Parse <box>.locations.csv -> sensor availability and track scan positions."""
    rows = list(csv.reader(open(path, newline="", encoding="utf-8-sig")))
    start = next(i for i, r in enumerate(rows) if r and r[0].strip().upper() == "SENSOR TYPE")
    header = [c.strip() for c in rows[start]]
    body = [r for r in rows[start + 1:] if r and r[0].strip()]
    df = pd.DataFrame(body, columns=header[: len(body[0])] if body else header)
    for c in df.columns:
        if "TRACK" in c.upper():
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("SENSOR AVAILABLE", "SENSOR SELECTED"):
        if c in df:
            df[c] = df[c].str.strip().str.lower().eq("true")
    return df


# --------------------------------------------------------------------------
# Session-level loader
# --------------------------------------------------------------------------
def load_sessions(root: str | Path, pattern: str = "*") -> pd.DataFrame:
    """Load every BoxScan session under ``root`` into one feature table.

    Rows with no XRF measurement (aborted point) are kept; downstream code
    decides whether to impute or drop them.
    """
    root = Path(root)
    frames = []
    for point_csv in sorted(root.glob(f"{pattern}/POINT_*.csv")):
        session = point_csv.parent.name
        df = read_point_csv(point_csv).copy()
        df.insert(0, "session", session)

        asd = next(point_csv.parent.glob("ASD_*.csv"), None)
        if asd is not None and "measurement_id" in df:
            a = read_asd_csv(asd)
            # Measurement IDs look like 01_BH1_0042_00610110; the ASD key is
            # 0005_42_000610110 -> match on the trailing digit run.
            def _tail(s):
                m = re.findall(r"(\d{6,})", str(s))
                return m[-1][-8:] if m else None
            df["_k"] = df["measurement_id"].map(_tail)
            a["_k"] = a["sps_name"].map(_tail)
            df = df.merge(a.drop(columns=["sps_name"]), on="_k", how="left").drop(columns="_k")

        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No POINT_*.csv found under {root}")

    out = pd.concat(frames, ignore_index=True)
    # Stable, human-readable sample key used to join laboratory results.
    out.insert(0, "sample_id", [
        f"{s}__{m}" for s, m in zip(out["session"], out["measurement_id"])
    ])
    return out


def write_label_template(df: pd.DataFrame, path: str | Path) -> Path:
    """Emit an empty proximate/CV template keyed by sample_id for the lab to fill."""
    from .config import TARGETS
    cols = ["sample_id", "session", "measurement_id", "box_id", "track",
            "borehole_depth_m", "section_depth_cm"]
    tpl = df[[c for c in cols if c in df.columns]].copy()
    for t in TARGETS:
        tpl[t] = np.nan
    tpl["basis"] = "air_dried"
    tpl["lab_ref"] = ""
    path = Path(path)
    tpl.to_csv(path, index=False)
    return path
