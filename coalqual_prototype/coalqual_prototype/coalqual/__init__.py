"""
coalqual - rapid, non-destructive coal quality estimation from Geotek BoxScan
VIS-SWIR reflectance + pXRF, for the CIL-IMin hackathon prototype.

Typical use::

    from coalqual import io_boxscan, simulate, pipeline, visualize

    real = io_boxscan.load_sessions("data/raw")          # 28 scan locations
    df   = simulate.generate_dataset(200, real_df=real)  # placeholder labels
    summary, full, store = pipeline.run_all(df)
    visualize.plot_scatter_grid(store)
"""
from . import (config, features, io_boxscan, models, pipeline, preprocess,
               selection, simulate, validation, visualize)

__all__ = [
    "config", "io_boxscan", "preprocess", "features", "selection",
    "models", "validation", "pipeline", "simulate", "visualize",
]
__version__ = "0.3.0"
