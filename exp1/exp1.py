"""
Experiment 1: Environment Setup & Package Version Verification
"""

import numpy as np
import pandas as pd
import matplotlib
import jupyterlab

packages = [
    ("NumPy", "numpy"),
    ("Pandas", "pandas"),
    ("Matplotlib", "matplotlib"),
    ("Seaborn", "seaborn"),
    ("Statsmodels", "statsmodels.api"),
    ("SciPy", "scipy"),
    ("Plotly", "plotly"),
    ("Bokeh", "bokeh"),
    ("JupyterLab", "jupyterlab"),
]

def check_versions():
    for name, pkg_name in packages:
        try:
            mod = __import__(pkg_name, fromlist=["__version__"])
            ver = getattr(mod, "__version__", getattr(mod, "version", "Unknown"))
            print(f"{name} Version: {ver}")
        except ImportError:
            print(f"{name} Version: Not installed")

if __name__ == "__main__":
    check_versions()
