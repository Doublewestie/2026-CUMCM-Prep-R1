"""
step2_shared.py — Q2 shared utilities
======================================
Centralized raw data loading for all step2_* scripts.
"""
import os
import json
import numpy as np
import pandas as pd
from step0_config import DATA_DIR_2025

EPS = 1e-3
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
AR6_JSON = os.path.join(OUTPUT_DIR, "step2_final_results.json")
CLEAN_CSV = os.path.join(OUTPUT_DIR, "clean_data.csv")


def load_raw_filt_data():
    """Load 2025 raw Excel data, preserving original FILT_NTU sequence integrity."""
    FILES = sorted([f for f in os.listdir(DATA_DIR_2025) if f.endswith('.xlsx')])
    RENAME = {
        'RIVER LEVEL': 'RIVER_LEVEL', 'R/W FLOW': 'RW_FLOW', 'R/W NTU': 'RW_NTU',
        'R/W CLR': 'RW_CLR', 'FILT. NTU': 'FILT_NTU', 'C/W WELL LEVEL': 'CW_WELL_LEVEL',
        'T/W FLOW': 'TW_FLOW', 'ALUM': 'ALUM', 'NTU': 'NTU', 'R/W PH': 'RW_PH',
        'R/W PUMP DUTY': 'RW_PUMP_DUTY', 'T/W PUMP DUTY': 'TW_PUMP_DUTY',
        'R/W PH': 'RW_PH', 'F/RIDE': 'F_RIDE', 'PH': 'PH', 'CLR': 'CLR', 'CL2': 'CL2',
    }
    NUM_COLS = ['RIVER_LEVEL', 'RW_FLOW', 'RW_NTU', 'RW_CLR', 'RW_PH',
                'FILT_NTU', 'CW_WELL_LEVEL', 'TW_FLOW', 'ALUM', 'NTU']
    data_all = []
    for fname in FILES:
        fp = os.path.join(DATA_DIR_2025, fname)
        dfm = pd.read_excel(fp, skiprows=1 if 'Jan' in fname else 0)
        dfm.rename(columns={k: v for k, v in RENAME.items() if k in dfm.columns}, inplace=True)
        newcols = [str(c).strip().replace('.', '_').replace(' ', '_') for c in dfm.columns]
        dfm.columns = newcols
        for c in NUM_COLS:
            if c in dfm.columns:
                dfm[c] = pd.to_numeric(dfm[c], errors='coerce')
        data_all.append(dfm)
    data = pd.concat(data_all, ignore_index=True)
    return data.dropna(subset=['FILT_NTU']).reset_index(drop=True)


class AR6Predictor:
    """log(FILT+eps) AR(6) + RidgeCV predictor for FILT forecasting.

    Loads pretrained coefficients from step2_final_results.json.
    Input: log-FILT history (at least 6 values). Output: FILT in original NTU space.
    """
    def __init__(self, json_path=None):
        path = json_path or AR6_JSON
        if not os.path.exists(path):
            raise FileNotFoundError(f"AR6 model not found: {path}. Run step2.5_logar_final.py first.")
        with open(path, encoding="utf-8") as f:
            ar = json.load(f)
        self.coefs = np.array([ar["coefficients"][f"AR_lag_{lag}"] for lag in range(1, 7)])
        df = pd.read_csv(CLEAN_CSV)
        log_filt = np.log(df["FILT_NTU"].values.astype(float) + self._eps())
        self.intercept = np.mean(log_filt) * (1 - self.coefs.sum())
        self._eps_val = self._eps()

    def _eps(self):
        return 1e-3

    def predict(self, history, n_steps):
        """Rolling AR(6) prediction.

        Args:
            history: list of log(FILT+eps) values (float), at least 6 values.
            n_steps: number of future steps to forecast.

        Returns:
            np.array of FILT predictions in original NTU space, length n_steps.
        """
        series = list(history)
        for _ in range(n_steps):
            y = self.intercept + sum(self.coefs[i] * series[-6 + i] for i in range(6))
            series.append(y)
        preds = np.exp(np.array(series[-n_steps:])) - self._eps_val
        return np.clip(preds, 0, None)
