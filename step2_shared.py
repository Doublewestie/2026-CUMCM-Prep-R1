"""
step2_shared.py — Q2 shared utilities
======================================
Centralized raw data loading for all step2_* scripts.
"""
import os
import pandas as pd
from step0_config import DATA_DIR_2025


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
