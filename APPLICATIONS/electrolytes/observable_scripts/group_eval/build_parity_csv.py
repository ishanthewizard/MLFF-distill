#!/usr/bin/env python3
"""Batch driver: walk a directory of production systems and build
``conductivity_parity.csv``.

Computes conductivity (via ``compute.run_onsager_conductivity`` /
``compute.run_onsager_conductivity_gromacs``) and density for every system
under an OPLS ``npt``/``nvt`` production root or a PAINN
``simulation_tf32`` root, merges with experimental conductivity/viscosity/
density from ``conductivity.csv`` (matched on cation/anion/solvent/
concentration/temperature), and writes a flat csv suitable for ``plot.py``.

The output schema is identical across sources/ensembles, so multiple
``conductivity_parity.csv`` files (e.g. OPLS npt, OPLS nvt, PAINN) can be
combined later in the same parity plot.
"""

import glob
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "conductivity"))
import compute

logger = logging.getLogger(__name__)


ION_MAP = {
    'lipf6': ('Li', 'PF6'),
    'napf6': ('Na', 'PF6'),
    'naotf': ('Na', 'OTf'),
}

# directory-name solvent token -> exp-table solvent name
SOLVENT_MAP = {
    'dme':     'DME',
    'diglyme': 'DEGDME',
    'tegdme':  'TEGDME',
    'tgdme':   'TEGDME',
    'pc':      'PC',
}

# directory-name solvent token -> component_dictionary solvent key (for ASE traj)
SOLVENT_DICT_KEY = {
    'dme':     'DME',
    'diglyme': 'Diglyme',
    'tegdme':  'TGDME',
    'tgdme':   'TGDME',
    'pc':      'PC',
}

OPLS_DIRNAME_RE = re.compile(r'^([a-z0-9]+)_([a-z]+)_([\d.]+)M_(\d+)K$')
PAINN_RUN_RE = re.compile(r'^(npt|nvt)_(\d+(?:_\d+)?)M_(\d+)K')


def _uS_from_mS(x):
    return None if x is None else x * 1000.0


def load_exp_csv(path):
    df = pd.read_csv(path)
    df = df.rename(columns={
        'concentration (M)': 'concentration',
        'temperature (K)': 'temperature_K',
        'IC2 (uS/cm)': 'exp_conductivity_uS_cm',
        'Kinematic Viscosity (mm^2/s)': 'exp_kinematic_viscosity_mm2_s',
        'Density (g/mL)': 'exp_density_g_mL',
    })
    return df


def match_exp(exp_df, cation, anion, solvent, concentration, T_K, T_tol=5.0):
    """Return the closest-temperature exp row matching (cation, anion, solvent,
    concentration), within T_tol Kelvin, or None."""
    cand = exp_df[
        (exp_df['cation'] == cation) &
        (exp_df['anion'] == anion) &
        (exp_df['solvent'] == solvent) &
        (np.isclose(exp_df['concentration'].astype(float), concentration, atol=1e-6)) &
        exp_df['exp_conductivity_uS_cm'].notna()
    ]
    if cand.empty:
        return None
    dT = (cand['temperature_K'].astype(float) - T_K).abs()
    if dT.min() > T_tol:
        return None
    return cand.loc[dT.idxmin()]


def _row_with_exp(row, exp_df, T_tol=5.0):
    exp = match_exp(exp_df, row['cation'], row['anion'], row['solvent'],
                     row['concentration'], row['temperature_K'], T_tol=T_tol)
    if exp is not None:
        row['exp_conductivity_uS_cm'] = float(exp['exp_conductivity_uS_cm'])
        row['exp_kinematic_viscosity_mm2_s'] = (
            float(exp['exp_kinematic_viscosity_mm2_s'])
            if pd.notna(exp['exp_kinematic_viscosity_mm2_s']) else np.nan
        )
        row['exp_density_g_mL'] = (
            float(exp['exp_density_g_mL']) if pd.notna(exp['exp_density_g_mL']) else np.nan
        )
        row['exp_temperature_K'] = float(exp['temperature_K'])
    else:
        row['exp_conductivity_uS_cm'] = np.nan
        row['exp_kinematic_viscosity_mm2_s'] = np.nan
        row['exp_density_g_mL'] = np.nan
        row['exp_temperature_K'] = np.nan
    return row


def process_opls(root, ensemble, exp_df, viscosity_cP=1.0):
    """root: .../simulation_results/OPLS ; ensemble: 'npt' or 'nvt'."""
    root = Path(root)
    ens_dir = root / ensemble
    rows = []
    for sys_dir in sorted(ens_dir.glob('*/')):
        name = sys_dir.name
        m = OPLS_DIRNAME_RE.match(name)
        if not m:
            continue
        salt, solvent_tok, conc, temp_label = m.groups()
        if salt not in ION_MAP or solvent_tok not in SOLVENT_MAP:
            print(f'[OPLS/{ensemble}] {name}: SKIP (unknown salt/solvent)')
            continue
        cation, anion = ION_MAP[salt]
        solvent = SOLVENT_MAP[solvent_tok]
        concentration = float(conc)
        T_nominal = float(temp_label)

        try:
            res = compute.run_onsager_conductivity_gromacs(sys_dir, ensemble, viscosity_cP=viscosity_cP)
            xtc_path = sys_dir / f'{ensemble}.xtc'
            tpr_path = sys_dir / f'{ensemble}.tpr'
            dens_mean, dens_std = compute.compute_density(
                xtc_path, dt_fs=1000.0, skip_ns=2.0, window_ns=1e6, topology=tpr_path,
            )
        except Exception as e:
            print(f'[OPLS/{ensemble}] {name}: FAILED ({e})')
            continue

        row = {
            'system': name,
            'source': 'OPLS',
            'ensemble': ensemble,
            'cation': cation,
            'anion': anion,
            'solvent': solvent,
            'concentration': concentration,
            'temperature_K': res['T_K'] if res['T_K'] is not None else T_nominal,
            'sim_conductivity_onsager_uS_cm': _uS_from_mS(res['sigma_onsager_mS_cm']),
            'sim_conductivity_NE_uS_cm': _uS_from_mS(res['sigma_NE_mS_cm']),
            'sim_density_g_mL': dens_mean,
        }
        rows.append(_row_with_exp(row, exp_df))
        print(f'[OPLS/{ensemble}] {name}: done')
    return rows


def process_painn(root, exp_df, viscosity_cP=1.0):
    """root: .../simulation_tf32 (one subdir per cation_anion_solvent system)."""
    root = Path(root)
    rows = []
    for outer in sorted(root.glob('*/')):
        m = re.match(r'^([a-z0-9]+)_([a-z]+)$', outer.name)
        if not m:
            continue
        salt, solvent_tok = m.groups()
        if salt not in ION_MAP or solvent_tok not in SOLVENT_MAP:
            print(f'[PAINN] {outer.name}: SKIP (unknown salt/solvent)')
            continue
        cation, anion = ION_MAP[salt]
        solvent = SOLVENT_MAP[solvent_tok]
        solvent_key = SOLVENT_DICT_KEY[solvent_tok]

        for run_dir in sorted(outer.glob('*/')):
            mr = PAINN_RUN_RE.match(run_dir.name)
            if not mr:
                continue
            ensemble, conc_tok, temp_label = mr.groups()
            concentration = float(conc_tok.replace('_', '.'))
            T_K = float(temp_label)

            traj_paths = glob.glob(str(run_dir / '*.traj'))
            if not traj_paths:
                print(f'[PAINN] {run_dir}: SKIP (no .traj)')
                continue
            traj_path = traj_paths[0]

            try:
                res = compute.run_onsager_conductivity(
                    traj_path, cation, anion, solvent_key,
                    dt_fs=100.0, T_K=T_K, viscosity_cP=viscosity_cP,
                )
                dens_mean, dens_std = compute.compute_density(
                    traj_path, dt_fs=100.0, skip_ns=2.0, window_ns=1e6,
                )
            except Exception as e:
                print(f'[PAINN] {run_dir.name}: FAILED ({e})')
                continue

            row = {
                'system': f'{outer.name}_{run_dir.name}',
                'source': 'PAINN',
                'ensemble': ensemble,
                'cation': cation,
                'anion': anion,
                'solvent': solvent,
                'concentration': concentration,
                'temperature_K': T_K,
                'sim_conductivity_onsager_uS_cm': _uS_from_mS(res['sigma_onsager_mS_cm']),
                'sim_conductivity_NE_uS_cm': _uS_from_mS(res['sigma_NE_mS_cm']),
                'sim_density_g_mL': dens_mean,
            }
            rows.append(_row_with_exp(row, exp_df))
            print(f'[PAINN] {run_dir.name}: done')
    return rows


COLUMNS = [
    'system', 'source', 'ensemble', 'cation', 'anion', 'solvent', 'concentration',
    'temperature_K', 'sim_conductivity_onsager_uS_cm', 'sim_conductivity_NE_uS_cm',
    'sim_density_g_mL', 'exp_conductivity_uS_cm', 'exp_kinematic_viscosity_mm2_s',
    'exp_density_g_mL', 'exp_temperature_K',
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, choices=['opls', 'painn'])
    parser.add_argument('--root', required=True,
                         help='OPLS: .../simulation_results/OPLS ; PAINN: .../simulation_tf32')
    parser.add_argument('--ensemble', choices=['npt', 'nvt'], default='npt',
                         help='Only used for --source opls.')
    parser.add_argument('--exp-csv', required=True,
                         help='Path to experimental conductivity.csv')
    parser.add_argument('--output', required=True,
                         help='Path to write conductivity_parity.csv')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    exp_df = load_exp_csv(args.exp_csv)

    if args.source == 'opls':
        rows = process_opls(args.root, args.ensemble, exp_df)
    else:
        rows = process_painn(args.root, exp_df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(out_path, index=False)
    print(f'Wrote {len(df)} rows to {out_path}')


if __name__ == '__main__':
    main()
