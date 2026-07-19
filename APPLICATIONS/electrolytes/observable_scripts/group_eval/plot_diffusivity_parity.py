"""Plot diffusivity parity plots (log + linear) from a diffusivity_with_exp_all.csv.

Rules:
  - cation / anion properties: exclude systems with concentration_M < 0.5
  - solvent properties: all systems included
  - Color by (cation, anion, solvent, temperature_K) so multi-temperature
    systems (e.g. Li-PF6-DME at 273/298/323 K) each get a distinct color.

Usage:
  python plot_diffusivity_parity.py <diffusivity_with_exp_all.csv> [--output-dir <dir>]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from parity_plot import plot_group_parity
from properties import PROPERTIES


def run(csv_path, output_dir=None):
    csv_path = Path(csv_path)
    if output_dir is None:
        output_dir = csv_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_all  = pd.read_csv(csv_path)
    df_hico = df_all[df_all['concentration_M'] >= 0.5]

    tmp_all  = output_dir / '_tmp_all.csv'
    tmp_hico = output_dir / '_tmp_hico.csv'
    df_all.to_csv(tmp_all,  index=False)
    df_hico.to_csv(tmp_hico, index=False)

    JOBS = [
        ('diffusivity_cation',              tmp_hico),
        ('diffusivity_anion',               tmp_hico),
        ('diffusivity_solvent',             tmp_all),
        ('diffusivity_cation_uncorrected',  tmp_hico),
        ('diffusivity_anion_uncorrected',   tmp_hico),
        ('diffusivity_solvent_uncorrected', tmp_all),
    ]

    for prop, csv in JOBS:
        cfg = PROPERTIES[prop]
        for log_scale, suffix in [(True, 'log'), (False, 'linear')]:
            plot_group_parity(
                [str(csv)],
                exp_col=cfg['exp_col'],
                sim_col=cfg['sim_col'],
                labels=['PAINN'],
                output=str(output_dir / f'diffusivity_parity_{prop}_{suffix}.png'),
                xlabel=cfg['xlabel'],
                ylabel=cfg['ylabel'],
                title=f'{prop.replace("_", " ")} ({suffix})',
                log_scale=log_scale,
                annotate_meta=True,
            )

    tmp_all.unlink(missing_ok=True)
    tmp_hico.unlink(missing_ok=True)
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='Path to diffusivity_with_exp_all.csv')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()
    run(args.csv, args.output_dir)
