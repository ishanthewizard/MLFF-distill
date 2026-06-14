"""
Parity plot: MD-computed conductivity (Onsager / Nernst-Einstein) vs.
experimental conductivity from cleaned_data.csv.

Two points are matched if they have the same (cation, anion, solvent,
concentration) and |T_MD - T_exp| <= 2 K.
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(__file__)
MD_CSV = os.path.join(BASE, 'results', 'conductivity_results.csv')
EXP_CSV = '/home/yuejian/project/byteff2/draft/ref/cleaned_all_properties/cleaned_data.csv'
OUT_PNG = os.path.join(BASE, 'results', 'parity_plot.png')
T_TOL = 2.0  # K


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    md_rows = load_csv(MD_CSV)
    exp_rows = load_csv(EXP_CSV)

    pairs = []  # (exp_conductivity_uS_cm, md_conductivity_uS_cm, label, ensemble, dT)
    for md in md_rows:
        key = (md['cation'], md['anion'], md['solvent'], float(md['concentration']))
        T_md = float(md['temperature_K'])
        for exp in exp_rows:
            if (exp['cation'], exp['anion'], exp['solvent'], float(exp['concentration'])) != key:
                continue
            T_exp = float(exp['temperature'])
            dT = T_md - T_exp
            if abs(dT) > T_TOL:
                continue
            exp_cond = float(exp['conductivity'])  # uS/cm
            md_onsager = float(md['conductivity_onsager_mS_cm']) * 1000  # -> uS/cm
            md_ne = float(md['conductivity_NE_mS_cm']) * 1000
            label = f"{md['cation']}{md['anion']}/{md['solvent']} {md['concentration']}M ({md['ensemble']})"
            pairs.append((exp_cond, md_onsager, md_ne, label, dT, float(md['concentration'])))

    if not pairs:
        print("No matching (system, temperature) pairs found.")
        return

    exp_vals = [p[0] for p in pairs]
    onsager_vals = [p[1] for p in pairs]
    ne_vals = [p[2] for p in pairs]
    conc_vals = [p[5] for p in pairs]

    lo = min(exp_vals + onsager_vals + ne_vals) * 0.5
    hi = max(exp_vals + onsager_vals + ne_vals) * 2

    def mae_r2(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mae = np.mean(np.abs(y_true - y_pred))
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - ss_res / ss_tot
        return mae, r2

    mae_o, r2_o = mae_r2(exp_vals, onsager_vals)
    mae_ne, r2_ne = mae_r2(exp_vals, ne_vals)

    lo_lin = 0
    hi_lin = max(exp_vals + onsager_vals + ne_vals) * 1.05

    for log_scale, suffix in [(True, ''), (False, '_linear')]:
        fig, ax = plt.subplots(figsize=(8.5, 7))
        if log_scale:
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='parity')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        else:
            ax.plot([lo_lin, hi_lin], [lo_lin, hi_lin], 'k--', lw=1, label='parity')
            ax.set_xlim(lo_lin, hi_lin)
            ax.set_ylim(lo_lin, hi_lin)

        norm = plt.Normalize(min(conc_vals), max(conc_vals))
        sc_o = ax.scatter(exp_vals, onsager_vals, c=conc_vals, cmap='Blues', norm=norm,
                           label=f'Onsager (MAE={mae_o:.0f}, R2={r2_o:.2f})',
                           alpha=0.8, edgecolors='gray', linewidths=0.5)
        sc_ne = ax.scatter(exp_vals, ne_vals, c=conc_vals, cmap='Oranges', norm=norm, marker='^',
                            label=f'Nernst-Einstein (MAE={mae_ne:.0f}, R2={r2_ne:.2f})',
                            alpha=0.8, edgecolors='gray', linewidths=0.5)
        fig.colorbar(sc_o, ax=ax, label='Conc. (M), Onsager', shrink=0.4, fraction=0.04, pad=0.04)
        fig.colorbar(sc_ne, ax=ax, label='Conc. (M), NE', shrink=0.4, fraction=0.04, pad=0.18)

        ax.set_xlabel('Experimental conductivity (uS/cm)')
        ax.set_ylabel('MD conductivity (uS/cm)')
        ax.set_title(f'MD vs experimental conductivity (|dT| <= {T_TOL} K)')
        ax.legend()
        ax.set_aspect('equal')
        fig.tight_layout()
        out_path = OUT_PNG.replace('.png', f'{suffix}.png')
        fig.savefig(out_path, dpi=150)
        print(f"Wrote {out_path} with {len(pairs)} matched points")

    print(f"Onsager: MAE={mae_o:.2f} uS/cm, R2={r2_o:.4f}")
    print(f"Nernst-Einstein: MAE={mae_ne:.2f} uS/cm, R2={r2_ne:.4f}")

    # also dump a table
    out_csv = os.path.join(BASE, 'results', 'parity_data.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['system', 'dT_K', 'exp_conductivity_uS_cm',
                          'md_onsager_uS_cm', 'md_NE_uS_cm', 'concentration_M'])
        for x, y_o, y_ne, label, dT, conc in pairs:
            writer.writerow([label, f'{dT:.2f}', x, y_o, y_ne, conc])
    print(f"Wrote {out_csv}")


if __name__ == '__main__':
    main()
