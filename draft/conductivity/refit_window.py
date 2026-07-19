"""
Refit the Onsager transport coefficients / conductivity on a 0.5-15 ns window
(final), reusing the saved MSD + collective cross-displacement arrays produced
by md_craft.py (no trajectory re-read).  Regenerates the cross-displacement and
self-MSD convergence plots, the conductivity parity plot, and report.md.

Validated: refitting the saved arrays on 1-10 ns reproduces md_craft.py exactly
(naotf 4701, napf6 14066 uS/cm).  kappa is COM-invariant (shown in the original
run), so the COM-removed arrays used here give the same kappa as the lab frame.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mdcraft.analysis.transport import (
    calculate_transport_coefficients, calculate_conductivity,
    calculate_transference_numbers)

OUTDIR = "/global/u1/y/yuejian/project/MLFF-distill/draft/conductivity"
REF_CSV = ("/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/"
           "simulation_results/PAINN/electrolytes_data/analysis/20ns_fp32/npt/"
           "group_results/conductivity/conductivity_parity_painn_npt_gt0p1M.csv")

T_K = 298.0
KBT = T_K * 8.314462618e-3                 # kJ/mol
DT_NS = 0.001                              # 1 ps per analysed frame
NS = np.array([17, 17])
Z = np.array([1.0, -1.0])
K2USCM = 1e19 * 1e4                         # mdcraft kappa -> uS/cm

WINDOWS = {"0.5-15ns": (0.5, 15.0), "1-10ns": (1.0, 10.0)}
MAIN = "0.5-15ns"

SYS = {
    "naotf_dme": dict(anion=r"OTf$^-$", ref_key="naotf_dme npt 1.0M 298K", color="tab:blue"),
    "napf6_dme": dict(anion=r"PF$_6^-$", ref_key="napf6_dme npt 1.0M 298K", color="tab:red"),
}


def fit(npz, s_ns, e_ns):
    t = npz["times_ps"]
    cross = npz["cross_collective"] / 6.0          # mdcraft msd_cross (collective/6)
    self_ = npz["msd_self"]
    dims = npz["dims"]
    s, e = int(round(s_ns / DT_NS)), int(round(e_ns / DT_NS))
    L, Lself, D = calculate_transport_coefficients(
        t, cross[:, None, :], self_[:, None, :], NS, dims, KBT, s, e, "linear")
    kap = calculate_conductivity(L, Z)[0] * K2USCM
    tnum = calculate_transference_numbers(L, Z)[0]
    sl = [np.polyfit(np.log(t[s:e]), np.log(self_[i][s:e]), 1)[0] for i in (0, 1)]
    return dict(start=s, stop=e, kappa=kap, L=L[0], D=D[0] * 1e-4, t=tnum, slope=sl)


def plot_cross(name, npz, win, anion):
    t = npz["times_ps"]
    cc, cx, aa = npz["cross_collective"]           # ++, +-, --
    s, e = win["start"], win["stop"]
    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.plot(t, cc, label=r"$\langle\Delta R_+\!\cdot\!\Delta R_+\rangle$ (cation-cation)")
    ax.plot(t, aa, label=rf"$\langle\Delta R_-\!\cdot\!\Delta R_-\rangle$ ({anion}-{anion})")
    ax.plot(t, cx, label=r"$\langle\Delta R_+\!\cdot\!\Delta R_-\rangle$ (cation-anion, cross)")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvspan(t[s], t[e - 1], color="grey", alpha=0.15, label=f"fit window {MAIN}")
    for y, c in ((cc, "C0"), (aa, "C1"), (cx, "C2")):
        p = np.polyfit(t[s:e], y[s:e], 1)
        ax.plot(t, np.polyval(p, t), c, ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("t [ps]")
    ax.set_ylabel(r"collective displacement corr.  $\langle\Delta R_i\!\cdot\!\Delta R_j\rangle$ [$\AA^2$]")
    ax.set_title(f"{name}: cross / self collective displacement (COM-removed)\n"
                 f"fit window {MAIN}; dashed = linear fit")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25); fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"cross_displacement_{name}.png"), dpi=140)
    plt.close(fig)


def plot_self(name, npz, win, anion, kappa):
    t = npz["times_ps"]; ms = npz["msd_self"]
    s, e = win["start"], win["stop"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(t[1:], ms[0][1:], label="Na$^+$ (self)")
    ax.loglog(t[1:], ms[1][1:], label=f"{anion} (self)")
    ax.axvspan(t[s], t[e - 1], color="grey", alpha=0.15, label=f"fit window {MAIN}")
    ax.loglog(t[s:e], ms[0][s] * t[s:e] / t[s], "k:", lw=1, label="slope 1")
    ax.set_xlabel("t [ps]"); ax.set_ylabel(r"self MSD $\langle\Delta r^2\rangle$ [$\AA^2$]")
    ax.set_title(f"{name}: self-MSD (COM-removed)\n$\\kappa$={kappa:.0f} $\\mu$S/cm "
                 f"(fit {MAIN})")
    ax.legend(); ax.grid(True, which="both", alpha=0.25); fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"msd_self_{name}.png"), dpi=140)
    plt.close(fig)


def load_ref():
    import csv
    with open(REF_CSV) as f:
        return {r["system"]: r for r in csv.DictReader(f)}


def parity_plot(rows):
    fig, ax = plt.subplots(figsize=(6.4, 6))
    allv = [v for x in rows for v in (x["mine"], x["ref_ons"], x["ref_ne"], x["exp"])]
    lo, hi = 0.5 * min(allv), 2.0 * max(allv)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="parity (y=x)")
    for x in rows:
        c = x["color"]
        ax.scatter(x["exp"], x["mine"], s=150, marker="o", color=c, edgecolor="k",
                   zorder=3, label=f"{x['name']} - mdcraft Onsager (this work, {MAIN})")
        ax.scatter(x["exp"], x["ref_ons"], s=120, marker="s", color=c, alpha=0.55,
                   edgecolor="k", zorder=2, label=f"{x['name']} - ref Onsager")
        ax.scatter(x["exp"], x["ref_ne"], s=100, marker="^", color=c, alpha=0.30,
                   edgecolor="k", zorder=2, label=f"{x['name']} - ref Nernst-Einstein")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"experimental $\kappa$ [$\mu$S/cm]")
    ax.set_ylabel(r"simulated $\kappa$ [$\mu$S/cm]")
    ax.set_title(f"Ionic conductivity parity (PAINN NPT, 1 M, 298 K; fit {MAIN})")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "conductivity_parity.png")
    fig.savefig(out, dpi=140); plt.close(fig); return out


def write_report(rows):
    L = []
    L.append("# Onsager ionic conductivity of PAINN electrolyte trajectories\n")
    L.append("Computed with **mdcraft** `analysis.transport.Onsager` "
             "(Fong-Self-McCloskey-Persson framework). Final fit window "
             f"**{MAIN}**.\n")
    L.append("Scripts: `md_craft.py` (read traj + MSDs + COM correction), "
             "`refit_window.py` (refit window + plots + this report). "
             "Date: 2026-06-19.\n")
    L.append("## Systems\n")
    L.append("Two ASE `.traj` NPT runs (PAINN MLFF), 1 M, 298 K, 20 ns, "
             "100 fs/frame, analysed every 10th frame (1 ps spacing, 20 000 "
             "frames). Each box: **17 ion pairs + 125 DME**.\n")
    L.append("## Method (recap)\n")
    L.append("- Single charge-carrier site per ion: Na atom (cation); central "
             "heavy atom S/P (anion). Long-time slope = molecular-COM slope, so "
             "L_ij/kappa are unaffected.\n")
    L.append("- Unwrapped with the per-frame fluctuating NPT box.\n")
    L.append("- **System COM removed** (barycentric frame). The Langevin "
             "thermostat (gamma=1 THz) drives a whole-system COM random walk "
             "(D_com=kBT/M/gamma ~0.017 A^2/ps, ~45-63 A over 20 ns). It cancels "
             "in the electroneutral charge contraction so **kappa is "
             "COM-invariant** (verified: lab frame = COM-removed in the original "
             "run), but it dominates D_i and t+, so those are reported "
             "COM-removed.\n")
    L.append(f"- L_ij and D_i from a **linear** fit over **{MAIN}** (final); "
             "1-10 ns shown for comparison.\n")
    L.append("## Final ionic conductivity (uS/cm)\n")
    L.append("| system | **mdcraft Onsager (0.5-15 ns)** | mdcraft (1-10 ns) | "
             "ref-code Onsager | ref Nernst-Einstein | **experiment** |")
    L.append("|---|---|---|---|---|---|")
    for x in rows:
        L.append(f"| {x['name']} | **{x['mine']:.0f}** | {x['alt']:.0f} | "
                 f"{x['ref_ons']:.0f} | {x['ref_ne']:.0f} | {x['exp']:.0f} |")
    L.append("\n## Transport coefficients (COM-removed, 0.5-15 ns)\n")
    L.append("| system | D(Na+) cm^2/s | D(anion) cm^2/s | t+ | t- | "
             "L++ | L+- | L-- (mol/kJ/A/ps) | self-MSD slope Na / anion |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for x in rows:
        Lij = x["L"]
        L.append(f"| {x['name']} | {x['D'][0]:.2e} | {x['D'][1]:.2e} | "
                 f"{x['t'][0]:.2f} | {x['t'][1]:.2f} | {Lij[0,0]:.2e} | "
                 f"{Lij[0,1]:.2e} | {Lij[1,1]:.2e} | "
                 f"{x['slope'][0]:.2f} / {x['slope'][1]:.2f} |")
    L.append("\n## Effect of the wider 0.5-15 ns window vs 1-10 ns\n")
    for x in rows:
        dpct = 100 * (x["mine"] - x["alt"]) / x["alt"]
        L.append(f"- **{x['name']}**: {x['alt']:.0f} -> {x['mine']:.0f} uS/cm "
                 f"({dpct:+.0f}%).")
    L.append("\nObserved: `naotf_dme` drops ~18% (4701 -> 3875 uS/cm) while "
             "`napf6_dme` is essentially unchanged (14066 -> 14083). The wider "
             "window pulls in the early sub-diffusive shoulder (0.5-1 ns) and "
             "the higher-noise >10 ns tail; naotf has small diagonal terms that "
             "wander in that tail, lowering the linear slope, whereas napf6 is "
             "dominated by the large, steadily-rising PF6-PF6 term whose slope "
             "is similar in both windows. The persistent ~1.7x gap between the "
             "mdcraft and reference Onsager values for napf6 is therefore **not "
             "a fit-window artifact** -- it points to a methodological/averaging "
             "difference and the genuinely super-diffusive PF6 term (anion "
             "slope > 1) that no single linear window fully resolves. The "
             "0.5-15 ns window also deliberately includes the higher-noise "
             ">13 ns tail (fewer independent time origins).\n")
    L.append("## Figures\n")
    L.append("- `conductivity_parity.png` - simulated vs experimental kappa "
             "(o this work 0.5-15 ns, s ref Onsager, ^ ref NE).")
    for x in rows:
        L.append(f"- `cross_displacement_{x['name']}.png` - collective "
                 "<DR_i.DR_j> vs t (cation-cation, anion-anion, cation-anion "
                 f"cross) with the {MAIN} linear fit; convergence diagnostic.")
        L.append(f"- `msd_self_{x['name']}.png` - self-MSD (log-log) with the "
                 f"{MAIN} window and a slope-1 guide.")
    L.append("\n## Notes / caveats\n")
    L.append("- Single block (n_blocks=1); for error bars, block-average or use "
             "multiple seeds.\n")
    L.append("- napf6_dme anion self-MSD slope > 1 (super-diffusive): its kappa "
             "remains the least converged; a longer trajectory would tighten it.\n")
    with open(os.path.join(OUTDIR, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ref = load_ref()
    rows = []
    print(f"{'system':12s} {'0.5-15ns':>9s} {'1-10ns':>8s} {'refOns':>8s} "
          f"{'refNE':>8s} {'exp':>8s}   t+    D(Na+)    D(an)   slope Na/an")
    for name, meta in SYS.items():
        npz = np.load(os.path.join(OUTDIR, f"onsager_{name}.npz"), allow_pickle=True)
        fits = {w: fit(npz, *WINDOWS[w]) for w in WINDOWS}
        m = fits[MAIN]
        plot_cross(name, npz, m, meta["anion"])
        plot_self(name, npz, m, meta["anion"], m["kappa"])
        d = ref[meta["ref_key"]]
        row = dict(name=name, color=meta["color"], mine=m["kappa"],
                   alt=fits["1-10ns"]["kappa"], L=m["L"], D=m["D"], t=m["t"],
                   slope=m["slope"], ref_ons=float(d["sim_conductivity_onsager_uS_cm"]),
                   ref_ne=float(d["sim_conductivity_NE_uS_cm"]),
                   exp=float(d["exp_conductivity_uS_cm"]))
        rows.append(row)
        print(f"{name:12s} {m['kappa']:9.0f} {fits['1-10ns']['kappa']:8.0f} "
              f"{row['ref_ons']:8.0f} {row['ref_ne']:8.0f} {row['exp']:8.0f}  "
              f"{m['t'][0]:5.2f} {m['D'][0]:.2e} {m['D'][1]:.2e}  "
              f"{m['slope'][0]:.2f}/{m['slope'][1]:.2f}")
    out = parity_plot(rows)
    write_report(rows)
    print(f"\nparity plot -> {out}")
    print(f"report       -> {os.path.join(OUTDIR, 'report.md')}")


if __name__ == "__main__":
    main()
