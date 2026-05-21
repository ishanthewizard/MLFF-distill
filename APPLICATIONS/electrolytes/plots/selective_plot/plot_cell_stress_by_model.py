import numpy as np
import matplotlib.pyplot as plt

systems = [
    "NaPF6_DME_0.1M_298K",
    "NaPF6_DME_0.5M_298K",
    "NaPF6_DME_1M_298K",
    "NaPF6_Diglyme_0.1M_298K",
]

system_labels = {
    "NaPF6_DME_0.1M_298K":    "NaPF6 DME 0.1M 298K",
    "NaPF6_DME_0.5M_298K":    "NaPF6 DME 0.5M 298K",
    "NaPF6_DME_1M_298K":      "NaPF6 DME 1M 298K",
    "NaPF6_Diglyme_0.1M_298K":"NaPF6 Diglyme 0.1M 298K",
}

system_colors = [
    "#E53935",  # red
    "#1E88E5",  # blue
    "#43A047",  # green
    "#FB8C00",  # orange
]

models = {
    "original_100ps": {
        "label": "Original 100ps",
        "path": (
            "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application"
            "/analysis/cell_energy_stress_mae/original_100ps_napf6"
            "/20260518_171734_cell_size_energy_mae_stress_mae"
        ),
    },
    "micro_acas_50ps": {
        "label": "Micro ACAS 50ps",
        "path": (
            "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application"
            "/analysis/cell_energy_stress_mae/micro_acas_50ps_napf6"
            "/20260518_173752_cell_size_energy_mae_stress_mae"
        ),
    },
}

out_dir = (
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application"
    "/analysis/cell_energy_stress_mae/group"
)

# ═════════════════════════════════════════════════════════════════════════════
# Cell size: one figure per model, 1×3 subplots (a, b, c), all 4 systems
# ═════════════════════════════════════════════════════════════════════════════
for mod_id, mod in models.items():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    for ax, param in zip(axes, ["a", "b", "c"]):
        for sys_id, color in zip(systems, system_colors):
            d = np.load(f"{mod['path']}/{sys_id}/cell_size_student.npz")
            ax.plot(d["times_ns"], d[param], color=color, lw=1.4,
                    label=system_labels[sys_id], alpha=0.85)
        ax.axhline(12, color="black", lw=1.2, ls="--", alpha=0.7,
                   label="12 Å (2×r_cut)")
        ax.set_title(f"Lattice {param}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (ns)", fontsize=11)
        ax.set_ylabel("Length (Å)", fontsize=11)
        ax.grid(True, alpha=0.3)

    # shared legend on the rightmost panel (systems + ref line)
    import matplotlib.lines as mlines
    handles, labels = axes[-1].get_legend_handles_labels()
    ref_handle = mlines.Line2D([], [], color="black", lw=1.2, ls="--",
                               alpha=0.7, label="12 Å (2×r_cut)")
    # deduplicate ref line (appears 3× from each panel but we only want one)
    handles = [h for h, l in zip(handles, labels) if l != "12 Å (2×r_cut)"]
    labels  = [l for l in labels if l != "12 Å (2×r_cut)"]
    axes[-1].legend(handles + [ref_handle], labels + ["12 Å (2×r_cut)"],
                    fontsize=8.5, framealpha=0.85, loc="best")

    fig.suptitle(
        f"Cell Lattice Parameters vs Time — {mod['label']}",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out = f"{out_dir}/cell_size_{mod_id}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# Stress MAE: one figure per model, single panel, all 4 systems
# ═════════════════════════════════════════════════════════════════════════════
for mod_id, mod in models.items():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for sys_id, color in zip(systems, system_colors):
        d = np.load(f"{mod['path']}/{sys_id}/stress_mae_student.npz")
        ax.plot(d["times_ns"], d["mae"], color=color, lw=1.4,
                label=system_labels[sys_id], alpha=0.85)

    ax.set_xlabel("Time (ns)", fontsize=12)
    ax.set_ylabel("Stress MAE (eV/Å³)", fontsize=12)
    ax.set_title(f"Stress MAE vs Time — {mod['label']}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"{out_dir}/stress_mae_{mod_id}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()
