import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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

models = {
    "original_100ps": {
        "label": "Original 100ps",
        "path": (
            "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application"
            "/analysis/cell_energy_stress_mae/original_100ps_napf6"
            "/20260518_171734_cell_size_energy_mae_stress_mae"
        ),
        "ls": "-",
        "lw": 1.5,
    },
    "micro_acas_50ps": {
        "label": "Micro ACAS 50ps",
        "path": (
            "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application"
            "/analysis/cell_energy_stress_mae/micro_acas_50ps_napf6"
            "/20260518_173752_cell_size_energy_mae_stress_mae"
        ),
        "ls": "--",
        "lw": 1.5,
    },
}

out_dir = (
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application"
    "/analysis/cell_energy_stress_mae/group"
)

# ── Palette ──────────────────────────────────────────────────────────────────
abc_colors = {"a": "#E53935", "b": "#1E88E5", "c": "#43A047"}
model_palette = {"original_100ps": "#333333", "micro_acas_50ps": "#FF7043"}

# ═════════════════════════════════════════════════════════════════════════════
# Plot 1: Cell size (lattice a, b, c) — 2×2 subplots, one per system
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
axes = axes.flatten()

for i, sys_id in enumerate(systems):
    ax = axes[i]
    for mod_id, mod in models.items():
        d = np.load(f"{mod['path']}/{sys_id}/cell_size_student.npz")
        t = d["times_ns"]
        for param, color in abc_colors.items():
            ax.plot(t, d[param], color=color, ls=mod["ls"], lw=mod["lw"],
                    alpha=0.85)

    ax.set_title(system_labels[sys_id], fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (ns)", fontsize=10)
    ax.set_ylabel("Lattice parameter (Å)", fontsize=10)
    ax.grid(True, alpha=0.3)

# legend: model linestyle + param color
legend_handles = []
for mod_id, mod in models.items():
    legend_handles.append(
        mlines.Line2D([], [], color="grey", ls=mod["ls"], lw=1.8,
                      label=mod["label"])
    )
for param, color in abc_colors.items():
    legend_handles.append(
        mlines.Line2D([], [], color=color, ls="-", lw=1.8, label=param)
    )
fig.legend(handles=legend_handles, loc="lower center", ncol=5,
           fontsize=9, framealpha=0.85, bbox_to_anchor=(0.5, -0.02))

fig.suptitle(
    "Cell Lattice Parameters (a, b, c) vs Time\nOriginal 100ps vs Micro ACAS 50ps",
    fontsize=13, fontweight="bold",
)
plt.tight_layout(rect=[0, 0.06, 1, 1])
out = f"{out_dir}/cell_size_combined.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# Plot 2: Stress MAE — 2×2 subplots, one per system
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
axes = axes.flatten()

for i, sys_id in enumerate(systems):
    ax = axes[i]
    for mod_id, mod in models.items():
        d = np.load(f"{mod['path']}/{sys_id}/stress_mae_student.npz")
        t = d["times_ns"]
        color = model_palette[mod_id]
        ax.plot(t, d["mae"], color=color, ls=mod["ls"], lw=mod["lw"],
                label=mod["label"], alpha=0.85)

    ax.set_title(system_labels[sys_id], fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (ns)", fontsize=10)
    ax.set_ylabel("Stress MAE (eV/Å³)", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.3)

fig.suptitle(
    "Stress MAE vs Time\nOriginal 100ps vs Micro ACAS 50ps",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
out = f"{out_dir}/stress_mae_combined.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()
