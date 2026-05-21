#!/usr/bin/env python3
"""Generate one .sh per system for both roots, then write a parallel launcher."""

import os
from pathlib import Path

PYTHON = "/global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2_new/bin/python"
PROJ = "/global/homes/y/yuejian/project/MLFF-distill"
MSD_PY = "APPLICATIONS/electrolytes/observable_scripts/mean_square_displacement/msds_calculation_batch.py"
TAU_MAX = 15000
DT = 0.1

HERE = Path(__file__).parent

ROOTS = {
    "micro": {
        "base": f"{PROJ}/m5024/distillation_project/results/diffusivity_main_results_ckpt/other_fix_run/micro_trained_on_all_concentration_50ps_window",
        "prefix_0_1M": "20ns_solvent_0_1M",            # no temp subdir
        "prefix_0_5M": "20ns_solvent_solute_0.5M",
        "prefix_1M":   "20ns_solute_solvent_1M/298K",
    },
    "original": {
        "base": f"{PROJ}/m5024/distillation_project/results/diffusivity_main_results_ckpt/other_fix_run/original_trained_on_1M_only_100ps_window",
        "prefix_0_1M": "20ns_solvent_0_1M/298K",       # has extra 298K subdir
        "prefix_0_5M": "20ns_solvent_solute_0.5M",
        "prefix_1M":   "20ns_solute_solvent_1M/298K",
    },
}

# (dirname, label, cation, anion, solvent, conc, temp)
SYSTEMS_0_1M = [
    ("md_omol_naotf_dme_s1p1_omol",                                         "NaOTf-DME 0.1M 298K",      "Na", "OTf", "DME",     "0_1M", "298K"),
    ("md_omol_naotf_diglyme_1m_s1p1",                                       "NaOTf-Diglyme 0.1M 298K",  "Na", "OTf", "Diglyme", "0_1M", "298K"),
    ("md_omol_naotf_pc_1m_s1p1",                                            "NaOTf-PC 0.1M 298K",       "Na", "OTf", "PC",      "0_1M", "298K"),
    ("md_omol_naotf_tgdme_1m_s1p1",                                         "NaOTf-TGDME 0.1M 298K",    "Na", "OTf", "TGDME",   "0_1M", "298K"),
    ("md_omol_napf6_diglyme_pfactor_0.1_1fs",                               "NaPF6-Diglyme 0.1M 298K",  "Na", "PF6", "Diglyme", "0_1M", "298K"),
    ("md_omol_napf6_dme_re1",                                               "NaPF6-DME 0.1M 298K",      "Na", "PF6", "DME",     "0_1M", "298K"),
    ("md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1",      "NaPF6-PC 0.1M 298K",       "Na", "PF6", "PC",      "0_1M", "298K"),
]

# (temp_dir, dirname, label, cation, anion, solvent, conc, temp)
SYSTEMS_0_5M = [
    ("273_2K", "md_omol_napf6_dme_re1",                "NaPF6-DME 0.5M 273K",   "Na", "PF6", "DME", "0_5M", "273K"),
    ("273_2K", "md_omol_lipf6_pfactor_0.1_1fs_mask_t", "LiPF6-DME 0.5M 273K",   "Li", "PF6", "DME", "0_5M", "273K"),
    ("298_2K", "md_omol_napf6_dme_re1",                "NaPF6-DME 0.5M 298K",   "Na", "PF6", "DME", "0_5M", "298K"),
    ("298_2K", "md_omol_lipf6_pfactor_0.1_1fs_mask_t", "LiPF6-DME 0.5M 298K",   "Li", "PF6", "DME", "0_5M", "298K"),
    ("323_2K", "md_omol_napf6_dme_re1",                "NaPF6-DME 0.5M 323K",   "Na", "PF6", "DME", "0_5M", "323K"),
    ("323_2K", "md_omol_lipf6_pfactor_0.1_1fs_mask_t", "LiPF6-DME 0.5M 323K",   "Li", "PF6", "DME", "0_5M", "323K"),
]

SYSTEMS_1M = [
    ("md_omol_napf6_diglyme_pfactor_0.1_1fs",                              "NaPF6-Diglyme 1M 298K",  "Na", "PF6", "Diglyme", "1M", "298K"),
    ("md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1",     "NaPF6-PC 1M 298K",       "Na", "PF6", "PC",      "1M", "298K"),
    ("naotf_diglyme",                                                      "NaOTf-Diglyme 1M 298K",  "Na", "OTf", "Diglyme", "1M", "298K"),
    ("naotf_dme",                                                          "NaOTf-DME 1M 298K",      "Na", "OTf", "DME",     "1M", "298K"),
    ("napf6_dme",                                                          "NaPF6-DME 1M 298K",      "Na", "PF6", "DME",     "1M", "298K"),
]


def script_name(label: str) -> str:
    return "msd_" + label.replace(" ", "_").replace(".", "p") + ".sh"


def make_script(traj_path: str, label: str, cat: str, anion: str, solvent: str,
                conc: str, temp: str, out_dir: str) -> str:
    target_json = (
        f'[["{traj_path}", "{label}", "{cat}", "{anion}", "{solvent}", "{conc}", "{temp}"]]'
    )
    return f"""#!/bin/bash
# System: {label}
set -euo pipefail
cd {PROJ}
export MPLBACKEND=Agg
export MSD_TARGETS_JSON='{target_json}'
{PYTHON} {MSD_PY} \\
    --out-dir "{out_dir}" \\
    --tau-max-fit-ps {TAU_MAX} \\
    --known-dt-ps {DT}
"""


all_scripts = []   # list of absolute paths to generated scripts

for root_name, cfg in ROOTS.items():
    base = cfg["base"]
    out_dir = f"{base}/diffusivity_results"
    script_dir = HERE / root_name

    # 0.1M
    for dirname, label, cat, anion, solvent, conc, temp in SYSTEMS_0_1M:
        traj = f"{base}/{cfg['prefix_0_1M']}/{dirname}/{dirname}.traj"
        content = make_script(traj, label, cat, anion, solvent, conc, temp, out_dir)
        fname = script_dir / script_name(label)
        fname.write_text(content)
        fname.chmod(0o755)
        all_scripts.append(str(fname))
        print(f"  wrote {fname.name}")

    # 0.5M
    for temp_dir, dirname, label, cat, anion, solvent, conc, temp in SYSTEMS_0_5M:
        traj = f"{base}/{cfg['prefix_0_5M']}/{temp_dir}/{dirname}/{dirname}.traj"
        content = make_script(traj, label, cat, anion, solvent, conc, temp, out_dir)
        fname = script_dir / script_name(label)
        fname.write_text(content)
        fname.chmod(0o755)
        all_scripts.append(str(fname))
        print(f"  wrote {fname.name}")

    # 1M
    for dirname, label, cat, anion, solvent, conc, temp in SYSTEMS_1M:
        traj = f"{base}/{cfg['prefix_1M']}/{dirname}/{dirname}.traj"
        content = make_script(traj, label, cat, anion, solvent, conc, temp, out_dir)
        fname = script_dir / script_name(label)
        fname.write_text(content)
        fname.chmod(0o755)
        all_scripts.append(str(fname))
        print(f"  wrote {fname.name}")

# Write parallel launcher
launcher = HERE / "launch_all_parallel.sh"
lines = ["#!/bin/bash", "# Launch all per-system MSD scripts in parallel", "set -euo pipefail", ""]
for s in all_scripts:
    log = s.replace(".sh", ".log")
    lines.append(f'bash "{s}" > "{log}" 2>&1 &')
lines += ["", "echo \"Launched ${#} background jobs, waiting...\"", "wait", "echo \"All done.\""]
launcher.write_text("\n".join(lines) + "\n")
launcher.chmod(0o755)
print(f"\nLauncher: {launcher}")
print(f"Total scripts: {len(all_scripts)}")
