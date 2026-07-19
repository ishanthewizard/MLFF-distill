"""Smoke test: 1 system / 1 replica of the 4 ns NPT conductivity config.

Validates the full eval.py conductivity path (byteff2 Onsager+NE and mdcraft
collective) with the [2,6] ns window + 5 ps lag dt before submitting the 141-traj
SLURM job. Runs on a single trajectory (~a few min).
"""
from config_painn_fp32_indist_4ns_npt_conductivity import (  # noqa: F401,F403
    SYSTEMS as _ALL, ANALYSES,
)

# pick one representative system: naotf_dme 1M @ 298 K (tests OTf anion grouping)
_pick = next(s for s in _ALL if s["name"].startswith("naotf_dme__npt_1M"))
_pick = dict(_pick)
_r0 = list(_pick["traj_paths"].items())[0]
_pick["traj_paths"] = {_r0[0]: _r0[1]}
SYSTEMS = [_pick]

OUTPUT_DIR = "/pscratch/sd/y/yuejian/smoke_4ns_cond"
WORKERS = 1
