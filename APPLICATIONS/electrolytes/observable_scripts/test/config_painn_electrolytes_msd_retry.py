"""Retry config: MSD for the 5 systems that failed in the previous full
rerun due to m5250 disk quota (napf6_pc * and napf6_tgdme npt 1.0M 298K).
Output redirected to /pscratch to avoid the quota issue.

Run with:
  python eval.py --config config_painn_electrolytes_msd_retry.py
"""
from config_painn_electrolytes_main import SYSTEMS as _ALL_SYSTEMS

_WANTED_NAMES = {
    "napf6_pc npt 0.1M 298K (npt_0_1M_298K_2ns_100fs)",
    "napf6_pc npt 1.0M 298K (npt_1M_298K_2ns_100fs)",
    "napf6_pc nvt 0.1M 298K (nvt_0_1M_298K_20ns_100fs)",
    "napf6_pc nvt 1.0M 298K (nvt_1M_298K_20ns_100fs)",
    "napf6_tgdme npt 1.0M 298K (npt_1M_298K_2ns_100fs)",
}

SYSTEMS = [s for s in _ALL_SYSTEMS if s["name"] in _WANTED_NAMES]

OUTPUT_DIR = "/pscratch/sd/y/yuejian/painn_msd_rerun"
ANALYSES = ["msd"]
WORKERS = 5
