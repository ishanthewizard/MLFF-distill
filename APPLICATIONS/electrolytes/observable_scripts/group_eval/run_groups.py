#!/usr/bin/env python3
"""Drive parity_plot.py per-group from a groups config (e.g. groups_example.py).

For each group, makes one parity plot per requested property using only
that group's parity csv (groups are plotted separately, not overlaid).

Usage:
    python run_groups.py groups_example
"""

import argparse
import importlib
from pathlib import Path

from parity_plot import plot_group_parity
from properties import PROPERTIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Module name of the groups config, e.g. groups_example")
    parser.add_argument("--out-dir", default=None,
                         help="Directory for output PNGs (default: alongside each group's csv)")
    args = parser.parse_args()

    cfg = importlib.import_module(args.config)

    for group_name, group in cfg.GROUPS.items():
        csv_path = Path(group["parity_csv"])
        for prop_name in group["properties"]:
            prop = PROPERTIES[prop_name]
            out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
            output = out_dir / f"parity_{prop_name}_{group_name}.png"
            plot_group_parity(
                [csv_path],
                exp_col=prop["exp_col"],
                sim_col=prop["sim_col"],
                labels=[group["label"]],
                output=output,
                xlabel=prop.get("xlabel"),
                ylabel=prop.get("ylabel"),
                title=prop.get("title"),
                log_scale=prop.get("log_scale", True),
                annotate_meta=True,
            )


if __name__ == "__main__":
    main()
