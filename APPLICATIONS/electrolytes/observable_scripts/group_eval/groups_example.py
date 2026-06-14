"""Example group config: one parity analysis per "group" of simulations.

Each entry in GROUPS maps a group name to:
  - ``label``: legend label used in the parity plot for this group
  - ``parity_csv``: the "*_parity.csv" produced by build_parity_csv.py for
    this group's simulations
  - ``properties``: which entries from properties.py to plot for this group

A "group" here is a set of simulations sharing the same source/ensemble
that were combined into one parity csv by build_parity_csv.py. Parity is
plotted *per group* (one PNG per group per property) -- groups are not
mixed together in a single plot, unlike the multi-csv overlay use case in
parity_plot.py's CLI.
"""

OUT_DIR = "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/analysis/together"

GROUPS = {
    "group1": {
        "label": "OPLS npt",
        "parity_csv": f"{OUT_DIR}/conductivity_parity_opls_npt.csv",
        "properties": ["conductivity", "conductivity_NE", "density"],
    },
    "group2": {
        "label": "PAINN tf32",
        "parity_csv": f"{OUT_DIR}/conductivity_parity_painn.csv",
        "properties": ["conductivity", "conductivity_NE", "density"],
    },
}
