"""Per-property defaults for ``parity_plot.py``.

Each entry maps a ``--property`` name to the column names / plot labels
used for that property's "*_parity.csv" files. Add a new entry here when
a new observable (density, viscosity, ...) gets a parity csv, instead of
hardcoding column names in ``parity_plot.py``.
"""

PROPERTIES = {
    "conductivity": {
        "exp_col": "exp_conductivity_uS_cm",
        "sim_col": "sim_conductivity_onsager_uS_cm",
        "xlabel": "Experimental conductivity (uS/cm)",
        "ylabel": "MD Onsager conductivity (uS/cm)",
        "title": "Ionic conductivity: MD vs experiment",
        "log_scale": True,
    },
    "conductivity_NE": {
        "exp_col": "exp_conductivity_uS_cm",
        "sim_col": "sim_conductivity_NE_uS_cm",
        "xlabel": "Experimental conductivity (uS/cm)",
        "ylabel": "MD Nernst-Einstein conductivity (uS/cm)",
        "title": "Ionic conductivity (NE): MD vs experiment",
        "log_scale": True,
    },
    "conductivity_linear": {
        "exp_col": "exp_conductivity_uS_cm",
        "sim_col": "sim_conductivity_onsager_uS_cm",
        "xlabel": "Experimental conductivity (uS/cm)",
        "ylabel": "MD Onsager conductivity (uS/cm)",
        "title": "Ionic conductivity: MD vs experiment",
        "log_scale": False,
    },
    "conductivity_NE_linear": {
        "exp_col": "exp_conductivity_uS_cm",
        "sim_col": "sim_conductivity_NE_uS_cm",
        "xlabel": "Experimental conductivity (uS/cm)",
        "ylabel": "MD Nernst-Einstein conductivity (uS/cm)",
        "title": "Ionic conductivity (NE): MD vs experiment",
        "log_scale": False,
    },
    "density": {
        "exp_col": "exp_density_g_mL",
        "sim_col": "sim_density_g_mL",
        "xlabel": "Experimental density (g/mL)",
        "ylabel": "MD density (g/mL)",
        "title": "Density: MD vs experiment",
        "log_scale": False,
    },
    "diffusivity_cation": {
        "exp_col": "exp_D_cation_1e-10_m2s",
        "sim_col": "D_cat_corrected_1e-10_m2s",
        "xlabel": r"Experimental cation D (x 10^-10 m^2/s)",
        "ylabel": r"MD cation D, finite-size corrected (x 10^-10 m^2/s)",
        "title": "Cation diffusivity: MD vs experiment",
        "log_scale": True,
    },
    "diffusivity_anion": {
        "exp_col": "exp_D_anion_1e-10_m2s",
        "sim_col": "D_ani_corrected_1e-10_m2s",
        "xlabel": r"Experimental anion D (x 10^-10 m^2/s)",
        "ylabel": r"MD anion D, finite-size corrected (x 10^-10 m^2/s)",
        "title": "Anion diffusivity: MD vs experiment",
        "log_scale": True,
    },
    "diffusivity_solvent": {
        "exp_col": "exp_D_solvent_1e-10_m2s",
        "sim_col": "D_sol_corrected_1e-10_m2s",
        "xlabel": r"Experimental solvent D (x 10^-10 m^2/s)",
        "ylabel": r"MD solvent D, finite-size corrected (x 10^-10 m^2/s)",
        "title": "Solvent diffusivity: MD vs experiment",
        "log_scale": True,
    },
    "diffusivity_cation_uncorrected": {
        "exp_col": "exp_D_cation_1e-10_m2s",
        "sim_col": "D_cat_1e-10_m2s",
        "xlabel": r"Experimental cation D (x 10^-10 m^2/s)",
        "ylabel": r"MD cation D, uncorrected (x 10^-10 m^2/s)",
        "title": "Cation diffusivity (uncorrected): MD vs experiment",
        "log_scale": True,
    },
    "diffusivity_anion_uncorrected": {
        "exp_col": "exp_D_anion_1e-10_m2s",
        "sim_col": "D_ani_1e-10_m2s",
        "xlabel": r"Experimental anion D (x 10^-10 m^2/s)",
        "ylabel": r"MD anion D, uncorrected (x 10^-10 m^2/s)",
        "title": "Anion diffusivity (uncorrected): MD vs experiment",
        "log_scale": True,
    },
    "diffusivity_solvent_uncorrected": {
        "exp_col": "exp_D_solvent_1e-10_m2s",
        "sim_col": "D_sol_1e-10_m2s",
        "xlabel": r"Experimental solvent D (x 10^-10 m^2/s)",
        "ylabel": r"MD solvent D, uncorrected (x 10^-10 m^2/s)",
        "title": "Solvent diffusivity (uncorrected): MD vs experiment",
        "log_scale": True,
    },
}
