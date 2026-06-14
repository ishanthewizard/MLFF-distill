"""
Compute ionic conductivity (Onsager formalism) for every GROMACS production
trajectory under production_runs/{npt,nvt}/*, and dump a summary CSV.

For each system directory we:
  - parse the .top [molecules] section + per-molecule .itp [atoms] sections
    to build species_order / species_mass / species_number / species_charge
  - parse the .mdp for the reference temperature (T_K)
  - load the .tpr/.xtc trajectory with MDAnalysis, unwrap PBC (unwrap +
    NoJump) to get continuous Cartesian positions in Angstrom
  - average the box volume over the analyzed frames
  - call onsager_calc and record conductivity_onsager / conductivity_NE

Dynamic viscosity does not affect the conductivity outputs (it only enters
the finite-size self-diffusivity correction Dself_inf), so a placeholder
value is used.
"""
import sys
sys.path.insert(0, '/home/yuejian/project/byteff2')

import os
import re
import csv
import glob

import numpy as np
import MDAnalysis as mda
from MDAnalysis.transformations import unwrap
from MDAnalysis.transformations.nojump import NoJump

from byteff2.md_utils.onsager_conductivity import onsager_calc

PROD_DIR = '/home/yuejian/project/byteff2/yuejian/bytemol/production_runs'
OUT_DIR = '/home/yuejian/project/byteff2/draft/conductivity/results'

VISCOSITY_CP = 1.0  # placeholder; only affects Dself_inf, not conductivity

ION_MAP = {
    'lipf6': ('Li', 'PF6'),
    'napf6': ('Na', 'PF6'),
    'naotf': ('Na', 'OTf'),
}
SOLVENT_MAP = {'dme': 'DME', 'diglyme': 'DEGDME', 'tegdme': 'TEGDME', 'pc': 'PC'}

DIRNAME_RE = re.compile(r'^([a-z0-9]+)_([a-z]+)_([\d.]+)M_(\d+)K$')


def parse_dirname(name):
    m = DIRNAME_RE.match(name)
    if not m:
        return None
    salt, solvent, conc, temp_label = m.groups()
    cation, anion = ION_MAP[salt]
    return {
        'cation': cation,
        'anion': anion,
        'solvent': SOLVENT_MAP[solvent],
        'concentration': float(conc),
        'temp_label_K': float(temp_label),
    }


def parse_top_molecules(top_path):
    """Return list of (molname, nmol) from the [ molecules ] section."""
    with open(top_path) as f:
        lines = f.readlines()
    molecules = []
    in_section = False
    for line in lines:
        line = line.split(';')[0].strip()
        if not line:
            continue
        if line.startswith('['):
            in_section = line.lower().replace(' ', '') == '[molecules]'
            continue
        if in_section:
            molname, nmol = line.split()
            molecules.append((molname, int(nmol)))
    return molecules


def parse_itp(itp_path):
    """Return (moleculetype_name, masses, charges) from an .itp file."""
    with open(itp_path) as f:
        lines = f.readlines()
    section = None
    name = None
    masses, charges = [], []
    for raw in lines:
        line = raw.split(';')[0].strip()
        if not line:
            continue
        if line.startswith('['):
            section = line.lower().replace(' ', '')
            continue
        if section == '[moleculetype]' and name is None:
            name = line.split()[0]
        elif section == '[atoms]':
            parts = line.split()
            charges.append(float(parts[6]))
            masses.append(float(parts[7]))
    return name, masses, charges


def build_species_dicts(sys_dir, molecules):
    """Map [molecules] entries to per-species mass/charge lists using the
    moleculetype names declared in the .itp files in sys_dir."""
    name_to_itp = {}
    for itp_path in glob.glob(os.path.join(sys_dir, '*.itp')):
        if 'forcefield' in os.path.basename(itp_path):
            continue
        molname, masses, charges = parse_itp(itp_path)
        if molname is not None:
            name_to_itp[molname] = (masses, charges)

    species_order, species_mass, species_number, species_charge = [], {}, {}, {}
    for molname, nmol in molecules:
        masses, charges = name_to_itp[molname]
        species_order.append(molname)
        species_mass[molname] = masses
        species_number[molname] = nmol
        species_charge[molname] = int(round(sum(charges)))
    return species_order, species_mass, species_number, species_charge


def parse_ref_temperature(mdp_path):
    with open(mdp_path) as f:
        for line in f:
            line = line.split(';')[0].strip()
            if line.lower().startswith('ref_t') or line.lower().startswith('ref-t'):
                return float(line.split('=')[1].split()[0])
    return None


def load_unwrapped_trajectory(tpr_path, xtc_path):
    """Returns (positions [nframes, natoms, 3] in Angstrom, mean box volume in Angstrom^3)."""
    u = mda.Universe(tpr_path, xtc_path)
    u.trajectory.add_transformations(unwrap(u.atoms), NoJump())

    nframes = len(u.trajectory)
    natoms = len(u.atoms)
    positions = np.empty((nframes, natoms, 3), dtype=np.float64)
    volumes = np.empty(nframes, dtype=np.float64)
    for i, ts in enumerate(u.trajectory):
        positions[i] = ts.positions
        dims = ts.dimensions
        volumes[i] = dims[0] * dims[1] * dims[2]
    return positions, volumes.mean()


def process_system(sys_dir, ensemble):
    name = os.path.basename(sys_dir.rstrip('/'))
    info = parse_dirname(name)
    if info is None:
        return None

    top_path = glob.glob(os.path.join(sys_dir, '*.top'))[0]
    mdp_path = os.path.join(sys_dir, f'{ensemble}.mdp')
    tpr_path = os.path.join(sys_dir, f'{ensemble}.tpr')
    xtc_path = os.path.join(sys_dir, f'{ensemble}.xtc')

    molecules = parse_top_molecules(top_path)
    species_order, species_mass, species_number, species_charge = build_species_dicts(sys_dir, molecules)
    T_K = parse_ref_temperature(mdp_path) or info['temp_label_K']

    print(f'[{ensemble}] {name}: loading trajectory ...')
    positions, volume_angstrom3 = load_unwrapped_trajectory(tpr_path, xtc_path)
    print(f'[{ensemble}] {name}: {positions.shape[0]} frames, '
          f'<V> = {volume_angstrom3:.2f} A^3, T = {T_K} K')

    results = onsager_calc(
        species_order=species_order,
        species_mass=species_mass,
        species_number=species_number,
        species_charge=species_charge,
        volume_angstrom3=volume_angstrom3,
        viscosity_cP=VISCOSITY_CP,
        T_K=T_K,
        positions=positions,
    )

    return {
        'cation': info['cation'],
        'anion': info['anion'],
        'solvent': info['solvent'],
        'concentration': info['concentration'],
        'temperature_K': T_K,
        'ensemble': ensemble,
        'conductivity_onsager_mS_cm': results['conductivity_onsager'],
        'conductivity_NE_mS_cm': results['conductivity_NE'],
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'conductivity_results.csv')

    fieldnames = ['cation', 'anion', 'solvent', 'concentration', 'temperature_K',
                   'ensemble', 'conductivity_onsager_mS_cm', 'conductivity_NE_mS_cm']

    rows = []
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            rows = list(csv.DictReader(f))
        done = {(r['cation'], r['anion'], r['solvent'], r['concentration'], r['ensemble']) for r in rows}
        print(f'Resuming: {len(rows)} systems already done')

    for ensemble in ('nvt', 'npt'):
        ens_dir = os.path.join(PROD_DIR, ensemble)
        for sys_dir in sorted(glob.glob(os.path.join(ens_dir, '*/'))):
            name = os.path.basename(sys_dir.rstrip('/'))
            info = parse_dirname(name)
            if info is None:
                continue
            key = (info['cation'], info['anion'], info['solvent'], str(info['concentration']), ensemble)
            if key in done:
                print(f'[{ensemble}] {name}: already done, skipping')
                continue
            try:
                row = process_system(sys_dir, ensemble)
            except Exception as e:
                print(f'[{ensemble}] {name}: FAILED ({e})')
                continue
            if row is not None:
                rows.append(row)
                # write incrementally so partial progress is preserved
                with open(out_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

    print(f'Wrote {len(rows)} rows to {out_path}')


if __name__ == '__main__':
    main()
