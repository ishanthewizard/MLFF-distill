import io
import re
from typing import TextIO, Union
import numpy as np
from ase import Atoms
from ase.io.lammpsdata import read_lammps_data  # keep your custom read if shadowing
from ase.units import create_units

def write_lammps_data_with_angles_dihedrals(
    fd: Union[str, TextIO],
    atoms: Atoms,
):
    """
    Write a LAMMPS data file preserving original type IDs from atoms.arrays['type'],
    including Masses, Bonds, Angles, Dihedrals if present.

    Assumes atoms.arrays may contain:
        'type'       : original LAMMPS type (int)
        'bonds'      : string encoding (i_idx(type), comma-separated) or '_'
        'angles'     : center atom array entries "i-j(type)"
        'dihedrals'  : first atom array entries "i-j-k(type)"
    """

    # ---- 1. Gather core data ----
    n_atoms = len(atoms)
    types_array = atoms.arrays.get('type')
    if types_array is None:
        raise ValueError("atoms.arrays['type'] not found; cannot preserve original type mapping.")
    types_array = np.asarray(types_array, dtype=int)
    max_type = int(types_array.max())
    # If you want to check for missing intermediate types:
    present_types = sorted(set(types_array.tolist()))

    # masses: use per-atom masses and pick representative for each type
    per_atom_masses = atoms.get_masses()
    type_to_mass = {}
    for t in present_types:
        idx = int(np.where(types_array == t)[0][0])
        type_to_mass[t] = per_atom_masses[idx]

    # (Optional) preserve original bounds if you stored them:
    # bounds = atoms.info.get('lammps_box_bounds')  # (xlo,xhi,ylo,yhi,zlo,zhi)
    # For now derive from cell and set origin at 0.
    cell = atoms.get_cell()
    xhi, yhi, zhi = cell[0,0], cell[1,1], cell[2,2]
    xlo = ylo = zlo = 0.0

    # ---- 2. Parse bonds / angles / dihedrals from arrays (if any) ----
    bond_entries = []
    bond_types = set()
    if 'bonds' in atoms.arrays:
        for a_idx, entry in enumerate(atoms.arrays['bonds']):
            if entry == '_' or not entry.strip():
                continue
            for token in entry.split(','):
                # token like "j(type)"
                m = re.match(r'^(\d+)\((\d+)\)$', token.strip())
                if not m:
                    raise ValueError(f"Unrecognized bond token '{token}'")
                j_str, t_str = m.groups()
                at1 = a_idx + 1          # current atom
                at2 = int(j_str) + 1     # stored as 0-based
                btype = int(t_str)
                # To avoid double counting, only keep if at1 < at2
                if at1 < at2:
                    bond_entries.append((btype, at1, at2))
                    bond_types.add(btype)
    n_bonds = len(bond_entries)
    n_bond_types = max(bond_types) if bond_types else 0

    angle_entries = []
    angle_types = set()
    if 'angles' in atoms.arrays:
        for center_idx, entry in enumerate(atoms.arrays['angles']):
            if entry == '_' or not entry.strip():
                continue
            for token in entry.split(','):
                # "i-j(type)"
                m = re.match(r'^(\d+)-(\d+)\((\d+)\)$', token.strip())
                if not m:
                    raise ValueError(f"Unrecognized angle token '{token}'")
                i_str, k_str, t_str = m.groups()
                a1 = int(i_str) + 1
                a2 = center_idx + 1
                a3 = int(k_str) + 1
                atype = int(t_str)
                # To avoid duplicates (each central atom list creates unique)
                angle_entries.append((atype, a1, a2, a3))
                angle_types.add(atype)
    n_angles = len(angle_entries)
    n_angle_types = max(angle_types) if angle_types else 0

    dihedral_entries = []
    dihedral_types = set()
    if 'dihedrals' in atoms.arrays:
        for first_idx, entry in enumerate(atoms.arrays['dihedrals']):
            if entry == '_' or not entry.strip():
                continue
            for token in entry.split(','):
                # "i-j-k(type)"
                m = re.match(r'^(\d+)-(\d+)-(\d+)\((\d+)\)$', token.strip())
                if not m:
                    raise ValueError(f"Unrecognized dihedral token '{token}'")
                i_str, j_str, k_str, t_str = m.groups()
                a1 = first_idx + 1
                a2 = int(i_str) + 1
                a3 = int(j_str) + 1
                a4 = int(k_str) + 1
                dtype = int(t_str)
                dihedral_entries.append((dtype, a1, a2, a3, a4))
                dihedral_types.add(dtype)
    n_dihedrals = len(dihedral_entries)
    n_dihedral_types = max(dihedral_types) if dihedral_types else 0

    # impropers not handled here; add analogously if needed
    n_impropers = 0
    n_improper_types = 0

    # ---- 3. Build header in canonical order ----
    header_lines = []
    header_lines.append("LAMMPS data file (written by custom ASE wrapper)")
    header_lines.append(f"{n_atoms} atoms")
    if n_bonds:
        header_lines.append(f"{n_bonds} bonds")
    if n_angles:
        header_lines.append(f"{n_angles} angles")
    if n_dihedrals:
        header_lines.append(f"{n_dihedrals} dihedrals")
    if n_impropers:
        header_lines.append(f"{n_impropers} impropers")

    header_lines.append(f"{max_type} atom types")
    if n_bonds:
        header_lines.append(f"{n_bond_types} bond types")
    if n_angles:
        header_lines.append(f"{n_angle_types} angle types")
    if n_dihedrals:
        header_lines.append(f"{n_dihedral_types} dihedral types")
    if n_impropers:
        header_lines.append(f"{n_improper_types} improper types")

    header_lines.append(f"{xlo:.6f} {xhi:.6f}  xlo xhi")
    header_lines.append(f"{ylo:.6f} {yhi:.6f}  ylo yhi")
    header_lines.append(f"{zlo:.6f} {zhi:.6f}  zlo zhi")
    header_lines.append("")  # blank line after header

    # ---- 4. Masses section ----
    # Ensure every type ID 1..max_type appears (even if absent) using fallback mass
    # fallback: use first present mass or 1.0
    fallback_mass = next(iter(type_to_mass.values())) if type_to_mass else 1.0
    masses_lines = ["Masses", ""]
    for t in range(1, max_type + 1):
        mass = type_to_mass.get(t, fallback_mass)
        masses_lines.append(f"{t:5d} {mass: .7f}")
    masses_lines.append("")  # blank line

    # ---- 5. Atoms section (# full) ----
    # Format: id mol-id type q x y z   (we assume mol-id array may or may not exist)
    # Your reader may have stored charges in 'initial_charges', else default 0.0
    if atoms.has('mol-id'):
        mol_ids = atoms.get_array('mol-id').astype(int)
    else:
        mol_ids = np.zeros(n_atoms, dtype=int)
    if 'initial_charges' in atoms.arrays:
        charges = atoms.arrays['initial_charges']
    elif 'mmcharges' in atoms.arrays:
        charges = atoms.arrays['mmcharges']
    else:
        charges = np.zeros(n_atoms)

    positions = atoms.get_positions()  # already in Å; LAMMPS "metal" expects Å
    atoms_lines = ["Atoms # full", ""]
    for i in range(n_atoms):
        # id, mol-id, type, q, x, y, z
        atoms_lines.append(
            f"{i+1:6d} {mol_ids[i]:6d} {types_array[i]:4d} {charges[i]: .6f} "
            f"{positions[i,0]: .6f} {positions[i,1]: .6f} {positions[i,2]: .6f}"
        )
    atoms_lines.append("")

    # ---- 6. Bonds / Angles / Dihedrals sections ----
    bonds_lines = []
    if n_bonds:
        bonds_lines = ["Bonds", ""]
        for idx, (btype, a1, a2) in enumerate(bond_entries, start=1):
            bonds_lines.append(f"{idx:6d} {btype:4d} {a1:6d} {a2:6d}")
        bonds_lines.append("")

    angles_lines = []
    if n_angles:
        angles_lines = ["Angles", ""]
        for idx, (atype, a1, a2, a3) in enumerate(angle_entries, start=1):
            angles_lines.append(f"{idx:6d} {atype:4d} {a1:6d} {a2:6d} {a3:6d}")
        angles_lines.append("")

    dihedrals_lines = []
    if n_dihedrals:
        dihedrals_lines = ["Dihedrals", ""]
        for idx, (dtype, a1, a2, a3, a4) in enumerate(dihedral_entries, start=1):
            dihedrals_lines.append(
                f"{idx:6d} {dtype:4d} {a1:6d} {a2:6d} {a3:6d} {a4:6d}"
            )
        dihedrals_lines.append("")

    # ---- 7. Assemble and write ----
    text_parts = (
        header_lines
        + masses_lines
        + atoms_lines
        + bonds_lines
        + angles_lines
        + dihedrals_lines
    )
    final_text = "\n".join(text_parts) + "\n"

    must_close = False
    if isinstance(fd, str):
        fd_obj = open(fd, "w")
        must_close = True
    else:
        fd_obj = fd

    fd_obj.write(final_text)
    fd_obj.flush()
    if must_close:
        fd_obj.close()


if __name__ == "__main__":
    atoms = read_lammps_data("APPLICATIONS/geometry.dat")
    write_lammps_data_with_angles_dihedrals("APPLICATIONS/output.dat", atoms)
