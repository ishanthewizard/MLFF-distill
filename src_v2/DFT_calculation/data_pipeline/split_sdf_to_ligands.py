import os
from rdkit import Chem
from rdkit.Chem import SDWriter

# Input SDF file
input_sdf = '/home/yuejian/project/MLFF-distill/strain/ligboundconf_2_crest_global_min.sdf'
# Output base directory
output_base = '/home/yuejian/project/MLFF-distill/data/ligandboundconf3.0/crest_global_min'

# Read molecules from SDF
supplier = Chem.SDMolSupplier(input_sdf, removeHs=False)

for mol in supplier:
    if mol is None:
        continue
    # Try to get ligand_id from title or a property
    ligand_id = mol.GetProp('_Name') if mol.HasProp('_Name') else None

    if not ligand_id:
        # Fallback: try a property, or use index
        ligand_id = mol.GetProp('ligand_id') if mol.HasProp('ligand_id') else None
    if not ligand_id:
        # If still not found, skip or use a generic name
        continue
    # Prepare output directory and file
    out_dir = os.path.join(output_base, ligand_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{ligand_id}.sdf')
    # Write single molecule to its SDF
    with SDWriter(out_path) as writer:
        writer.write(mol)
print('Splitting complete.') 