from mace.calculators import mace_mp
from ase import build

atoms = build.molecule('H2O')
calc = mace_mp(model="medium", dispersion=False, default_dtype="float64",device='cuda', )#device='cpu')
atoms.calc = calc
breakpoint()
hessian=calc.get_hessian(atoms=atoms)
print("h:",hessian.shape)