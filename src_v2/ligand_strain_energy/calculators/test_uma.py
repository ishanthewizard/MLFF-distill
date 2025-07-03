import pytest
from rdkit import Chem
from ._uma import UMA_min


@pytest.mark.gpu
def test_UMA_min(mols: dict[str, Chem.Mol], model_path: str):
    """Test UMA minimisation with a small number of iterations."""
    energies, mols = UMA_min(
        mols,
        str(model_path),
        maxIters=1,
        fmax=0.05,
        fexit=250,
        device="cuda",
        task_name="omol",
    )
    # Conformers will not have been minimised in 1 iteration and so will be removed.
    assert all([energy == {} for energy in energies.values()])
    assert all([mol.GetNumConformers() == 0 for mol in mols.values()])


@pytest.mark.cpu
def test_UMA_min_cpu(mols: dict[str, Chem.Mol], model_path: str):
    """Test UMA minimisation on CPU with a small number of iterations."""
    energies, mols = UMA_min(
        mols,
        str(model_path),
        maxIters=1,
        fmax=0.05,
        fexit=250,
        device="cpu",
        task_name="omol",
    )
    # Conformers will not have been minimised in 1 iteration and so will be removed.
    assert all([energy == {} for energy in energies.values()])
    assert all([mol.GetNumConformers() == 0 for mol in mols.values()]) 