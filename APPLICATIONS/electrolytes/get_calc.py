from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from ase.calculators.calculator import Calculator
import numpy as np

uma_path = "/data/ishan-amin/OMOL/ESEN_OMol_ckpts/uma-s-1p1.pt"

class UMACalculatorWrapper(Calculator):
    """Wrapper around FAIRChemCalculator to implement ASE Calculator interface"""
    
    # Declare which properties this calculator implements
    implemented_properties = ['energy', 'forces', 'stress']
    
    def __init__(self, predictor, task_name="oc20"):
        Calculator.__init__(self)
        self.fairchem_calc = FAIRChemCalculator(predictor, task_name=task_name)
        self.counter = 0
        
    def calculate(self, atoms, properties=['energy', 'forces'], system_changes=None):
        """Implement ASE Calculator interface"""

        Calculator.calculate(self, atoms, properties, system_changes)
        
        # Preprocess atoms.info["stress"] to 3x3 if in Voigt form
        stress = atoms.info.get("stress")
        if isinstance(stress, (list, np.ndarray)) and len(stress) == 6:
            atoms.info["stress"] = np.array(voigt_to_tensor6(stress))
        
        # Call the fairchem calculator with only the required arguments
        self.fairchem_calc.calculate(atoms, properties, system_changes)
        
        # Copy results from fairchem calculator
        self.results = self.fairchem_calc.results.copy()
        
        # If stress was requested but not computed, try to get it from fairchem calculator
        if 'stress' in properties and 'stress' not in self.results:
            try:
                # Try to get stress from the underlying calculator
                stress = self.fairchem_calc.get_stress(atoms)
                if stress is not None:
                    self.results['stress'] = stress
                else:
                    # If no stress available, create a zero stress tensor as fallback
                    # This is not ideal but prevents crashes
                    print("Warning: Stress calculation not available, using zero stress tensor")
                    self.results['stress'] = np.zeros((3, 3))
            except (AttributeError, NotImplementedError):
                # If stress calculation is not supported, use zero stress tensor
                print("Warning: Stress calculation not supported by underlying calculator, using zero stress tensor")
                self.results['stress'] = np.zeros((3, 3))
        
        self.counter += 1

def get_uma_calc():
    predictor = load_predict_unit(uma_path, device="cuda")
    calc = UMACalculatorWrapper(predictor, task_name="oc20")
    return calc

def voigt_to_tensor6(six):
    # Assumes six = [xx, yy, zz, yz, xz, xy]
    tensor = [
        [six[0], six[5], six[4]],
        [six[5], six[1], six[3]],
        [six[4], six[3], six[2]]
    ]
    return tensor
