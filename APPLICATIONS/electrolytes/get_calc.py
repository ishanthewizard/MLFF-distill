from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from ase.calculators.calculator import Calculator
import numpy as np
from fairchem.core.units.mlip_unit import InferenceSettings


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
        

        # atoms.get_positions()
        # Call the fairchem calculator with only the required arguments
        # breakpoint()
        self.fairchem_calc.calculate(atoms, properties, system_changes)
        # breakpoint()
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

def get_uma_calc(uma_path, small_model=False, wigner_cuda=True, activation_checkpointing=False):
    if small_model:
        inference_settings = InferenceSettings(
            tf32=True,
            activation_checkpointing=activation_checkpointing,
            merge_mole=False,
            compile=True,
            wigner_cuda=wigner_cuda,
            external_graph_gen=False,
            internal_graph_gen_version=2,
        )
        predictor = load_predict_unit(uma_path, device="cuda", inference_settings=inference_settings, overrides={'_target_': 'fairchem.core.models.base.HydraModel'})
    else:
        predictor = load_predict_unit(uma_path, device="cuda")
    calc = UMACalculatorWrapper(predictor, task_name="omol")
    return calc

def get_customized_uma_calc(uma_path):
    inference_settings = InferenceSettings(
        tf32=False,
        activation_checkpointing=False,
        merge_mole=True,
        compile=True,
        wigner_cuda=False,
        external_graph_gen=False,
        internal_graph_gen_version=2,
    )
    predictor = load_predict_unit(uma_path, device="cuda", inference_settings=inference_settings, overrides={'_target_': 'fairchem.core.models.base.HydraModel'})
    calc = UMACalculatorWrapper(predictor, task_name="omol")
    return calc

def get_customized_eval_uma_calc(uma_path):
    inference_settings = InferenceSettings(
        tf32=True,
        activation_checkpointing=False,
        merge_mole=False,
        compile=False,
        wigner_cuda=False,
        external_graph_gen=False,
        internal_graph_gen_version=2,
    )
    predictor = load_predict_unit(uma_path, device="cuda", inference_settings=inference_settings, overrides={'_target_': 'fairchem.core.models.base.HydraModel'})
    calc = UMACalculatorWrapper(predictor, task_name="omol")
    return calc

def voigt_to_tensor6(six):
    # Assumes six = [xx, yy, zz, yz, xz, xy]
    tensor = [
        [six[0], six[5], six[4]],
        [six[5], six[1], six[3]],
        [six[4], six[3], six[2]]
    ]
    return tensor
