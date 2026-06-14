from functools import partial

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from ase.calculators.calculator import Calculator
import numpy as np
from fairchem.core.units.mlip_unit import InferenceSettings


class UMACalculatorWrapper(Calculator):
    """Wrapper around FAIRChemCalculator to implement ASE Calculator interface"""

    # Declare which properties this calculator implements
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, predictor, task_name="oc20", freeze_graph=True):
        Calculator.__init__(self)
        self.fairchem_calc = FAIRChemCalculator(predictor, task_name=task_name)
        self.counter = 0
        self.freeze_graph = freeze_graph
        self._frozen_edge_index = None
        self._frozen_cell_offsets = None
        self._frozen_nedges = None

    def calculate(self, atoms, properties=['energy', 'forces'], system_changes=None):
        """Implement ASE Calculator interface"""

        Calculator.calculate(self, atoms, properties, system_changes)

        # Preprocess atoms.info["stress"] to 3x3 if in Voigt form
        stress = atoms.info.get("stress")
        if isinstance(stress, (list, np.ndarray)) and len(stress) == 6:
            atoms.info["stress"] = np.array(voigt_to_tensor6(stress))

        # # Step 0: build the graph once, cache topology, then flip otf_graph=False
        # if self.freeze_graph and self._frozen_edge_index is None:
        #     from fairchem.core.graph.compute import generate_graph

        #     original_a2g = self.fairchem_calc.a2g
        #     model_module = self.fairchem_calc.predictor.model.module
        #     # HydraModel wraps a backbone; cutoff/max_neighbors live there
        #     backbone = getattr(model_module, 'backbone', model_module)

        #     data_cpu = original_a2g(atoms)
        #     graph_dict = generate_graph(
        #         data_cpu,
        #         cutoff=backbone.cutoff,
        #         max_neighbors=backbone.max_neighbors,
        #         enforce_max_neighbors_strictly=getattr(backbone, 'enforce_max_neighbors_strictly', False),
        #         radius_pbc_version=1,
        #         pbc=data_cpu.pbc,
        #     )
        #     self._frozen_edge_index = graph_dict["edge_index"].clone()
        #     self._frozen_cell_offsets = graph_dict["cell_offsets"].clone()
        #     self._frozen_nedges = graph_dict["neighbors"].clone()

        #     backbone.otf_graph = False

        #     _ei = self._frozen_edge_index
        #     _co = self._frozen_cell_offsets
        #     _ne = self._frozen_nedges

        #     def _frozen_a2g(atoms_obj):
        #         data = original_a2g(atoms_obj)
        #         data.edge_index = _ei
        #         data.cell_offsets = _co
        #         data.nedges = _ne
        #         return data

        #     self.fairchem_calc.a2g = _frozen_a2g
        #     print(f"Graph frozen: {_ne.item()} edges cached, reusing topology for all steps")

        self.fairchem_calc.calculate(atoms, properties, system_changes)

        self.results = self.fairchem_calc.results.copy()

        # If stress was requested but not computed, try to get it from fairchem calculator
        if 'stress' in properties and 'stress' not in self.results:
            try:
                stress = self.fairchem_calc.get_stress(atoms)
                if stress is not None:
                    self.results['stress'] = stress
                else:
                    print("Warning: Stress calculation not available, using zero stress tensor")
                    self.results['stress'] = np.zeros((3, 3))
            except (AttributeError, NotImplementedError):
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
        tf32=True,
        activation_checkpointing=False,
        merge_mole=True,
        compile=True,
        wigner_cuda=False,
        external_graph_gen=False,
        internal_graph_gen_version=3,
    )
    predictor = load_predict_unit(uma_path, device="cuda", inference_settings=inference_settings)
    calc = UMACalculatorWrapper(predictor, task_name="omol", freeze_graph=True)
    return calc

def get_customized_uma_calc_frozen(uma_path):
    """Same settings as get_customized_uma_calc but with graph frozen after step 0.

    external_graph_gen=False keeps the model in otf_graph=True mode initially.
    On the first calculate() call, UMACalculatorWrapper.freeze_graph=True builds
    the neighbor list once using the model's own parameters via the CPU-compatible
    v1 kernel (no pymatgen), then flips model.otf_graph=False and installs a frozen
    a2g that injects cached edge_index/cell_offsets/nedges every step — skipping
    the expensive neighbor-list rebuild while still running the full NN forward pass.
    """
    # --- FREEZE_GRAPH CHANGE: external_graph_gen=False so a2g never calls pymatgen ---
    inference_settings = InferenceSettings(
        tf32=True,
        activation_checkpointing=False,
        merge_mole=True,
        compile=True,
        wigner_cuda=False,
        external_graph_gen=False,  # a2g uses r_edges=False; we inject graph ourselves
        internal_graph_gen_version=3,
    )
    # --- END FREEZE_GRAPH CHANGE ---
    predictor = load_predict_unit(uma_path, device="cuda", inference_settings=inference_settings, overrides={'_target_': 'fairchem.core.models.base.HydraModel'})
    calc = UMACalculatorWrapper(predictor, task_name="omol", freeze_graph=True)
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
