# scripts/extract_model_checkpoint.py
import os
import torch
from omegaconf import OmegaConf

# 1) Import the MLIPInferenceCheckpoint class so we can allowlist it
from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint

# 2) Point at your inference‐style .pt file
CKPT_PATH = "/data/ishan-amin/ESEN_checkpoints/esen_omol/esen_md_direct_all.pt"
OUT_DIR  = "extracted_configs"
os.makedirs(OUT_DIR, exist_ok=True)

# 3) Load under a safe_globals context so PyTorch knows it's allowed
with torch.serialization.safe_globals([MLIPInferenceCheckpoint]):
    # weights_only=False ensures we get the entire checkpoint object, not just raw tensors
    ckpt: MLIPInferenceCheckpoint = torch.load(
        CKPT_PATH, 
        map_location="cpu", 
        weights_only=False
    )

# 4) Extract the Hydra DictConfigs
model_cfg = ckpt.model_config
tasks_cfg = ckpt.tasks_config

# 5) Write them out as YAML
OmegaConf.save(model_cfg, os.path.join(OUT_DIR, "model_config.yaml"))
OmegaConf.save(tasks_cfg, os.path.join(OUT_DIR, "tasks_config.yaml"))

print(f"Wrote:\n  {OUT_DIR}/model_config.yaml\n  {OUT_DIR}/tasks_config.yaml")
