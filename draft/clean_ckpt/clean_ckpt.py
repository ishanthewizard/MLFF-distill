import torch
import os

def clean_checkpoint(root_dir):
    """
    Clean checkpoint by removing the last task from tasks_config.
    
    Args:
        root_dir (str): Root directory containing checkpoints/final/inference_ckpt.pt
    """
    ckpt_path = os.path.join(root_dir, "checkpoints/final/inference_ckpt.pt")
    output_path = os.path.join(root_dir, "cleaned_inference_ckpt.pt")
    
    try:
        print(f"Processing: {root_dir}")
        
        # Check if input file exists
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint file not found: {ckpt_path}")
            return False
            
        # Load checkpoint
        config = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        # Remove last task from tasks_config
        config.tasks_config.pop(-1)
        
        # Save cleaned checkpoint
        torch.save(config, output_path)
        print(f"Successfully saved cleaned checkpoint to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {root_dir}: {str(e)}")
        return False

def main():
    # List of root directories to process
    root_directories = [
        "/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablation_ckpt/small_batch/202510-2322-2744-5337-napf6-10",
        # Add more root directories here as needed
        # "/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablation_ckpt/small_batch/202510-1614-0433-0ca3-naotf-10",
        # "/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablation_ckpt/small_batch/202510-1614-1411-1784-lipf6-20",
        # "/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablation_ckpt/small_batch/202510-1714-3239-2d8b-naotf-20",
        # "/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablation_ckpt/small_batch/202510-1714-3305-c9b6-naotf-50",
        # "/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablation_ckpt/small_batch/202510-1714-3336-8b81-naotf-80",
        # "/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablation_ckpt/small_batch/202510-1714-3356-164a-naotf-500",
    ]
    
    print(f"Processing {len(root_directories)} root directories...")
    
    success_count = 0
    for root_dir in root_directories:
        if clean_checkpoint(root_dir):
            success_count += 1
    
    print(f"\nCompleted: {success_count}/{len(root_directories)} directories processed successfully")

if __name__ == "__main__":
    main()