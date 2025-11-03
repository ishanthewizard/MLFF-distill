from rdf_single_official import main
from pathlib import Path

def rdf_wrapper(cation: str = "Na", traj_paths: dict = None, out_root: Path = None, first_n_frames: int = None):
    main(cation=cation, traj_paths=traj_paths, out_root=out_root, first_n_frames=first_n_frames)

if __name__ == "__main__":
    cations = [
            #    "Li",
            #    "Na",
            #    "Na",
               # "Na",
               "Na",
            #    "Na"
               ]
    first_n_frames = 40000
    traj_paths_list = [
        # # LiPF6
        # {
        #     "pert_10": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/cleanuped_results/ablation_section_3/ablation_md_simulation_1ns_wwo_hessian_pert_10/md_omol_lipf6_pfactor_0.1_1fs_mask_t/md_omol_lipf6_pfactor_0.1_1fs_mask_t.traj",
        #     "w_o hessian": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/cleanuped_results/ablation_section_3/ablation_md_simulation_1ns_wwo_hessian_pert_10/md_omol_lipf6_pfactor_0.1_1fs_mask_t_wo_hessian/md_omol_lipf6_pfactor_0.1_1fs_mask_t_wo_hessian.traj",
        # },
        # # NaOTF_PC
        # {
        #     "pert_10": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/cleanuped_results/ablation_section_3/ablation_md_simulation_1ns_wwo_hessian_pert_10/md_omol_naotf_pc_1m_s1p1/md_omol_naotf_pc_1m_s1p1.traj",
        #     "w_o hessian": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/cleanuped_results/ablation_section_3/ablation_md_simulation_1ns_wwo_hessian_pert_10/md_omol_naotf_pc_1m_s1p1_wo_hessian/md_omol_naotf_pc_1m_s1p1_wo_hessian.traj",
        # },
        
        # # NaOTF_TGDME
        # {
        #     "pert_10": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/cleanuped_results/ablation_section_3/napf6/md_omol_napf6_tgdme_1m_s1p1_w_hessian/md_omol_napf6_tgdme_1m_s1p1_w_hessian.traj",
        #     "w_o hessian": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/cleanuped_results/ablation_section_3/napf6/md_omol_napf6_tgdme_1m_s1p1_wo_hessian/md_omol_napf6_tgdme_1m_s1p1_wo_hessian.traj",
        # },
        
        
        # # NaOTF_DME
        # {
        #     "pert_10": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/cleanuped_results/ablation_section_3/napf6/md_omol_napf6_dme_re1_w_hessian/md_omol_napf6_dme_re1_w_hessian.traj",
        #     "w_o hessian": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/cleanuped_results/ablation_section_3/napf6/md_omol_napf6_dme_re1_wo_hessian/md_omol_napf6_dme_re1_wo_hessian.traj",
        # },
        
        # # NaOTF_DEG
        # {
        #     "pert_10": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/cleanuped_results/ablation_section_3/napf6/md_omol_napf6_diethyleneglycol_pfactor_0.1_1fs_mask_t_w_hessian/md_omol_napf6_diethyleneglycol_pfactor_0.1_1fs_mask_t_w_hessian.traj",
        #     "w_o hessian": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/cleanuped_results/ablation_section_3/napf6/md_omol_napf6_diethyleneglycol_pfactor_0.1_1fs_mask_t_wo_hessian/md_omol_napf6_diethyleneglycol_pfactor_0.1_1fs_mask_t_wo_hessian.traj",
        #     "uma": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/323_UMA_simulation_ckpt/MD_data_rest/md_omol_napf6_diethyleneglycol_pfactor_0.1_1fs_mask_t/md_omol_napf6_diethyleneglycol_pfactor_0.1_1fs_mask_t.traj",
        # },
        # NaOTF_DME
        {
            "pert_10": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/cleanuped_results/ablation_section_3/napf6/md_omol_napf6_dme_re1_w_hessian/md_omol_napf6_dme_re1_w_hessian.traj",
            "w_o hessian": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/cleanuped_results/ablation_section_3/napf6/md_omol_napf6_dme_re1_wo_hessian/md_omol_napf6_dme_re1_wo_hessian.traj",
            "uma": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/323_UMA_simulation_ckpt/MD_data_rest/md_omol_napf6_dme_re1_pfactor_0.1_1fs_mask_t/md_omol_napf6_dme_re1_pfactor_0.1_1fs_mask_t.traj",
        },
        # # NaOTF_COL
        # {
        #     "pert_500": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/cleanuped_results/ablation_section_3/ablate_cols/md_omol_naotf_pc_1m_s1p1_500/md_omol_naotf_pc_1m_s1p1_500.traj",
        #     "pert_80": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/cleanuped_results/ablation_section_3/ablate_cols/md_omol_naotf_pc_1m_s1p1_80/md_omol_naotf_pc_1m_s1p1_80.traj",
        #     "pert_50": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/cleanuped_results/ablation_section_3/ablate_cols/md_omol_naotf_pc_1m_s1p1_50/md_omol_naotf_pc_1m_s1p1_50.traj",
        #     "pert_20": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/cleanuped_results/ablation_section_3/ablate_cols/md_omol_naotf_pc_1m_s1p1_20/md_omol_naotf_pc_1m_s1p1_20.traj",
        #     "pert_10": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/cleanuped_results/ablation_section_3/ablate_cols/md_omol_naotf_pc_1m_s1p1_10/md_omol_naotf_pc_1m_s1p1_10.traj",
        #     "w_o hessian": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/cleanuped_results/ablation_section_3/ablate_cols/md_omol_naotf_pc_1m_s1p1_wo_hessian/md_omol_naotf_pc_1m_s1p1_wo_hessian.traj",
        # }
    ]
    out_root_dir = Path("/home/yuejian/project/MLFF-distill/ablate_distillation/rdf_output/section_3_ablation/ablate_with_vs_without_hessian")
    out_root_dir.mkdir(parents=True, exist_ok=True)
    out_subroot_list = [
            # out_root_dir / "lipf6",
            # out_root_dir / "naotf_pc",
            # out_root_dir / "naopf6_tgdme",
            # out_root_dir / "naopf6_dme",
            # out_root_dir / "naopf6_deg",
            # out_root_dir / "uma_napf6_deg_400ps",
            out_root_dir / "uma_napf6_dme_400ps",
            # out_root_dir / "naotf_col"
    ]
    for cation, traj_paths, out_subroot in zip(cations, traj_paths_list, out_subroot_list):
        rdf_wrapper(cation=cation, traj_paths=traj_paths, out_root=out_subroot, first_n_frames=first_n_frames)