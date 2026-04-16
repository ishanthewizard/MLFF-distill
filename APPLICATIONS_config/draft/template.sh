
# UMA no opt
python APPLICATIONS/electrolytes/solv_uma_npt_flex_ablation.py \
    /global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablation_speed/UMA_no_opt/md_omol_naotf_diglyme_1m_s1p1 \
    --models /global/homes/y/yuejian/project/MLFF-distill/m4558/distillation_project/models/uma-s-1p1.pt \
    --steps 10000000 \
    --interval 10
 
# UMA opt
python APPLICATIONS/electrolytes/solv_uma_npt_flex_ablation.py \
    /global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablation_speed/UMA_opt/md_omol_naotf_diglyme_1m_s1p1 \
    --models /global/homes/y/yuejian/project/MLFF-distill/m4558/distillation_project/models/uma-s-1p1.pt \
    --steps 10000000 \
    --interval 10


# student model
python APPLICATIONS/electrolytes/solv_uma_npt_flex_ablation.py \
    /global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/unformal_ablation/ablation_speed/student_opt/md_omol_naotf_diglyme_1m_s1p1 \
    --models /global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/unformal_ablation/ablation_speed/student_ckpt/inference_ckpt.pt \
    --steps 10000000 \
    --interval 10

# student model requeue
python APPLICATIONS/electrolytes/solv_uma_npt_flex_ablation.py \
    /global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablation_speed/requeue/md_omol_naotf_diglyme_1m_s1p1 \
    --models /global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablation_speed/student_ckpt/inference_ckpt.pt \
    --steps 10000000 \
    --interval 10