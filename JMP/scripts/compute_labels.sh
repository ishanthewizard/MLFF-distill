python compute_jmp_labels.py \
    --data_path /data/shared/ishan_stuff/md22 \
    --checkpoint_path /data/shared/ishan_stuff/nanotube_jmp-s.ckpt \
    --split val \
    --mode finetune \
    --molecule double-walled_nanotube \
    --direct_forces

python compute_jmp_labels.py \
    --data_path /data/shared/ishan_stuff/md22 \
    --checkpoint_path /data/shared/ishan_stuff/nanotube_jmp-s.ckpt \
    --split train \
    --mode finetune \
    --molecule double-walled_nanotube \
    --direct_forces






