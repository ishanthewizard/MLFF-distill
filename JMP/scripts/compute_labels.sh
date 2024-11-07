python compute_jmp_labels.py \
    --data_path /data/shared/ishan_stuff/md22 \
    --checkpoint_path /data/shared/ishan_stuff/buckyball-catcher_jmp-s.ckpt \
    --split val \
    --mode finetune \
    --molecule buckyball-catcher \
    --direct_forces

python compute_jmp_labels.py \
    --data_path /data/shared/ishan_stuff/md22 \
    --checkpoint_path /data/shared/ishan_stuff/buckyball-catcher_jmp-s.ckpt \
    --split train \
    --mode finetune \
    --molecule buckyball-catcher \
    --direct_forces






