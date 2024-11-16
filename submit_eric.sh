#! /bin/bash

function launch_slurm {
    config=$1
    identifier=$2
    cmd="/private/home/ericqu/.conda/envs/distill/bin/python /private/home/ericqu/MLFF-distill/main.py \
    --num-nodes 1 --num-gpus 4 --submit --mode train \
    --config-yml ${config} \
    --identifier ${identifier} \
    --optim.batch_size=1 --logger.project="spice_baselines" --run-dir /checkpoint/ericqu/distill \
    --slurm-mem 480 --slurm-partition learnaccel --slurm-timeout 24"
    echo $cmd
    $cmd
}

# launch_slurm "configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-n2n-weight10.yml" "Solvated-gemSmall-dist-n2n_weight10"
# launch_slurm "configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-n2n-weight100.yml" "Solvated-gemSmall-dist-n2n_weight100"
launch_slurm "configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-n2n-weight1000.yml" "Solvated-gemSmall-dist-n2n_weight1000"
launch_slurm "configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-n2n-weight10000.yml" "Solvated-gemSmall-dist-n2n_weight10000"

launch_slurm "configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-weight10.yml" "Solvated-gemSmall-dist-forcejacweight10"
launch_slurm "configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-weight100.yml" "Solvated-gemSmall-dist-forcejacweight100"
launch_slurm "configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-weight400.yml" "Solvated-gemSmall-dist-forcejacweight400"
launch_slurm "configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-weight1000.yml" "Solvated-gemSmall-dist-forcejacweight10000"

launch_slurm "configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-teacherforces.yml" "Solvated-gemSmall-dist-teacherforces"
launch_slurm "configs/SPICE/solvated_amino_acids/distill/painn-small-teacherforces.yml" "Solvated-painnSmall-dist-teacherforces"

launch_slurm "configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-atomemb.yml" "Solvated-gemSmall-dist-atomemb"
launch_slurm "configs/SPICE/solvated_amino_acids/distill/painn-small-atomemb.yml" "Solvated-painnSmall-dist-atomemb"

launch_slurm "configs/SPICE/monomers/distill/gemnet-dT-small-teacherforces.yml" "Monomers-gemSmall-dist-teacherforces"
launch_slurm "configs/SPICE/monomers/distill/painn-small-teacherforces.yml" "Monomers-painnSmall-dist-teacherforces"

launch_slurm "configs/SPICE/iodine/distill/gemnet-dT-small-teacherforces.yml" "Iodine-gemSmall-dist-teacherforces"
launch_slurm "configs/SPICE/iodine/distill/painn-small-teacherforces.yml" "Iodine-painnSmall-dist-teacherforces"

launch_slurm "configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-atomemb.yml" "Solvated-gemSmall-dist-atomemb"
launch_slurm "configs/SPICE/solvated_amino_acids/distill/painn-small-atomemb.yml" "Solvated-painnSmall-dist-atomemb"

launch_slurm "configs/SPICE/monomers/distill/gemnet-dT-small-atomemb.yml" "Monomers-gemSmall-dist-atomemb"
launch_slurm "configs/SPICE/monomers/distill/painn-small-atomemb.yml" "Monomers-painnSmall-dist-atomemb"

launch_slurm "configs/SPICE/iodine/distill/gemnet-dT-small-atomemb.yml" "Iodine-gemSmall-dist-atomemb"
launch_slurm "configs/SPICE/iodine/distill/painn-small-atomemb.yml" "Iodine-painnSmall-dist-atomemb"

# optimal n2n weight runs on iodine and monomers (2 splits x 2 models = 4 runs)
# optimal n2n weight run on solvated with PaiNN (1 split x 1 model = 1 run)