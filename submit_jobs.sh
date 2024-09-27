#!/bin/bash


# python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/iodine/distill/gemnet-dT-small-teacherforces.yml --identifier iodine-gemSmall-DIST-teacherforces
# python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/monomers/distill/gemnet-dT-small-teacherforces.yml --identifier monomers-gemSmall-DIST-teacherforces

###### Hessian Subsampling Expts ##############
# python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small.yml --optim.force_jac_sample_size=1 --identifier solvated-gemSmall-DIST-s1
python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small.yml --optim.force_jac_sample_size=15 --identifier solvated-gemSmall-DIST-s15
# python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/painn-small.yml --optim.force_jac_sample_size=1 --identifier solvated-PaiNN-DIST-pbc-s1
python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/painn-small.yml --optim.force_jac_sample_size=10 --identifier solvated-PaiNN-DIST-pbc-s10
python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/painn-small.yml --optim.force_jac_sample_size=15 --identifier solvated-PaiNN-DIST-pbc-s15

###### MPTraj Baseline Expts #################
python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/MPTraj/Bandgap_greater_5/distill/gemnet-dT-small-n2n.yml --identifier Bandgap-gemSmall-DIST-n2n
python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/MPTraj/Bandgap_greater_5/distill/gemnet-dT-small-atomemb.yml --identifier Bandgap-gemSmall-DIST-atomemb
python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/MPTraj/Bandgap_greater_5/distill/painn-small-n2n.yml --identifier Bandgap-PaiNN-DIST-n2n
python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/MPTraj/Bandgap_greater_5/distill/painn-small-atomemb.yml --identifier Bandgap-PaiNN-DIST-atomemb

python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/MPTraj/Perovskites/distill/gemnet-dT-small-n2n.yml --identifier Perov-gemSmall-DIST-n2n
python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/MPTraj/Perovskites/distill/gemnet-dT-small-atomemb.yml --identifier Perov-gemSmall-DIST-atomemb
python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/MPTraj/Perovskites/distill/painn-small-n2n.yml --identifier Perov-PaiNN-DIST-n2n
python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/MPTraj/Perovskites/distill/painn-small-atomemb.yml --identifier Perov-PaiNN-DIST-atomemb

python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/MPTraj/Yttrium/distill/gemnet-dT-small-n2n.yml --identifier Ytt-gemSmall-DIST-n2n
python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/MPTraj/Yttrium/distill/gemnet-dT-small-atomemb.yml --identifier Ytt-gemSmall-DIST-atomemb
python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/MPTraj/Yttrium/distill/painn-small-n2n.yml --identifier Ytt-PaiNN-DIST-n2n
python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/MPTraj/Yttrium/distill/painn-small-atomemb.yml --identifier Ytt-PaiNN-DIST-atomemb