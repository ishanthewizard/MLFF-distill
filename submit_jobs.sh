#!/bin/bash


# python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/iodine/distill/gemnet-dT-small-teacherforces.yml --identifier iodine-gemSmall-DIST-teacherforces
# python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/monomers/distill/gemnet-dT-small-teacherforces.yml --identifier monomers-gemSmall-DIST-teacherforces

# python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small.yml --optim.force_jac_sample_size=1 --identifier solvated-gemSmall-DIST-s1
python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small.yml --optim.force_jac_sample_size=15 --identifier solvated-gemSmall-DIST-s15


# python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/painn-small.yml --optim.force_jac_sample_size=1 --identifier solvated-PaiNN-DIST-pbc-s1
python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/painn-small.yml --optim.force_jac_sample_size=10 --identifier solvated-PaiNN-DIST-pbc-s10
python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/painn-small.yml --optim.force_jac_sample_size=15 --identifier solvated-PaiNN-DIST-pbc-s15

