#!/bin/bash


python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/iodine/distill/gemnet-dT-small-teacherforces.yml --identifier iodine-gemSmall-DIST-teacherforces
python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/monomers/distill/gemnet-dT-small-teacher.yml --identifier monomers-gemSmall-DIST-teacherforces

python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/iodine/distill/painn-small-teacherforces.yml --identifier iodine-PaiNN-DIST-teacherforces
python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/monomers/distill/painn-small-teacher.yml --identifier monomers-PaiNN-DIST-teacherforces
