#!/bin/bash


###### SPICE GemNet-T Baselines (total 6 runs) ##############
python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/iodine/distill/gemnet-T-small-atomemb.yml --identifier iodine-gemTSmall-atomemb --logger.project="spice_baselines" --optim.batch_size=1
python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/iodine/distill/gemnet-T-small-n2n.yml --identifier iodine-gemTSmall-n2n --logger.project="spice_baselines" --optim.batch_size=1

python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/monomers/distill/gemnet-T-small-atomemb.yml --identifier monomers-gemTSmall-atomemb --logger.project="spice_baselines" --optim.batch_size=1
python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/monomers/distill/gemnet-T-small-n2n.yml --identifier monomers-gemTSmall-n2n --logger.project="spice_baselines" --optim.batch_size=1

python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-T-small-atomemb.yml --identifier solvated-gemTSmall-atomemb --logger.project="spice_baselines" --optim.batch_size=1
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-T-small-n2n.yml --identifier solvated-gemTSmall-n2n --logger.project="spice_baselines" --optim.batch_size=1