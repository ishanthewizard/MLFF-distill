#!/bin/bash

# n2n expts (varying loss weight)
python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-n2n-weight10.yml --identifier Solvated-gemSmall-dist-n2n_weight10 --optim.batch_size=1
python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-n2n-weight100.yml --identifier Solvated-gemSmall-dist-n2n_weight100 --optim.batch_size=1
python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-n2n-weight1000.yml --identifier Solvated-gemSmall-dist-n2n_weight1000 --optim.batch_size=1
python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-n2n-weight10000.yml --identifier Solvated-gemSmall-dist-n2n_weight10000 --optim.batch_size=1

# distillation runs (varying loss weight)
python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-weight10.yml --identifier Solvated-gemSmall-dist-forcejacweight10 --optim.batch_size=1
python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-weight100.yml --identifier Solvated-gemSmall-dist-forcejacweight100 --optim.batch_size=1
python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-weight400.yml --identifier Solvated-gemSmall-dist-forcejacweight400 --optim.batch_size=1
python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-weight1000.yml --identifier Solvated-gemSmall-dist-forcejacweight10000 --optim.batch_size=1

# teacher forces on gemnet and PaiNN
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-teacherforces.yml --identifier Solvated-gemSmall-dist-teacherforces --optim.batch_size=1
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/painn-small-teacherforces.yml --identifier Solvated-painnSmall-dist-teacherforces --optim.batch_size=1


# # atom embeddings on gemnet and PaiNN
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-atomemb.yml --identifier Solvated-gemSmall-dist-atomemb --optim.batch_size=1
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/painn-small-atomemb.yml --identifier Solvated-painnSmall-dist-atomemb --optim.batch_size=1