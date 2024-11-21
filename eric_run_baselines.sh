#!/bin/bash

### TUNING RUNS####
# n2n expts (varying loss weight)
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-n2n-weight10.yml --identifier Solvated-gemSmall-dist-n2n_weight10 --optim.batch_size=1 --logger.project="spice_baselines"
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-n2n-weight100.yml --identifier Solvated-gemSmall-dist-n2n_weight100 --optim.batch_size=1 --logger.project="spice_baselines"
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-n2n-weight1000.yml --identifier Solvated-gemSmall-dist-n2n_weight1000 --optim.batch_size=1 --logger.project="spice_baselines"
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode=train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-n2n-weight10000.yml --identifier=Solvated-gemSmall-dist-n2n_weight10000 --optim.batch_size=1 --logger.project="spice_baselines"
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode=train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-n2n-weight100000.yml --identifier=Solvated-gemSmall-dist-n2n_weight100000 --optim.batch_size=1 --logger.project="spice_baselines"

# # distillation runs (varying loss weight)
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-weight10.yml --identifier Solvated-gemSmall-dist-forcejacweight10 --optim.batch_size=1 --logger.project="spice_baselines"
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-weight100.yml --identifier Solvated-gemSmall-dist-forcejacweight100 --optim.batch_size=1 --logger.project="spice_baselines"
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-weight400.yml --identifier Solvated-gemSmall-dist-forcejacweight400 --optim.batch_size=1 --logger.project="spice_baselines"
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-weight1000.yml --identifier Solvated-gemSmall-dist-forcejacweight10000 --optim.batch_size=1 --logger.project="spice_baselines"

# ##### NO TUNING RUNS #######
# # teacher forces on gemnet and PaiNN

# # Solvated
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-teacherforces.yml --identifier Solvated-gemSmall-dist-teacherforces --optim.batch_size=1 --logger.project="spice_baselines"
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/painn-small-teacherforces.yml --identifier Solvated-painnSmall-dist-teacherforces --optim.batch_size=1 --logger.project="spice_baselines"

# # Monomers
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/monomers/distill/gemnet-dT-small-teacherforces.yml --identifier Monomers-gemSmall-dist-teacherforces --optim.batch_size=1 --logger.project="spice_baselines"
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/monomers/distill/painn-small-teacherforces.yml --identifier Monomers-painnSmall-dist-teacherforces --optim.batch_size=1 --logger.project="spice_baselines"

# # Iodine
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/iodine/distill/gemnet-dT-small-teacherforces.yml --identifier Iodine-gemSmall-dist-teacherforces --optim.batch_size=1 --logger.project="spice_baselines"
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/iodine/distill/painn-small-teacherforces.yml --identifier Iodine-painnSmall-dist-teacherforces --optim.batch_size=1 --logger.project="spice_baselines"



# # # atom embeddings on gemnet and PaiNN
# # Solvated
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-atomemb.yml --identifier Solvated-gemSmall-dist-atomemb --optim.batch_size=1 --logger.project="spice_baselines"
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/painn-small-atomemb.yml --identifier Solvated-painnSmall-dist-atomemb --optim.batch_size=1 --logger.project="spice_baselines"

# # Monomers
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/monomers/distill/gemnet-dT-small-atomemb.yml --identifier Monomers-gemSmall-dist-atomemb --optim.batch_size=1 --logger.project="spice_baselines"
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/monomers/distill/painn-small-atomemb.yml --identifier Monomers-painnSmall-dist-atomemb --optim.batch_size=1 --logger.project="spice_baselines"

# # Iodine
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/iodine/distill/gemnet-dT-small-atomemb.yml --identifier Iodine-gemSmall-dist-atomemb --optim.batch_size=1 --logger.project="spice_baselines"
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/iodine/distill/painn-small-atomemb.yml --identifier Iodine-painnSmall-dist-atomemb --optim.batch_size=1 --logger.project="spice_baselines"



### Still TODO #### 
# optimal n2n weight runs on iodine and monomers (2 splits x 2 models = 4 runs)
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/iodine/distill/gemnet-dT-small-n2n-weight10000.yml --identifier Iodine-gemSmall-dist-n2n_weight10000 --optim.batch_size=1 --logger.project="spice_baselines"
python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/monomers/distill/gemnet-dT-small-n2n-weight10000.yml --identifier Monomers-gemSmall-dist-n2n_weight10000-continued --checkpoint /global/homes/s/sanjeevr/MLFF-distill/checkpoints/2024-11-18-10-40-00-Monomers-gemSmall-dist-n2n_weight10000/checkpoint.pt --optim.batch_size=1 --logger.project="spice_baselines"

# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/iodine/distill/painn-small-n2n.yml --identifier Iodine-painnSmall-dist-n2n_weight10000 --optim.batch_size=1 --logger.project="spice_baselines"
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/monomers/distill/painn-small-n2n.yml --identifier Monomers-painnSmall-dist-n2n_weight10000 --optim.batch_size=1 --logger.project="spice_baselines"

# # optimal n2n weight run on solvated with PaiNN (1 split x 1 model = 1 run)
# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/painn-small-n2n.yml --identifier Solvated-painnSmall-dist-n2n_weight10000 --optim.batch_size=1 --logger.project="spice_baselines"


# python main.py --num-nodes 1 --num-gpus 4 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-weight0.yml --identifier Solvated-gemSmall-dist-forcejacweight0 --optim.batch_size=1 --logger.project="spice_baselines"

