#!/bin/bash


python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/iodine/distill/gemnet-dT-small-n2n.yml --identifier iodine-gemSmall-DIST-n2n-correctemb
python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/iodine/distill/gemnet-dT-small-atomemb.yml --identifier iodine-gemSmall-DIST-atomemb-correctemb

python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/monomers/distill/gemnet-dT-small-n2n.yml --identifier monomers-gemSmall-DIST-n2n-correctemb
python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/monomers/distill/gemnet-dT-small-atomemb.yml --identifier monomers-gemSmall-DIST-atomemb-correctemb

# python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-n2n.yml --identifier solvated-gemSmall-DIST-n2n
# python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/gemnet-dT-small-atomemb.yml --identifier solvated-gemSmall-DIST-atomemb


python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/iodine/distill/painn-small-n2n.yml --identifier iodine-PaiNN-DIST-n2n-correctemb
python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/iodine/distill/painn-small-atomemb.yml --identifier iodine-PaiNN-DIST-atomemb-correctemb

python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/monomers/distill/painn-small-n2n.yml --identifier monomers-PaiNN-DIST-n2n-correctemb
python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/monomers/distill/painn-small-atomemb.yml --identifier monomers-PaiNN-DIST-atomemb-correctemb

# python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/painn-small-n2n.yml --identifier solvated-PaiNN-DIST-n2n
# python main.py --num-gpus 1 --submit --nersc --mode train --config-yml configs/SPICE/solvated_amino_acids/distill/painn-small-atomemb.yml --identifier solvated-PaiNN-DIST-atomemb