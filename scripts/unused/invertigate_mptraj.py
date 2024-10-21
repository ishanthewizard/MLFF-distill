

# Paths
from fairchem.core.common.registry import registry


output_path = '/data/ishan-amin/MPtraj/mptraj_seperated_all_splits/Yttrium/'
output_label_path = 'labels/mace_mp_all_splits_Yttrium'

eric_lmdb_path = '/data/ishan-amin/mptraj_seperated_eric/Perovskites/train'
my_lmdb_path = '/data/ishan-amin/MPtraj/mptraj_seperated/Perovskites/train'
my_dataset = registry.get_dataset_class("lmdb")({"src": my_lmdb_path})
eric_dataset = registry.get_dataset_class("lmdb")({"src": my_lmdb_path})

for sample in my_dataset:
    if len(sample.pos) == 1:
        for eric_samp in eric_dataset:
            if eric_samp.mp_id == sample.mp_id:
                print(sample.pos)
                print(eric_samp.pos)
