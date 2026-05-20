


class Dataset:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.load_dataset()

    def load_dataset(self):
        self.dataset = AseDBDataset({"src": str(self.input_dir)})

    def apply_augmentation(self, augmentation):
        if augmentation == "volume_preserving_distortion":
            self.dataset = self.sequential_apply_augmentations_to_frames( frame=self.dataset, augmentations=[volume_preserving_distortion] )
        elif augmentation == "rattle":
            self.dataset = self.sequential_apply_augmentations_to_frames( frame=self.dataset, augmentations=[rattle] )
        elif augmentation == "distort":
            self.dataset = self.sequential_apply_augmentations_to_frames( frame=self.dataset, augmentations=[distort] )
    
    def sequential_apply_augmentations_to_frames(self, frame, augmentations):
        pass

    def relabel_with_new_calculator(self, calculator_path):
        self.dataset = self.dataset.relabel_with_new_calculator(calculator_path)

    def save(self, output_dir):
        self.dataset.save(output_dir)


def volume_preserving_distortion(frame):
    pass


def rattle(frame):
    pass