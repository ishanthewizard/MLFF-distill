import torch


class model_with_matching_label_output(torch.nn.Module):
    """
    A model wrapper that adds a method to compute any label for distillation matching use.
    In the mean time, it does not change the model output dictionary form, so the model within it is replaceable
    """
    # TODO: should also pass in the distillation task and it should be a list, 
    # this is to support multiple distillation tasks in the future. Ensemble distillation is one of the examples.
    def __init__(self, student_model: torch.nn.Module):
        super().__init__()
        self.student_model = student_model

    def forward(self, *args, **kwargs):
        
        pred_dict = self.student_model(*args, **kwargs)
        
        matching_label = self.get_matching_label(*args, **kwargs)
        pred_dict["matching_label"] = matching_label
        
        return pred_dict

    def get_matching_label(self, *args, **kwargs):
        """
        This method should be implemented in the student model to return the matching label for distillation.
        """
        raise NotImplementedError("This method should be implemented in the student model.")

    
