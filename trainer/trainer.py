import copy

import deepspeed
from transformers import Trainer

from .losses import get_loss


class CustomTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop('loss_type')
        self.ref_model = kwargs.pop('ref_model')

        self.forget_coeff = kwargs.pop('forget_coeff')
        self.regularization_coeff = kwargs.pop('regularization_coeff')
        self.beta = kwargs.pop('beta')

        super(CustomTrainerForgetting, self).__init__(*args, **kwargs)

        self.ref_model = self.e_prepare_deepspeed(self.ref_model)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        forget_loss, regularization_loss = get_loss(model, self.ref_model, inputs, self.loss_type, self.beta)
        loss = self.forget_coeff * forget_loss + self.regularization_coeff * regularization_loss

        return (loss, None) if return_outputs else loss

    def e_prepare_deepspeed(self, model):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        return model
