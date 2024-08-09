import os
import warnings

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from trl.models.modeling_sd_base import DefaultDDPOStableDiffusionPipeline 
from trl.models.sd_utils import convert_state_dict_to_diffusers


class ExtendDDPOStableDiffusionPipeline(DefaultDDPOStableDiffusionPipeline):
    def __init__(self, pretrained_model_name: str, *, pretrained_model_revision: str = "main", train_config=None):
        self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name, revision=pretrained_model_revision
        )

        self.use_lora = train_config.use_lora
        self.pretrained_model = pretrained_model_name
        self.pretrained_revision = pretrained_model_revision

        self.train_config = train_config 

        try:
            self.sd_pipeline.load_lora_weights(
                pretrained_model_name,
                weight_name="pytorch_lora_weights.safetensors",
                revision=pretrained_model_revision,
            )
            self.use_lora = True
        except OSError:
            if self.use_lora:
                warnings.warn(
                    "If you are aware that the pretrained model has no lora weights to it, ignore this message. "
                    "Otherwise please check the if `pytorch_lora_weights.safetensors` exists in the model folder."
                )

        self.sd_pipeline.scheduler = DDIMScheduler.from_config(self.sd_pipeline.scheduler.config)
        self.sd_pipeline.safety_checker = None

        # memory optimization
        self.sd_pipeline.vae.requires_grad_(False)
        self.sd_pipeline.text_encoder.requires_grad_(False)
        self.sd_pipeline.unet.requires_grad_(False)

    def get_trainable_layers(self):
        if self.use_lora:
            trainable_params = []

            # Unet lora
            if self.train_config.unet_lora_r > 0:
                lora_config = LoraConfig(
                    r=self.train_config.unet_lora_r,
                    lora_alpha=self.train_config.unet_lora_alpha,
                    init_lora_weights="gaussian",
                    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                )
                self.sd_pipeline.unet.add_adapter(lora_config)

                # To avoid accelerate unscaling problems in FP16.
                for param in self.sd_pipeline.unet.parameters():
                    # only upcast trainable parameters (LoRA) into fp32
                    if param.requires_grad:
                        param.data = param.to(torch.float32)
                        trainable_params.append(param)
            
            # Text encoder lora
            if self.train_config.text_lora_r > 0:
                lora_config = LoraConfig(
                    r=self.train_config.text_lora_r,
                    lora_alpha=self.train_config.text_lora_alpha,
                    init_lora_weights="gaussian",
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                )
                self.sd_pipeline.text_encoder.add_adapter(lora_config)

                # To avoid accelerate unscaling problems in FP16.
                for param in self.sd_pipeline.text_encoder.parameters():
                    # only upcast trainable parameters (LoRA) into fp32
                    if param.requires_grad:
                        param.data = param.to(torch.float32)
                        trainable_params.append(param)

            return trainable_params
        else:
            return self.sd_pipeline.unet

    def save_checkpoint(self, models, weights, output_dir):
        # if len(models) != 1:
        #     raise ValueError("Given how the trainable params were set, this should be of length 1")

        if self.use_lora:
            unet_lora = None
            text_lora = None
            if hasattr(models[0], "peft_config") and getattr(models[0], "peft_config", None) is not None:
                unet_lora = convert_state_dict_to_diffusers(get_peft_model_state_dict(models[0]))
            if hasattr(models[1], "peft_config") and getattr(models[1], "peft_config", None) is not None:
                text_lora = convert_state_dict_to_diffusers(get_peft_model_state_dict(models[1]))
            
            self.sd_pipeline.save_lora_weights(save_directory=output_dir, unet_lora_layers=unet_lora, text_encoder_lora_layers=text_lora)
        elif not self.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")

    def load_checkpoint(self, models, input_dir):
        if len(models) != 1:
            raise ValueError("Given how the trainable params were set, this should be of length 1")
        if self.use_lora:
            lora_state_dict, network_alphas = self.sd_pipeline.lora_state_dict(
                input_dir, weight_name="pytorch_lora_weights.safetensors"
            )
            self.sd_pipeline.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=models[0])

        elif not self.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")