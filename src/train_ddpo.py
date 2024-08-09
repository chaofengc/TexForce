# Copyright 2023 metric-space, The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from collections import defaultdict

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import textwrap

from dataclasses import dataclass, field

from transformers import HfArgumentParser

from trl import DDPOConfig
from sd_pipeline import ExtendDDPOStableDiffusionPipeline

from ddpo_trainer import DDPOTrainer

from prompt import ImagenetAnimalPrompts, SinglePrompt, ImageRewardPrompt, HPSPrompt
from rewards import MultiReward


@dataclass
class ScriptArguments:
    pretrained_model: str = field(
        default="runwayml/stable-diffusion-v1-5", metadata={"help": "the pretrained model to use"}
    )
    pretrained_revision: str = field(default="main", metadata={"help": "the pretrained model revision to use"})
    hf_hub_model_id: str = field(
        default="ddpo-finetuned-stable-diffusion", metadata={"help": "HuggingFace repo to save model weights to"}
    )
    hf_hub_aesthetic_model_id: str = field(
        default="trl-lib/ddpo-aesthetic-predictor",
        metadata={"help": "HuggingFace model ID for aesthetic scorer model weights"},
    )
    hf_hub_aesthetic_model_filename: str = field(
        default="aesthetic-model.pth",
        metadata={"help": "HuggingFace model filename for aesthetic scorer model weights"},
    )
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})
    unet_lora_r: int = field(default=0, metadata={"help": "LoRA rank for the UNet."})
    unet_lora_alpha: float = field(default=16, metadata={"help": "LoRA alpha for the UNet."})
    text_lora_r: int = field(default=16, metadata={"help": "LoRA rank for the UNet."})
    text_lora_alpha: float = field(default=16, metadata={"help": "LoRA alpha for the UNet."})

    prompt: str = field(
        default="imagereward", metadata={"help": "prompt dataset"}
    )
    single_prompt_type: str = field(
        default=None, metadata={"help": "prompt dataset"}
    )
    reward_list: str = field(
        default="imagereward", metadata={"help": "prompt dataset"}
    )


def prompt_fn(args):
    if args.prompt == 'animal':
        prompts = ImagenetAnimalPrompts() 
    elif args.prompt == 'single':
        prompts = SinglePrompt(args.single_prompt_type)
    elif args.prompt == 'imagereward':
        prompts = ImageRewardPrompt()
    elif args.prompt == 'hps':
        prompts = HPSPrompt(phase='train')
    
    def fn(index=None):
        return prompts[index], None
    
    return fn 


def reward_fn(args):
    reward_model = MultiReward(args.reward_list)
    return reward_model 


def write_text_on_image_tensor(image_tensor, text, position=(10, 10), font_size=20, color=(255, 0, 0)):
    # Convert the PyTorch tensor to a PIL image
    transform_to_pil = transforms.ToPILImage()
    img = transform_to_pil(image_tensor)

    txt_blk_height = 100
    new_img = Image.new("RGB", (img.width, img.height + txt_blk_height), "white")
    new_img.paste(img, (0, 0))
    
    # Draw text on the image
    draw = ImageDraw.Draw(new_img)
    font = ImageFont.load_default(font_size)

    # Wrap the text
    wrapped_text = textwrap.fill(text, width=50)

    # Calculate the position for the text
    text_x = 10
    text_y = img.height 

    # Add text to the image line by line
    for line in wrapped_text.split('\n'):
        draw.text((text_x, text_y), line, font=font, fill=color)
        text_y += 20 

    # draw.text((10, 10 + img.height), text, font=font, fill=color)
    
    # Convert the PIL image back to a PyTorch tensor
    transform_to_tensor = transforms.ToTensor()
    image_tensor_with_text = transform_to_tensor(new_img)
    
    return image_tensor_with_text


def image_outputs_logger(image_data, global_step, accelerate_logger):
    # For the sake of this example, we will only log the last batch of images
    # and associated data
    result = defaultdict(list) 
    images, prompts, _, rewards, _ = image_data[-1]

    for i, image in enumerate(images):
        prompt = prompts[i]
        reward = rewards[i].item()

        image = write_text_on_image_tensor(image, f'{reward:.2f} | {prompt}')

        # result[f"{prompt:.25} | {reward:.2f}"] = image.unsqueeze(0).float()
        if 'images' in result:
            result['images'] = torch.cat([result['images'], image.unsqueeze(0).float()], dim=0)
            # result['prompts'].append(prompt)
        else:
            result['images'] = image.unsqueeze(0).float()

    accelerate_logger.log_images(
        result,
        step=global_step,
    )

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, DDPOConfig))
    args, ddpo_config = parser.parse_args_into_dataclasses()
    ddpo_config.project_kwargs = {
        "logging_dir": "./logs",
        "automatic_checkpoint_naming": False,
        "total_limit": 5,
        "project_dir": "./save",
    }

    pipeline = ExtendDDPOStableDiffusionPipeline(
        args.pretrained_model, pretrained_model_revision=args.pretrained_revision, train_config=args
    )

    trainer = DDPOTrainer(
        ddpo_config,
        reward_fn(args),
        prompt_fn(args),
        pipeline,
        image_samples_hook=image_outputs_logger,
        train_config = args
    )

    trainer.train()
