# Enhancing Diffusion Models with Text-Encoder Reinforcement Learning

Official PyTorch codes for paper [Enhancing Diffusion Models with Text-Encoder Reinforcement Learning](https://arxiv.org/abs/2311.15657)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2311.15657)
<a href="https://colab.research.google.com/drive/1pC6lzFR4hohwWyBgnkeZUghi2jqjl9gI?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> 
[![huggingface](https://img.shields.io/badge/HuggingFace-model-red.svg)](https://huggingface.co/chaofengc/sd-turbo_texforce)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=chaofengc/TexForce)

![teaser_img](./assets/fig_teaser.jpg)

## Requirements & Installation

- Clone the repo and install required packages with 
```
# git clone this repository
git clone https://github.com/chaofengc/TexForce.git
cd TexForce 

# create new anaconda env
conda create -n texforce python=3.8
source activate texforce 

# install python dependencies
pip3 install -r requirements.txt
```

## Results on SDXL-Turbo

We also applied our method to the recent model [sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo). The model is trained with [ImageReward](https://github.com/THUDM/ImageReward) feedback through direct back-propagation to save training time. Test with the following codes

```
## Note: sdturboxl requires latest diffusers installed from source with the following command
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```
```
from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to("cuda")
pipe.load_lora_weights('chaofengc/sdxl-turbo_texforce')

pt = ['a photo of a cat.']
img = pipe(prompt=pt, num_inference_steps=1, guidance_scale=0.0).images[0]
```

Here are some example results:

<table>
<thead>
  <tr>
    <th width="50%">sdxl-turbo</th>
    <th width="50%">sdxl-turbo + TexForce</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td colspan="2">
      A photo of a cat.
    </td>
  </tr>
  <tr>
    <td colspan="2">
        <img src='assets/image_sdxlturbo_001.jpg'>
    </td>
  </tr>
  <tr>
    <td colspan="2">
      An astronaut riding a horse.
    </td>
  </tr>
  <tr>
    <td colspan="2">
        <img src='assets/image_sdxlturbo_002.jpg'>
    </td>
  </tr>
  <tr>
    <td colspan="2">
      water bottle.
    </td>
  </tr>
  <tr>
    <td colspan="2">
        <img src='assets/image_sdxlturbo_003.jpg'>
    </td>
  </tr>
</tbody>
</table>


## Results on SD-Turbo

We applied our method to the recent model [sdturbo](https://huggingface.co/stabilityai/sd-turbo). The model is trained with [Q-Instruct](https://github.com/Q-Future/Q-Instruct) feedback through direct back-propagation to save training time. Test with the following codes

```
## Note: sdturbo requires latest diffusers>=0.24.0 with AutoPipelineForText2Image class

from diffusers import AutoPipelineForText2Image
from peft import PeftModel
import torch

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to("cuda")
PeftModel.from_pretrained(pipe.text_encoder, 'chaofengc/sd-turbo_texforce')

pt = ['a photo of a cat.']
img = pipe(prompt=pt, num_inference_steps=1, guidance_scale=0.0).images[0]
```

Here are some example results:

<table>
<thead>
  <tr>
    <th width="50%">sd-turbo</th>
    <th width="50%">sd-turbo + TexForce</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td colspan="2">
      A photo of a cat.
    </td>
  </tr>
  <tr>
    <td>
        <img src='assets/image_001_sdturbo.jpg'>
    </td>
    <td>
        <img src='assets/image_001_sdturbo_texforce.jpg'>
    </td>
  </tr>
  <tr>
    <td colspan="2">
      A photo of a dog.
    </td>
  </tr>
  <tr>
    <td>
        <img src='assets/image_002_sdturbo.jpg'>
    </td>
    <td>
        <img src='assets/image_002_sdturbo_texforce.jpg'>
    </td>
  </tr>
  <tr>
    <td colspan="2">
      A photo of a boy, colorful.
    </td>
  </tr>
  <tr>
    <td>
        <img src='assets/image_003_sdturbo.jpg'>
    </td>
    <td>
        <img src='assets/image_003_sdturbo_texforce.jpg'>
    </td>
  </tr>

</tbody>
</table>

## Results on SD-1.4, SD-1.5, SD-2.1

Due to code compatibility, you need to install the following diffusers first:
```
pip uninstall diffusers
pip install diffusers==0.16.0
```

You may simply load the pretrained lora weights with the following code block to improve performance of original stable diffusion model:
```
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler 
from peft import PeftModel
import torch

def load_model_weights(pipe, weight_path, model_type):
    if model_type == 'text+lora':
        text_encoder = pipe.text_encoder
        PeftModel.from_pretrained(text_encoder, weight_path)
    elif model_type == 'unet+lora':
        pipe.unet.load_attn_procs(weight_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

load_model_weights(pipe, './lora_weights/sd14_refl/', 'unet+lora')
load_model_weights(pipe, './lora_weights/sd14_texforce/', 'text+lora')

prompt = ['a painting of a dog.']
img = pipe(prompt).images[0]

```

Here are some example results:

<table>
<thead>
  <tr>
    <th width="25%">SDv1.4</th>
    <th width="25%">ReFL</th>
    <th width="25%">TexForce</th>
    <th width="25%">ReFL+TexForce</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td colspan="4">
      astronaut drifting afloat in space, in the darkness away from anyone else, alone, black background dotted with stars, realistic
    </td>
  </tr>
  <tr>
    <td colspan="4">
        <img src='assets/image_0021.jpg'>
    </td>
  </tr>
  <tr>
    <td colspan="4">
      portrait of a cute cyberpunk cat, realistic, professional
    </td>
  </tr>
  <tr>
    <td colspan="4">
        <img src='assets/image_0058.jpg'>
    </td>
  </tr>
  <tr>
    <td colspan="4">
      a coffee mug made of cardboard
    </td>
  </tr>
  <tr>
    <td colspan="4">
        <img src='assets/image_0099.jpg'>
    </td>
  </tr>
</tbody>
</table>


## Training

We rewrite the training codes based on [trl](https://github.com/huggingface/trl) with the latest diffusers library. 
> [!NOTE]  
> The latest diffusers support simple loading of lora weights with `pipeline.load_lora_weights` after training.

You may train the model with the following command:

### Example script for single prompt training
```
accelerate launch --num_processes 2 src/train_ddpo.py \
    --mixed_precision="fp16" \
    --sample_num_steps 50 --train_timestep_fraction 0.5 \
    --num_epochs 40 \
    --sample_batch_size 4 --sample_num_batches_per_epoch 64 \
    --train_batch_size 4 --train_gradient_accumulation_steps 1 \
    --prompt="single" --single_prompt_type="hand" --reward_list="handdetreward" \
    --per_prompt_stat_tracking=True \
    --tracker_project_name="texforce_hand" \
    --log_with="tensorboard"
```
The supported prompts and reward functions are listed below:
- prompts: `hand`, `face`, `color`, `count`, `comp`, `location`
- rewards: `handdetreward`, `topiq_nr-face`, `imagereward` 

### Example script for complex multi-prompt training
```
accelerate launch --num_processes 2 src/train_ddpo.py \
    --mixed_precision="fp16" \
    --sample_num_steps 50 --train_timestep_fraction 0.5 \
    --num_epochs 50 \
    --sample_batch_size 4 --sample_num_batches_per_epoch 128 \
    --train_batch_size 4 --train_gradient_accumulation_steps 4 \
    --prompt="imagereward" --reward_list="imagereward" \
    --per_prompt_stat_tracking=True \
    --tracker_project_name="texforce_imgreward" \
    --log_with="tensorboard"
``` 
The supported prompts and reward functions are:
- prompts: `imagereward`, `hps`
- rewards: `imagereward`, `hpsreward`, `laion_aes`

## Citation

If you find this code useful for your research, please cite our paper:
```
@inproceedings{chen2024texforce,
  title={Enhancing Diffusion Models with Text-Encoder Reinforcement Learning},
  author={Chaofeng Chen and Annan Wang and Haoning Wu and Liang Liao and Wenxiu Sun and Qiong Yan and Weisi Lin},
  year={2024},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
}
```

## License

This work is licensed under [NTU S-Lab License 1.0](./LICENCE_S-Lab) and a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>


## Acknowledgement

This project is largely based on [trl](https://github.com/huggingface/trl). The hand detection codes are taken from [Unified-Gesture-and-Fingertip-Detection](https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection). Many thanks to their great work :hugs:!
