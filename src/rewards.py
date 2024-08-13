import os
import numpy as np 
from PIL import Image
import io
import numpy as np

import torch
from torch import nn
import torchvision.transforms as T
from transformers import AutoProcessor, AutoModel

import ImageReward as RM

from pyiqa import create_metric
from pyiqa.archs.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

import hpsv2
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

import sys
sys.path.append('./src/Fingertip/')
from hand_detector.detector import YOLO

from functools import wraps


def suppress_print(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Save the original standard output
        original_stdout = sys.stdout
        # Redirect standard output to a null device
        sys.stdout = open(os.devnull, 'w')
        try:
            return func(*args, **kwargs)
        finally:
            # Restore the original standard output
            sys.stdout.close()
            sys.stdout = original_stdout
    return wrapper


class jpeg_incompressibility(nn.Module):
    def forward(self, images):
        org_imgs = images
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]

        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]

        return torch.tensor(sizes).to(org_imgs.device)


class jpeg_compressibility(nn.Module):
    def forward(self, images):
        return -jpeg_incompressibility()(images)


class MultiReward(nn.Module):
    def __init__(self, reward_name_list = 'topiq_iaa+topiq_nr+clipscore'):
        """
        Args: 
            reward_name_list: str, a string of reward names separated by '+'. Multiple metrics will be averaged. Support reward list: [pickscore, imagereward, hpsreward, handdetreward, compress, incompress] and metrics in pyiqa such as [topiq_iaa, topiq_nr, topiq_nr-face, clipscore, ...]
        """
        super().__init__()

        reward_name_list = reward_name_list.split('+')

        self.reward_dict = nn.ModuleDict()
        for reward_name in reward_name_list:
            if reward_name == 'pickscore':
                self.reward_dict[reward_name] = PickScore()
            elif reward_name == 'imagereward':
                self.reward_dict[reward_name] = ImgReward()
            elif reward_name == 'hpsreward':
                self.reward_dict[reward_name] = HPSReward()
            elif reward_name == 'handdetreward':
                self.reward_dict[reward_name] = HandDetectReward()
            elif reward_name == 'compress':
                self.reward_dict[reward_name] = jpeg_compressibility()
            elif reward_name == 'incompress':
                self.reward_dict[reward_name] = jpeg_incompressibility()
            else:
                self.reward_dict[reward_name] = create_metric(reward_name, loss_reduction='none') 
        self.reward_name_list = reward_name_list

    @torch.no_grad()
    def forward(self, images, prompts, prompt_metadata=None, **kwargs):

        if prompts is not None:
            assert isinstance(prompts, (list, tuple)), f'prompts should be a list or tuple, but got {type(prompts)}, prompts are: {prompts}'

        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).permute(0, 3, 1, 2) / 255
        
        if hasattr(self, 'device'):
            images = images.to(self.device)
        
        score = 0
        for rname in self.reward_name_list:
            reward_func = self.reward_dict[rname]
            reward_func.eval()
            if rname == 'clipscore':
                tmp_score = reward_func(images, caption_list=prompts)
            elif rname in ['pickscore', 'imagereward', 'hpsreward']:
                tmp_score = reward_func(images, prompts, **kwargs)
            else:
                reward_func.dummy_param = reward_func.dummy_param.to(images.device)
                tmp_score = reward_func(images, **kwargs)
                if rname == 'topiq_iaa':
                    tmp_score = tmp_score / 10

            score += tmp_score.squeeze()

        return score, None


class PickScore(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

        self.processor = AutoProcessor.from_pretrained(processor_name_or_path)
        self.model = AutoModel.from_pretrained(model_pretrained_name_or_path)
        self.model.eval()
    
    def preprocess(self, x):
        # Bicubic interpolation
        x = T.functional.resize(x, (224, 224), interpolation=T.InterpolationMode.BICUBIC)
        x = T.functional.normalize(x, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
        return x

    @torch.no_grad()
    def forward(self, images, prompt, return_relative=False):
        device = images.device

        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        image_inputs = {'pixel_values': self.preprocess(images)}
        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        if not return_relative:
            scores = (text_embs * image_embs).sum(dim=-1)
        else:
            scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            # get probabilities if you have multiple images to choose from
            scores = torch.softmax(scores, dim=-1)
                
        return scores


class ImgReward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = RM.load("ImageReward-v1.0")
    
    def preprocess(self, x):
        # Bicubic interpolation
        x = T.functional.resize(x, (224, 224), interpolation=T.InterpolationMode.BICUBIC)
        x = T.functional.normalize(x, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
        return x

    @torch.no_grad()
    def forward(self, images, prompt, return_relative=False):

        self.device = images.device

        text_input = self.model.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
        
        # image encode
        image = self.preprocess(images)
        image_embeds = self.model.blip.visual_encoder(image)
            
        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)
        text_output = self.model.blip.text_encoder(text_input.input_ids,
                                                attention_mask = text_input.attention_mask,
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,
                                                return_dict = True,
                                            )
        txt_features = text_output.last_hidden_state[:,0,:]
            
        rewards = self.model.mlp(txt_features) # [image_num, 1]
        rewards = (rewards - self.model.mean) / self.model.std
        
        if return_relative:
            rewards = torch.softmax(rewards, dim=0)

        return rewards


class HPSReward(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        model_name = "ViT-H-14"
        self.model, preprocess_train, preprocess_val = create_model_and_transforms(
            model_name,
            'laion2B-s32B-b79K',
            precision=torch.float16,
            device='cpu',
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )    
    
        self.tokenizer = get_tokenizer(model_name)
    
        checkpoint_path = f"{os.path.expanduser('~')}/.cache/hpsv2/HPS_v2_compressed.pt"
        # force download of model via score
        hpsv2.score([], "")
    
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        self.target_size =  224
        self.normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
        
    def forward(self, im_pix, prompts, return_relative=False):    

        with torch.cuda.amp.autocast():
            x_var = T.Resize(self.target_size)(im_pix)
            x_var = self.normalize(x_var).to(im_pix.dtype)        
            caption = self.tokenizer(prompts)
            caption = caption.to(im_pix.device)
            outputs = self.model(x_var, caption)
            image_features, text_features = outputs["image_features"], outputs["text_features"]
            logits = image_features @ text_features.T
            if return_relative:
                scores = logits.softmax(dim=0)
            else:
                scores = torch.diagonal(logits)

            torch.nan_to_num_(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return scores 


class HandDetectReward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hand_yolo = YOLO('./src/Fingertip/weights/yolo.h5')
        
    @suppress_print
    def forward(self, x):
        scores = self.hand_yolo.detect_prob(x.cpu())
        scores = torch.tensor(scores).to(x)

        return scores 
