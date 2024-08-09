import random
import json
import re

from torch.utils.data import Dataset


class SinglePrompt(Dataset):
    def __init__(self, type='', prefix='', postfix='', num=10000, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if type == 'color':
            self.prompt_texts = ['a green colored rabbit'] * num 
        elif type == 'face':
            self.prompt_texts = [
                'a complete face of a man', 
                'a complete face of a woman', 
            ] * num
        elif type == 'count':
            self.prompt_texts = ['Four wolves in the forest'] * num
        elif type == 'comp':
            self.prompt_texts = ['A cat and a dog'] * num
        elif type == 'location':
            self.prompt_texts = ['A dog on the moon'] * num
        elif type == 'hand':
            self.prompt_texts = ['A photo of a hand'] * num

        self.num = num
        
    def __getitem__(self, index):
        return random.choice(self.prompt_texts)
    
    def __len__(self):
        'Denotes the total number of samples'
        return self.num


class ImageRewardPrompt(Dataset):
    def __init__(self, file_path='./prompt_dataset/refl_data.json', test_file_path='./prompt_dataset/imagereward_benchmark.json', phase='train', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase = phase
        if phase == 'train':
            self.prompt_texts = []
            with open(file_path) as json_file:
                data = json.load(open(file_path))
                for p in data:
                    self.prompt_texts.append(p['text'])
            self.num = len(self.prompt_texts)
        elif phase == 'test':
            self.prompt_texts = []
            with open(test_file_path) as json_file:
                data = json.load(open(test_file_path))
                for p in data:
                    self.prompt_texts.append(p['prompt'])
            self.num = len(self.prompt_texts)
        
    def __getitem__(self, index):
        if self.phase == 'train': 
            return random.choice(self.prompt_texts)
        else:
            return self.prompt_texts[index]
    
    def __len__(self):
        'Denotes the total number of samples'
        return self.num


class HPSPrompt(Dataset):
    def __init__(self, train_file_path='./prompt_dataset/hps_v2_all.txt', val_file_path='./prompt_dataset/hps_v2_all_eval.txt', phase='train', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.phase = phase

        if phase == 'train':
            self.prompt_texts = [i.strip() for i in open(train_file_path).readlines()] * 1000
        elif phase == 'test':
            self.prompt_texts = [i.strip() for i in open(val_file_path).readlines()] 

        self.num = len(self.prompt_texts)
        
    def __getitem__(self, index):
        if self.phase == 'train': 
            return random.choice(self.prompt_texts)
        else:
            return self.prompt_texts[index]
    
    def __len__(self):
        'Denotes the total number of samples'
        return self.num


animals_train_45 = [
    "cat", "dog", "horse", "monkey", "rabbit", "zebra", "spider", "bird",
    "sheep", "deer", "cow", "goat", "lion", "tiger", "bear", "raccoon",
    "fox", "wolf", "lizard", "beetle", "ant", "butterfly", "fish", "shark",
    "whale", "dolphin", "squirrel", "mouse", "rat", "snake", "turtle",
    "frog", "chicken", "duck", "goose", "bee", "pig", "turkey", "fly",
    "llama", "camel", "bat", "gorilla", "hedgehog", "kangaroo"
]

animals_eval = [
    'cheetach', 'elephant', 'girraffe', 'hippo',
    'jellyfish', 'panda', 'penguin', 'swan'
]

activities = [
    "washing the dishes", "riding a bike", "playing chess",
    "playing the piano",  
]

class ImagenetAnimalPrompts(Dataset):
    """
    Pipeline of prompts consisting of animals from ImageNet, as used in the original `DDPO paper <https://arxiv.org/abs/2305.13301>`_.
    """
    def __init__(self, prefix='A photo of a', postfix='', use_act=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.animal_classes = animals_train_45 * 100
        self.use_act = use_act
        self.prefix = prefix

        self.num = len(self.animal_classes)

    def __getitem__(self, index):
        if not self.use_act:
            return f'{self.prefix} {self.animal_classes[index]}'
        else:
            animal = random.choice(self.animal_classes)
            return f'a {animal} is {random.choice(activities)}'
    
    def __len__(self):
        'Denotes the total number of samples'
        return self.num