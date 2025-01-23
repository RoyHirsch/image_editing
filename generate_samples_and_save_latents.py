"""Script for generating images using SD3 and saving mean activations."""

import sys
sys.path.append('/home/royhirsch_google_com/t2v_metrics')
import t2v_metrics
from PIL import Image
import os
import pickle
import json
import random
import numpy as np
import torch
from sd_utils import get_sd_pipe, _GENERATION_CONFIGS
from hspace_helpers import get_pipeline_with_h_space


def main():

    file_name = 'fruits'
    prompt = 'A wooden bowl with 2 bannas and one orange'
    model_name = '3'
    device = torch.device('cuda:1')
    torch_dtype = torch.float16

    clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl') 
    # h_space_layer_names = ['down_zero', 'down_one', 'down_two', 'down_three', 'mid']
    h_space_layer_names = ['noise', 'prompt_embeds', 'pooled_prompt_embeds', 'block_1', 'block_5', 'block_10', 'block_20']
    num_reps = 100
    
    root_dir = os.path.join('/home/royhirsch_google_com/image_editing/files', file_name)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    pipe = get_sd_pipe(model_name, device, torch_dtype)
    pipe_h = get_pipeline_with_h_space('sd3', pipe, h_space_layer_names)
    config = _GENERATION_CONFIGS[model_name]

    print('#' * 80)
    print(f'Fileaname: {file_name}')
    print('#' * 80)

    for iter in range(num_reps):
        print(f'#### file {iter} ####')
        out_path = os.path.join(root_dir,  f'sd{model_name}_{iter}.pickle')

        seed = random.randint(0, 1e6)
        print(f'# seed {seed} ({iter}/{num_reps}) #')
        generator = torch.Generator(device=device).manual_seed(seed)
        out = pipe_h(prompt, generator=generator, **config)
        
        image = out.images[0]
        filename = f'image_{iter}.png'
        image_path = os.path.join(root_dir, filename)
        image.save(image_path, format='PNG')

        score = clip_flant5_score(images=[image_path], texts=[prompt]).flatten()[0].item()
        print(score)
        with open(out_path, 'wb') as f:
            pickle.dump(
                {
                    'prompt': prompt,
                    'id': file_name,
                    'model_name': model_name,
                    'image': np.array(out.images[0].resize((256, 256))),
                    'h_space': out.h_space,
                    'vqa_score': score,
                    'seed': seed
                },
                f)


if  __name__ == '__main__':
    main()
        