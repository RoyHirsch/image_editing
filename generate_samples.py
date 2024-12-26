"""Script for generating images using SD and saving mean activations."""

import os
import pickle
import json
import random
import numpy as np
import torch
from sd_utils import get_sd_pipe, _GENERATION_CONFIGS
from hspace_helpers import get_pipeline_with_h_space


def main():
    qa_json_path = '/home/royhirsch_google_com/image_editing/sample_qa.json'
    with open(qa_json_path, 'r') as f:
      qa = json.load(f)
    
    sample = qa[-1]
    file_name = sample['id']
    prompt = sample['prompt']
    model_name = 'xl_turbo'
    device = torch.device('cuda:0')
    torch_dtype = torch.float16

    # h_space_layer_names = ['down_zero', 'down_one', 'down_two', 'down_three', 'mid']
    h_space_layer_names = ['down_zero']
    num_reps_per_file = 25
    num_files = 4
    
    root_dir = os.path.join('/home/royhirsch_google_com/image_editing/files', file_name)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    pipe = get_sd_pipe(model_name, device, torch_dtype)
    pipe_h = get_pipeline_with_h_space(model_name, pipe, h_space_layer_names)
    config = _GENERATION_CONFIGS[model_name]

    print(f'######## {file_name} ########')

    for iter in range(num_files):
        print(f'#### file {iter} ####')
        out_path = os.path.join(root_dir, file_name + f'_SD{model_name}_{iter}.pickle')
        results = []
        
        for i in range(num_reps_per_file):
            seed = random.randint(0, 1e6)
            print(f'# seed {seed} ({i}/{num_reps_per_file}) #')
            generator = torch.Generator(device=device).manual_seed(seed)
            out = pipe_h(prompt, generator=generator, **config)
            results.append({
                'image': np.array(out.images[0]),
                'h_space': out.h_space,
                'seed': seed
                })
            
        with open(out_path, 'wb') as f:
            pickle.dump(
                {
                    'prompt': prompt,
                    'id': file_name,
                    'model_name': model_name,
                    'num_reps': num_reps_per_file,
                    'results': results},
                f)

if  __name__ == '__main__':
    main()
        