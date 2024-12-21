"""Script for generating images using SD and saving mean activations."""
import os
import pickle
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DPMSolverMultistepScheduler
from hspace_helpers import PipelineWithHspace


def get_sd_pipe(model_name='2.1', device=torch.device('cuda:0'), torch_dtype=torch.float16):
    if model_name == '1.4':
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch_dtype)

    elif model_name == '2':
        model_id = "stabilityai/stable-diffusion-2"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch_dtype)

    elif model_name == '2.1':
        model_id = "stabilityai/stable-diffusion-2-1"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    else:
        raise ValueError

    pipe = pipe.to(device)
    return pipe


def main():
    file_name = f'cat_dog_sd2_1__150_199.pickle'
    model_name = '2.1'
    device = torch.device('cuda:0')
    torch_dtype = torch.float16
    prompt = "A photo of a black cat with a white dog"
    # h_space_layer_names = ['down_zero', 'down_one', 'down_two', 'down_three', 'mid']
    h_space_layer_names = ['down_zero', 'down_one', 'mid']
    num_reps = 50
    start_seed = 150
    
    root_dir = '/home/royhirsch_google_com/image_editing'
    out_path = os.path.join(root_dir, file_name)
    pipe = get_sd_pipe(model_name, device, torch_dtype)
    pipe_h = PipelineWithHspace(pipe, h_space_layer_names)

    results = []
    for i in range(start_seed, start_seed + num_reps):
        print(f'## number {i} ###')
        generator = torch.Generator(device=device).manual_seed(i)
        out = pipe_h(prompt, generator=generator)
        results.append({
            'image': np.array(out.images[0]),
            'h_space': out.h_space,
            'seed': i
            })

    with open(out_path, 'wb') as f:
        pickle.dump(
            {'prompt': prompt,
             'model_name': model_name,
             'num_reps': num_reps,
             'results': results},
            f)

if  __name__ == '__main__':
    main()
        