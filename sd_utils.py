"""Utils for loading StableDiffusion models."""

import time
import torch
from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image, DiffusionPipeline, EulerDiscreteScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler


_SUPPORTED_MODELS = [
    '1.4', '2', '2.1', 'xl', 'xl_turbo', '3'
    ]


_GENERATION_CONFIGS = {
    '1.4': {
        }, 
    
    '2': {
        },
    
    '2.1': {
        }, 
    
    'xl': {
        'num_inference_steps': 50,
        'high_noise_frac': 0.8,
        },
    
    'xl_turbo': {
        'num_inference_steps': 4,
        'guidance_scale': 0.0,
        },
    
    '3': {
        'negative_prompt': '',
        'num_inference_steps': 28,
        'guidance_scale': 7.0,
        }, 

}


class StableDiffusionXLPipelineWrapper(torch.nn.Module):
    def __init__(self, base, refiner):
        super().__init__()
        
        self.unet = base.unet
        self._base = base
        self._refiner = refiner

    def __getattr__(self, name):
        return getattr(self._base, name, None)
    
    def __call__(self, prompt, num_inference_steps=50, high_noise_frac=0.8):
        image = self._base(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images

        out = self._refiner(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            denoising_start=high_noise_frac,
            image=image)
        return out

    def enable_model_cpu_offload(self):
        self._base.enable_model_cpu_offload()
        self._refiner.enable_model_cpu_offload()


def get_sd_pipe(model_name='2.1',
                device=torch.device('cuda:0'),
                torch_dtype=torch.float16,
                enable_model_cpu_offload=False):
    
    variant = "fp16" if str(torch_dtype) == 'torch.float16' else "fp32"
    
    if model_name == '1.4':
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch_dtype)

    elif model_name == '2':
        model_id = "stabilityai/stable-diffusion-2"
        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id,
            subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch_dtype)

    elif model_name == '2.1':
        model_id = "stabilityai/stable-diffusion-2-1"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    elif model_name == 'xl':
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        refiner_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch_dtype,
            use_safetensors=True,
            variant=variant)
        
        # base = DiffusionPipeline.from_pretrained(
        #     model_id, 
        #     torch_dtype=torch_dtype,
        #     variant=variant,
        #     use_safetensors=True
        #     )
        
        # refiner = DiffusionPipeline.from_pretrained(
        #     refiner_id,
        #     text_encoder_2=base.text_encoder_2,
        #     vae=base.vae,
        #     torch_dtype=torch_dtype,
        #     use_safetensors=True,
        #     variant=variant,
        # )
        # pipe = StableDiffusionXLPipelineWrapper(base, refiner)
        
    elif model_name == 'xl_turbo':
        pipe = StableDiffusionXLPipeline.from_single_file(
            "https://huggingface.co/stabilityai/sdxl-turbo/blob/main/sd_xl_turbo_1.0_fp16.safetensors",
            torch_dtype=torch_dtype,
            variant=variant)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing")

        # pipe = AutoPipelineForText2Image.from_pretrained(
        #     "stabilityai/sdxl-turbo",
        #     torch_dtype=torch_dtype,
        #     variant=variant)

    elif model_name == '3':
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.bfloat16,
            token='',  # TODO: need to put your own token
            )

    else:
        raise ValueError(f'Invalid model variant {model_name}')

    
    if enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    return pipe


if __name__ == '__main__':
    for model_name in _SUPPORTED_MODELS:
        enable_model_cpu_offload = False
        if model_name == '3':
            enable_model_cpu_offload = True
        pipe = get_sd_pipe(model_name, enable_model_cpu_offload=enable_model_cpu_offload)

        prompt = "a photo of an astronaut riding a horse on mars"
        try:
            start_time = time.time()
            image = pipe(prompt=prompt, **_GENERATION_CONFIGS[model_name]).images[0]
            end_time = time.time()
            print(f'Generated an image using SD-{model_name} | took {end_time - start_time:.4f} s')
        except:
            print(F'Error SD-{model_name}')
        del pipe

