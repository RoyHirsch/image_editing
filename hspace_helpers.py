import torch
import numpy as np
import PIL.Image

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from diffusers.utils import BaseOutput
from diffusers.image_processor import PipelineImageInput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps, rescale_noise_cfg
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput


@dataclass
class StableDiffusionWithHSpacePipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]
    h_space: Optional[Union[List[torch.tensor], Dict[str, torch.tensor]]]
    nsfw_content_detected: Optional[List[bool]]


class PipelineWithHspace():
    def __init__(self, pipe, h_space_layer_names=['down_zero', 'down_one', 'down_two', 'down_three', 'mid']):
        self.pipe = pipe
        self.h_space_layer_names = h_space_layer_names
        
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor
        width = width or self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.pipe.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self.pipe._guidance_scale = guidance_scale
        self.pipe._guidance_rescale = guidance_rescale
        self.pipe._clip_skip = clip_skip
        self.pipe._cross_attention_kwargs = cross_attention_kwargs
        self.pipe._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.pipe._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.pipe.cross_attention_kwargs.get("scale", None) if self.pipe.cross_attention_kwargs is not None else None
        )
        with torch.no_grad():

            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                self.pipe.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=lora_scale,
                clip_skip=self.pipe.clip_skip,
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if self.pipe.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                image_embeds = self.pipe.prepare_ip_adapter_image_embeds(
                    ip_adapter_image,
                    ip_adapter_image_embeds,
                    device,
                    batch_size * num_images_per_prompt,
                    self.pipe.do_classifier_free_guidance,
                )

            # 4. Prepare timesteps
            timesteps, num_inference_steps = retrieve_timesteps(
                self.pipe.scheduler, num_inference_steps, device, timesteps, sigmas
            )

            # 5. Prepare latent variables
            num_channels_latents = self.pipe.unet.config.in_channels
            latents = self.pipe.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)

            # 6.1 Add image embeds for IP-Adapter
            added_cond_kwargs = (
                {"image_embeds": image_embeds}
                if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
                else None
            )

            # 6.2 Optionally get Guidance Scale Embedding
            timestep_cond = None
            if self.pipe.unet.config.time_cond_proj_dim is not None:
                guidance_scale_tensor = torch.tensor(self.pipe.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
                timestep_cond = self.pipe.get_guidance_scale_embedding(
                    guidance_scale_tensor, embedding_dim=self.pipe.unet.config.time_cond_proj_dim
                ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        down_zero_h_space = []
        def get_down_zero_h_space(module, input, output):
            down_zero_h_space.append(output[0].clone().detach().cpu())
        down_zero_block_hock = self.pipe.unet.down_blocks[0].register_forward_hook(get_down_zero_h_space)

        down_one_h_space = []
        def get_down_one_h_space(module, input, output):
            down_one_h_space.append(output[0].clone().detach().cpu())
        down_one_block_hock = self.pipe.unet.down_blocks[1].register_forward_hook(get_down_one_h_space)

        down_two_h_space = []
        def get_down_two_h_space(module, input, output):
            down_two_h_space.append(output[0].clone().detach().cpu())
        down_two_block_hock = self.pipe.unet.down_blocks[2].register_forward_hook(get_down_two_h_space)

        down_three_h_space = []
        def get_down_three_h_space(module, input, output):
            down_three_h_space.append(output[0].clone().detach().cpu())
        down_three_block_hock = self.pipe.unet.down_blocks[3].register_forward_hook(get_down_three_h_space)

        mid_h_space = []
        def get_mid_h_space(module, input, output):
            mid_h_space.append(output.clone().detach().cpu())
        mid_block_hock = self.pipe.unet.mid_block.register_forward_hook(get_mid_h_space)
        
        with torch.no_grad():
            num_warmup_steps = len(timesteps) - num_inference_steps * self.pipe.scheduler.order
            self.pipe._num_timesteps = len(timesteps)
            with self.pipe.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.pipe.interrupt:
                        continue

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.pipe.do_classifier_free_guidance else latents
                    latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = self.pipe.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.pipe.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.pipe.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if self.pipe.do_classifier_free_guidance and self.pipe.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.pipe.guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.pipe.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.pipe.scheduler, "order", 1)
                            callback(step_idx, t, latents)

        with torch.no_grad():
            if not output_type == "latent":
                image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False, generator=generator)[
                    0
                ]
                image, has_nsfw_concept = self.pipe.run_safety_checker(image, device, prompt_embeds.dtype)
            else:
                image = latents
                has_nsfw_concept = None

            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
            image = self.pipe.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

            # Offload all models
            self.pipe.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        down_zero_block_hock.remove()
        down_one_block_hock.remove()
        down_two_block_hock.remove()
        down_three_block_hock.remove
        mid_block_hock.remove()
        
        h_space = {'down_zero': down_zero_h_space,
                   'down_one': down_one_h_space,
                   'down_two': down_two_h_space,
                   'down_three': down_three_h_space,
                   'mid': mid_h_space}
        
        h_space_keep = {}
        for layer_name  in self.h_space_layer_names:
            h_space_keep[layer_name] = h_space[layer_name]

        return StableDiffusionWithHSpacePipelineOutput(
            images=image, 
            h_space=h_space_keep,
            nsfw_content_detected=has_nsfw_concept)