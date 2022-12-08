from diffusers.utils import deprecate, logging
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.configuration_utils import FrozenDict
from diffusers.pipeline_utils import DiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPVisionModelWithProjection
from packaging import version
from diffusers.utils import is_accelerate_available
import PIL
from typing import Callable, List, Optional, Union
import inspect
import modules.scripts as scripts
import gradio as gr
import os

from modules import images
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state


import json
import math
import os
import sys
import warnings

import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import random
import cv2
from skimage import exposure
from typing import Any, Dict, List, Optional

import modules.sd_hijack
from modules import devices, prompt_parser, masking, sd_samplers, lowvram, generation_parameters_copypaste
from modules.sd_hijack import model_hijack
from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.face_restoration
import modules.images as images
import modules.styles
import logging


from modules.processing import create_infotext, get_fixed_seed, StableDiffusionProcessing, decode_first_stage, apply_overlay, apply_color_correction

from modules.prompt_parser import ScheduledPromptConditioning, MulticondLearnedConditioning, ComposableScheduledPromptConditioning

import torch.nn.functional as F


def process_images_special(p: StableDiffusionProcessing, image_embedding) -> Processed:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    if type(p.prompt) == list:
        assert (len(p.prompt) > 0)
    else:
        assert p.prompt is not None

    devices.torch_gc()

    seed = get_fixed_seed(p.seed)
    subseed = get_fixed_seed(p.subseed)

    modules.sd_hijack.model_hijack.apply_circular(p.tiling)
    modules.sd_hijack.model_hijack.clear_comments()

    comments = {}

    if type(p.prompt) == list:
        p.all_prompts = [shared.prompt_styles.apply_styles_to_prompt(
            x, p.styles) for x in p.prompt]
    else:
        p.all_prompts = p.batch_size * p.n_iter * \
            [shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)]

    if type(p.negative_prompt) == list:
        p.all_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(
            x, p.styles) for x in p.negative_prompt]
    else:
        p.all_negative_prompts = p.batch_size * p.n_iter * \
            [shared.prompt_styles.apply_negative_styles_to_prompt(
                p.negative_prompt, p.styles)]

    if type(seed) == list:
        p.all_seeds = seed
    else:
        p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0)
                       for x in range(len(p.all_prompts))]

    if type(subseed) == list:
        p.all_subseeds = subseed
    else:
        p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]

    def infotext(iteration=0, position_in_batch=0):
        return create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments, iteration, position_in_batch)

    with open(os.path.join(shared.script_path, "params.txt"), "w", encoding="utf8") as file:
        processed = Processed(p, [], p.seed, "")
        file.write(processed.infotext(p, 0))

    if os.path.exists(cmd_opts.embeddings_dir) and not p.do_not_reload_embeddings:
        model_hijack.embedding_db.load_textual_inversion_embeddings()

    if p.scripts is not None:
        p.scripts.process(p)

    infotexts = []
    output_images = []

    with torch.no_grad(), p.sd_model.ema_scope():
        with devices.autocast():
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)

        if state.job_count == -1:
            state.job_count = p.n_iter

        for n in range(p.n_iter):
            if state.skipped:
                state.skipped = False

            if state.interrupted:
                break

            prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            negative_prompts = p.all_negative_prompts[n *
                                                      p.batch_size:(n + 1) * p.batch_size]
            seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
            subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

            if len(prompts) == 0:
                break

            if p.scripts is not None:
                p.scripts.process_batch(
                    p, batch_number=n, prompts=prompts, seeds=seeds, subseeds=subseeds)

            image_embedding = image_embedding.squeeze(1)
            image_embedding = image_embedding.cuda()
            #torch.nn.functional.pad
            image_embedding =  F.pad(input=image_embedding, pad=(0, 0,0,75), mode='constant', value=0)

            print("padded",image_embedding.shape)

            res = [
                [
                    ComposableScheduledPromptConditioning(
                        [
                        ScheduledPromptConditioning(
                            end_at_step=20, cond=image_embedding)
                        ]
                    )
                ]
            ]

            image_cond = MulticondLearnedConditioning(shape=(1,), batch=res)

            prompts = ["test"]

            with devices.autocast():
                uc = prompt_parser.get_learned_conditioning(
                    shared.sd_model, negative_prompts, p.steps)
                c = prompt_parser.get_multicond_learned_conditioning(
                    shared.sd_model, prompts, p.steps)
                #c = image_cond

            print("Why???", c, c.batch, c.batch[0][0].schedules, c.batch[0]
                  [0].schedules[0].cond.shape, image_embedding.shape)
            print("Why2???", image_cond, image_cond.batch,
                  image_cond.batch[0][0].schedules, image_cond.batch[0][0].schedules[0].cond.shape)

            # c.batch[0][0].schedules[0].cond=image_embedding
            c = image_cond

            if len(model_hijack.comments) > 0:
                for comment in model_hijack.comments:
                    comments[comment] = 1

            if p.n_iter > 1:
                shared.state.job = f"Batch {n+1} out of {p.n_iter}"

            with devices.autocast():
                samples_ddim = p.sample(conditioning=c, unconditional_conditioning=uc, seeds=seeds,
                                        subseeds=subseeds, subseed_strength=p.subseed_strength, prompts=prompts)

            x_samples_ddim = [decode_first_stage(p.sd_model, samples_ddim[i:i+1].to(
                dtype=devices.dtype_vae))[0].cpu() for i in range(samples_ddim.size(0))]
            x_samples_ddim = torch.stack(x_samples_ddim).float()
            x_samples_ddim = torch.clamp(
                (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            del samples_ddim

            if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
                lowvram.send_everything_to_cpu()

            devices.torch_gc()

            if opts.filter_nsfw:
                import modules.safety as safety
                x_samples_ddim = modules.safety.censor_batch(x_samples_ddim)

            for i, x_sample in enumerate(x_samples_ddim):
                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)

                if p.restore_faces:
                    if opts.save and not p.do_not_save_samples and opts.save_images_before_face_restoration:
                        images.save_image(Image.fromarray(x_sample), p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(
                            n, i), p=p, suffix="-before-face-restoration")

                    devices.torch_gc()

                    x_sample = modules.face_restoration.restore_faces(x_sample)
                    devices.torch_gc()

                image = Image.fromarray(x_sample)

                if p.color_corrections is not None and i < len(p.color_corrections):
                    if opts.save and not p.do_not_save_samples and opts.save_images_before_color_correction:
                        image_without_cc = apply_overlay(
                            image, p.paste_to, i, p.overlay_images)
                        images.save_image(image_without_cc, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(
                            n, i), p=p, suffix="-before-color-correction")
                    image = apply_color_correction(
                        p.color_corrections[i], image)

                image = apply_overlay(image, p.paste_to, i, p.overlay_images)

                if opts.samples_save and not p.do_not_save_samples:
                    images.save_image(
                        image, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p)

                text = infotext(n, i)
                infotexts.append(text)
                if opts.enable_pnginfo:
                    image.info["parameters"] = text
                output_images.append(image)

            del x_samples_ddim

            devices.torch_gc()

            state.nextjob()

        p.color_corrections = None

        index_of_first_image = 0
        unwanted_grid_because_of_img_count = len(
            output_images) < 2 and opts.grid_only_if_multiple
        if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
            grid = images.image_grid(output_images, p.batch_size)

            if opts.return_grid:
                text = infotext()
                infotexts.insert(0, text)
                if opts.enable_pnginfo:
                    grid.info["parameters"] = text
                output_images.insert(0, grid)
                index_of_first_image = 1

            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", p.all_seeds[0], p.all_prompts[0], opts.grid_format, info=infotext(
                ), short_filename=not opts.grid_extended_filename, p=p, grid=True)

    devices.torch_gc()

    res = Processed(p, output_images, p.all_seeds[0], infotext() + "".join(["\n\n" + x for x in comments]),
                    subseed=p.all_subseeds[0], index_of_first_image=index_of_first_image, infotexts=infotexts)

    if p.scripts is not None:
        p.scripts.postprocess(p, res)

    return res


# logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class StableDiffusionImageVariationPipeline_partial(DiffusionPipeline):
    r"""
    Pipeline to generate variations from an input image using Stable Diffusion.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder. Stable Diffusion Image Variation uses the vision portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModelWithProjection),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNet2DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        safety_checker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(
            unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely .If you're checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0",
                      deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (
            len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(
            requires_safety_checker=requires_safety_checker)

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError(
                "Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.image_encoder, self.vae, self.safety_checker]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def encode_image(self, image, device, num_images_per_prompt, do_classifier_free_guidance):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(
                images=image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)

        image = image.float()
        image = image.cpu()
        print("about to die", image.dtype, image.device)

        print("dying here?", self.image_encoder.device)

        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
        image_embeddings = image_embeddings.view(
            bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            uncond_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([uncond_embeddings, image_embeddings])

        return image_embeddings


class Script(scripts.Script):

    # The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):

        return "sd-image-variations"


# Determines when the script should be shown in the dropdown menu via the
# returned value. As an example:
# is_img2img is True if the current tab is img2img, and False if it is txt2img.
# Thus, return is_img2img to only show the script on the img2img tab.

    def show(self, is_img2img):

        return is_img2img

# How the script's is displayed in the UI. See https://gradio.app/docs/#components
# for the different UI components you can use and how to create them.
# Most UI components can return a value, such as a boolean for a checkbox.
# The returned values are passed to the run method as parameters.

    def ui(self, is_img2img):
        return []


# This is where the additional processing is implemented. The parameters include
# self, the model object "p" (a StableDiffusionProcessing class, see
# processing.py), and the parameters returned by the ui method.
# Custom functions can be defined here, and additional libraries can be imported
# to be used in processing. The return value should be a Processed object, which is
# what is returned by the process_images method.

    def run(self, p):

        # generate image embedding using

        # image=p.image
        image = p.init_images[0]
        device = 'cuda'
        num_images_per_prompt = 1
        do_classifier_free_guidance = True

        with devices.autocast():
            embed = StableDiffusionImageVariationPipeline_partial.from_pretrained(
                "lambdalabs/sd-image-variations-diffusers",
                #            revision="v2.0",
            )

            image_embeddings = embed.encode_image(
                image, device, num_images_per_prompt, do_classifier_free_guidance)

        proc = process_images_special(p, image_embeddings)

        return proc
