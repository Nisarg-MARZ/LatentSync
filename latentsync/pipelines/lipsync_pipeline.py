# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import math
import os
import shutil
from dataclasses import dataclass
from typing import Callable, List, Optional, Union, Literal
import subprocess

import numpy as np
import torch
import torchvision
from PIL.Image import Image
from torchvision import transforms

from packaging import version

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange
import cv2

from ..models.unet import UNet3DConditionModel
from ..utils.util import read_video, read_audio, write_video, check_ffmpeg_installed
from ..utils.image_processor import ImageProcessor, load_fixed_mask
from ..whisper.audio2feature import Audio2Feature
import tqdm
import soundfile as sf

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        denoising_unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(denoising_unet.config, "_diffusers_version") and version.parse(
            version.parse(denoising_unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = (
            hasattr(denoising_unet.config, "sample_size") and denoising_unet.config.sample_size < 64
        )
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(denoising_unet.config)
            new_config["sample_size"] = 64
            denoising_unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            denoising_unet=denoising_unet,
            scheduler=scheduler,
        )
        self.rolling_step_table = None
        self.latents = None

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.denoising_unet, "_hf_hook"):
            return self.device
        for module in self.denoising_unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        decoded_latents = self.vae.decode(latents).sample
        return decoded_latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            batch_size,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def encode_latents(
            self,
            images: torch.Tensor,  # (b, c, f, h, w) in [0, 1]
            dtype: torch.dtype,
            device: torch.device,
            generator: torch.Generator | None = None,
    ):
        """
        Encode RGB images into VAE latent space so that a later call to
        `decode_latents` reconstructs them loss-lessly (modulo VAE quantisation).

        Returns
        -------
        latents : torch.Tensor         # (b, c_latent, f, h//8, w//8)  —  already
                                       # scaled/shifted to match `decode_latents`.
        """
        # -----------------------------------------------------------------
        # 1.  Sanity & reshape  (b, c, f, h, w)  ->  (b*f, c, h, w)
        # -----------------------------------------------------------------
        b, _, f, _, _ = images.shape
        images = images.to(device=device, dtype=dtype)
        flat = rearrange(images, "b c f h w -> (b f) c h w")

        # -----------------------------------------------------------------
        # 2.  VAE encode  →  sample from the latent distribution
        # -----------------------------------------------------------------
        latent_dist = self.vae.encode(flat).latent_dist
        latents = latent_dist.sample(generator=generator)  # (b*f, c_latent, h/8, w/8)

        # -----------------------------------------------------------------
        # 3.  Apply the same scaling/shift convention as `prepare_mask_latents`
        #     (inverse of what `decode_latents` does).
        # -----------------------------------------------------------------
        # latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        latents = latents * self.vae.config.scaling_factor

        # -----------------------------------------------------------------
        # 4.  Restore original batch/frame layout
        # -----------------------------------------------------------------
        latents = rearrange(latents, "(b f) c h w -> b c f h w", b=b, f=f)
        latents = latents.to(device=device, dtype=dtype)

        return latents

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    def affine_transform_video(self, video_frames: np.ndarray):
        faces = []
        boxes = []
        affine_matrices = []
        print(f"Affine transforming {len(video_frames)} faces...")
        for frame in tqdm.tqdm(video_frames):
            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)

        faces = torch.stack(faces)
        return faces, boxes, affine_matrices

    def restore_video(self, faces: torch.Tensor, video_frames: np.ndarray, boxes: list, affine_matrices: list):
        video_frames = video_frames[: len(faces)]
        out_frames = []
        print(f"Restoring {len(faces)} faces...")
        for index, face in enumerate(tqdm.tqdm(faces)):
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)
            face = torchvision.transforms.functional.resize(
                face, size=(height, width), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
            )
            out_frame = self.image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
            out_frames.append(out_frame)
        return np.stack(out_frames, axis=0)

    def loop_video(self, whisper_chunks: list, video_frames: np.ndarray):
        # If the audio is longer than the video, we need to loop the video
        if len(whisper_chunks) > len(video_frames):
            faces, boxes, affine_matrices = self.affine_transform_video(video_frames)
            num_loops = math.ceil(len(whisper_chunks) / len(video_frames))
            loop_video_frames = []
            loop_faces = []
            loop_boxes = []
            loop_affine_matrices = []
            for i in range(num_loops):
                if i % 2 == 0:
                    loop_video_frames.append(video_frames)
                    loop_faces.append(faces)
                    loop_boxes += boxes
                    loop_affine_matrices += affine_matrices
                else:
                    loop_video_frames.append(video_frames[::-1])
                    loop_faces.append(faces.flip(0))
                    loop_boxes += boxes[::-1]
                    loop_affine_matrices += affine_matrices[::-1]

            video_frames = np.concatenate(loop_video_frames, axis=0)[: len(whisper_chunks)]
            faces = torch.cat(loop_faces, dim=0)[: len(whisper_chunks)]
            boxes = loop_boxes[: len(whisper_chunks)]
            affine_matrices = loop_affine_matrices[: len(whisper_chunks)]
        else:
            video_frames = video_frames[: len(whisper_chunks)]
            faces, boxes, affine_matrices = self.affine_transform_video(video_frames)

        return video_frames, faces, boxes, affine_matrices

    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        video_mask_path: str = None,
        num_frames: int = 20,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask_image_path: str = "latentsync/utils/mask.png",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        starting_timestep: Optional[int] = 500,
        **kwargs,
    ):
        is_train = self.denoising_unet.training
        self.denoising_unet.eval()

        check_ffmpeg_installed()

        # 0. Define call parameters
        batch_size = 1
        device = self._execution_device
        mask_image = load_fixed_mask(height, mask_image_path)
        self.image_processor = ImageProcessor(height, device="cuda", mask_image=mask_image)
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        # 1. Default height and width to unet
        height = height or self.denoising_unet.config.sample_size * self.vae_scale_factor
        width = width or self.denoising_unet.config.sample_size * self.vae_scale_factor

        # 2. Check inputs
        self.check_inputs(height, width, callback_steps)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 4. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

        audio_samples = read_audio(audio_path)
        video_frames = read_video(video_path, use_decord=False)

        video_frames, faces, boxes, affine_matrices = self.loop_video(whisper_chunks, video_frames)

        synced_video_frames = []

        num_channels_latents = self.vae.config.latent_channels

        num_inferences = len(whisper_chunks) - num_frames + 1

        for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):
            # 3. set timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            if self.denoising_unet.add_audio_layer:
                audio_embeds = torch.stack(whisper_chunks[i: i + num_frames])
                audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                if do_classifier_free_guidance:
                    null_audio_embeds = torch.zeros_like(audio_embeds)
                    audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
            else:
                audio_embeds = None
            inference_faces = faces[i: i + num_frames]
            if i == 0:
                # Prepare latent variables
                latents = self.prepare_latents(
                    batch_size,
                    num_frames,  # len(whisper_chunks),
                    num_channels_latents,
                    height,
                    width,
                    weight_dtype,
                    device,
                    generator,
                )

            ref_pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                inference_faces, affine_transform=False
            )

            # 7. Prepare mask latent variables
            mask_latents, masked_image_latents = self.prepare_mask_latents(
                masks,
                masked_pixel_values,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )

            # 8. Prepare image latents
            ref_latents = self.prepare_image_latents(
                ref_pixel_values,
                device,
                weight_dtype,
                generator,
                do_classifier_free_guidance,
            )

            # 9. Denoising loop
            if i == 0:
                logger.info("Doing initial sequence denoising...")
                num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
                with self.progress_bar(total=num_inference_steps) as progress_bar:
                    for j, t in enumerate(timesteps):
                        # expand the latents if we are doing classifier free guidance
                        denoising_unet_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                        denoising_unet_input = self.scheduler.scale_model_input(denoising_unet_input, t)

                        # concat latents, mask, masked_image_latents in the channel dimension
                        denoising_unet_input = torch.cat(
                            [denoising_unet_input, mask_latents, masked_image_latents, ref_latents], dim=1
                        )

                        t_tensor = torch.full(
                            (denoising_unet_input.shape[0], latents.shape[2]),  # [B, T]
                            t,
                            device=latents.device,
                            dtype=torch.long,
                        )

                        # predict the noise residual
                        noise_pred = self.denoising_unet(
                            denoising_unet_input, t_tensor, encoder_hidden_states=audio_embeds
                        ).sample

                        # perform guidance
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = step_per_frame(self.scheduler, latents, noise_pred, t_tensor)

                        # call the callback, if provided
                        if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                            progress_bar.update()
                            if callback is not None and j % callback_steps == 0:
                                callback(j, t, latents)

                scheduler = DDIMScheduler(num_train_timesteps=999)

                self.latents = init_rolling_latents_ddim(latents,
                                                         scheduler,
                                                         max_t=starting_timestep,
                                                         denoise_steps=num_inference_steps)
                # Recover the pixel values
                decoded_latents = self.decode_latents(latents[:, :, :1, ...])
                decoded_latents = self.paste_surrounding_pixels_back(
                    decoded_latents, ref_pixel_values[:1], 1 - masks[:1], device, weight_dtype
                )
                synced_video_frames.append(decoded_latents)

            else:
                # init_latent = self.encode_single_image(
                #     inference_faces[-1],
                #     device=device,
                #     dtype=weight_dtype
                # )
                last_infer = i == num_inferences - 1

                # Prepare timesteps
                self.scheduler.set_timesteps(num_inference_steps, device=device)
                latents = append_noised_latent_tail(
                    latents=self.latents,
                    scheduler=self.scheduler,
                    timestep=starting_timestep,
                    initial_latent=ref_latents[:, :, -1]
                )

                if self.rolling_step_table is None:
                    self.rolling_step_table = get_tedi_timestep_tensor_all(
                        window_size=num_frames,
                        max_denoise_step=num_inference_steps,
                        max_timestep=starting_timestep,
                        device=device
                    )

                latents = self.rolling_forward(
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    mask_latents=mask_latents,
                    masked_image_latents=masked_image_latents,
                    ref_latents=ref_latents,
                    audio_embeds=audio_embeds,
                    latents=latents,
                    last_infer=last_infer,
                    starting_timestep=starting_timestep
                )
                if not last_infer:
                    # Recover the pixel values
                    decoded_latents = self.decode_latents(latents[:, :, :1, ...])
                    decoded_latents = self.paste_surrounding_pixels_back(
                        decoded_latents, ref_pixel_values[:1], 1 - masks[:1], device, weight_dtype
                    )
                else:
                    decoded_latents = self.decode_latents(latents)
                    decoded_latents = self.paste_surrounding_pixels_back(
                        decoded_latents, ref_pixel_values, 1 - masks, device, weight_dtype
                    )

                synced_video_frames.append(decoded_latents)

        synced_video_frames = self.restore_video(torch.cat(synced_video_frames), video_frames, boxes, affine_matrices)

        audio_samples_remain_length = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        if is_train:
            self.denoising_unet.train()

        temp_dir = "temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        write_video(os.path.join(temp_dir, "video.mp4"), synced_video_frames, fps=25)

        sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

        command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -crf 18 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        subprocess.run(command, shell=True)

    @torch.no_grad()
    def rolling_forward(
            self,
            num_inference_steps,
            guidance_scale,
            mask_latents,
            masked_image_latents,
            ref_latents,
            audio_embeds,
            latents,
            last_infer=False,
            starting_timestep=500
    ):

        do_classifier_free_guidance = guidance_scale > 1.0

        if not last_infer:
            rolling_timesteps = num_inference_steps // latents.shape[2]
        else:
            rolling_timesteps = num_inference_steps

        with self.progress_bar(total=rolling_timesteps) as progress_bar:
            for j, t in enumerate(range(rolling_timesteps)):
                # expand the latents if we are doing classifier free guidance
                denoising_unet_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                if not last_infer:
                    t_tensor = self.rolling_step_table[-(j + 1)][None,...]
                else:
                    t_tensor = self.rolling_step_table[-1][None,...]
                    t_tensor = t_tensor - int((starting_timestep / num_inference_steps) * j)

                t_tensor = torch.where(t_tensor < 0, torch.ones_like(t_tensor), t_tensor)
                denoising_unet_input = self.scheduler.scale_model_input(denoising_unet_input, t_tensor)

                # concat latents, mask, masked_image_latents in the channel dimension
                denoising_unet_input = torch.cat(
                    [denoising_unet_input, mask_latents, masked_image_latents, ref_latents], dim=1
                )

                t_tensor = t_tensor.repeat(denoising_unet_input.shape[0], 1)
                # predict the noise residual
                noise_pred = self.denoising_unet(
                    denoising_unet_input,
                    t_tensor,
                    encoder_hidden_states=audio_embeds
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = step_per_frame(self.scheduler, latents, noise_pred, t_tensor)

        self.latents = latents.clone().detach().contiguous()

        return latents

    def encode_single_image(self, image, **kwargs):
        image = image.unsqueeze(0).unsqueeze(2)  # (1,3,1,H,W)
        lat = self.encode_latents(image, **kwargs)  # (1,c_lat,1,h',w')
        return lat[:, :, 0]  # (c_lat,h',w')


def append_noised_latent_tail(latents: torch.Tensor,
                              scheduler,
                              timestep=500,
                              initial_latent: torch.Tensor = None,
                              ):
    """
    Append a new noisy latent to the tail by reusing the first frame with added noise at a fixed timestep.

    Args:
        latents: [B, C, T, H, W]
        scheduler: diffusion scheduler with add_noise()
        timestep: noise level to apply to the first frame (default 500)

    Returns:
        latents: [B, C, T+1, H, W] with new frame appended
    """
    B, C, T, H, W = latents.shape
    first_frame = initial_latent[0].unsqueeze(0) #latents[:, :, 0, :, :]  # [B, C, H, W]
    noise = torch.randn_like(first_frame)
    timestep_tensor = torch.full((B,), timestep, device=latents.device, dtype=torch.long)
    noised_frame = scheduler.add_noise(first_frame, noise, timesteps=timestep_tensor)
    noised_frame = noised_frame.unsqueeze(2)  # [B, C, 1, H, W]
    return torch.cat([latents[:, :, 1:, :, :], noised_frame], dim=2).contiguous()


def get_tedi_timestep_tensor_all(
    window_size: int,
    max_denoise_step: int,
    max_timestep: int,
    device: torch.device
    ) -> torch.Tensor:
    """
    Generate TEDi timestep tensor of shape [C, T], where:
    - C = max_denoise_step // window_size (number of rolling steps)
    - T = window size (frames per window)

    The value at t[c][j] = timesteps[c * T + j], where timesteps is evenly interpolated from [0..max_timestep].

    Args:
        window_size (int): T, number of frames per window
        max_denoise_step (int): K, number of denoise steps per frame
        max_timestep (int): highest noise timestep allowed (e.g., 500)
        device (torch.device): torch device

    Returns:
        Tensor [C, T] with correct timestep layout
    """
    if max_denoise_step % window_size != 0:
        raise ValueError("max_denoise_step must be divisible by window_size.")

    C = max_denoise_step // window_size
    T = window_size
    K = max_denoise_step
    shift = round(0.15 * max_timestep / max_denoise_step)
    # Create synthetic timesteps from 0 to max_timestep
    # synthetic_timesteps = torch.linspace(0, max_timestep, steps=K).long()  # [K]
    synthetic_timesteps = make_schedule(
        K, max_timestep, mode='shift', shift=7, device=device
    )                                                 # [K]
    # Build index: t[c][j] = synthetic_timesteps[c * T + j]
    index_matrix = torch.arange(T).unsqueeze(1) * C + torch.arange(C).unsqueeze(0)
    index_matrix = index_matrix.permute(1,0)

    t_tensor = synthetic_timesteps[index_matrix]  # shape [C, T]
    return t_tensor.to(device)


def init_rolling_latents_ddim(clean_latents, scheduler, max_t=999, denoise_steps=30):
    """
    Add increasing noise to a sequence of latents for rolling inference, batch-wise.

    Args:
        clean_latents: [B, C, T, H, W]
        scheduler: DDIMScheduler (or compatible with add_noise)
        max_t: total training steps (e.g., 999)
        denoise_steps: denoising steps per latent (e.g., 30)

    Returns:
        [B, C, T, H, W] noisy latents
    """
    B, C, T, H, W = clean_latents.shape
    shift = round(0.15 * max_t / denoise_steps)

    # Compute timestep for each position t in [0..T-1]
    # t_values = torch.linspace(0, max_t * (T - 1) / denoise_steps, steps=T).long().to(clean_latents.device)  # [T]
    t_values = make_schedule(
        T,                        # we need one timestep per **frame**
        max_t,
        mode='shift',
        shift=7,
        device=clean_latents.device
    )
    # Generate random noise for each sample, frame
    noise = torch.randn_like(clean_latents)  # [B, C, T, H, W]
    noisy_latents = []

    # Apply noise frame-wise
    for t in range(T):
        t_tensor = t_values[t]
        # clean_latents[:, :, t] → [B, C, H, W]
        x_start = clean_latents[:, :, t, :, :]  # [B, C, H, W]
        eps = noise[:, :, t, :, :]              # [B, C, H, W]
        # scheduler.add_noise supports broadcasting t if it's a scalar or [B]
        latent_t = scheduler.add_noise(x_start, eps, timesteps=t_tensor)
        noisy_latents.append(latent_t)  # [B, C, H, W]

    # Stack → [T, B, C, H, W] → permute to [B, C, T, H, W]
    return torch.stack(noisy_latents, dim=0).permute(1, 2, 0, 3, 4).contiguous()


def make_schedule(
    K: int,
    max_t: int,
    mode: Literal["linear", "linear_quadratic", "shift"] = "linear",
    shift: int | float = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Return a length-K tensor of integer timesteps ∈ [0, max_t].

    - *linear*            :  0 … max_t  (your current behaviour)
    - *linear_quadratic*  :  (i / K)**2 * max_t  → coarse → fine
    - *shift*             :  ((i + s) / (K-1 + s)) * max_t
                             (skips the very noisiest part;         s ≈ 0.15*K is a good start)
    """
    i = torch.arange(K, device=device)

    if mode == "linear":
        r = i / (K - 1)
    elif mode == "linear_quadratic":
        r = (i / (K - 1)) ** 2
    elif mode == "shift":
        r = (i + shift) / (K - 1 + shift)
    else:
        raise ValueError(f"Unknown mode {mode}")

    return (r * max_t).round().long()          # ↑ convert to scheduler integer domain


def step_per_frame(scheduler, latents, noise_pred, t_tensor, **kwargs):
    B, C, T, H, W = latents.shape
    t_each = t_tensor[0]
    latents_next = [
        scheduler.step(
            noise_pred[:, :, j], t_each[j].item(), latents[:, :, j],
            **kwargs, return_dict=False
        )[0]
        for j in range(T)
    ]
    return torch.stack(latents_next, dim=2)
