import torch
import numpy as np
import PIL
from diffusers import StableVideoDiffusionPipeline


class CustomSVDPipeline(StableVideoDiffusionPipeline):
    def __call__(
        self,
        image,
        height=None,
        width=None,
        num_frames=25,
        decode_chunk_size=8,
        motion_bucket_id=127,
        fps=7,
        noise_aug_strength=0.02,
        control_input=None,  # New!
        generator=None,
        **kwargs,
    ):
        device = self.device
        dtype = self.unet.dtype
        batch_size = 1

        # === Step 1: Encode image prompt ===
        if isinstance(image, PIL.Image.Image):
            image = np.array(image)

        if image.shape[-1] == 4:
            print("[CustomSVDPipeline] Stripping 4th channel (Canny) from prompt encoder input")
            image = image[..., :3]

        inputs = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder(inputs).image_embeds  # [B, D]
        image_embeds = image_embeds.unsqueeze(1).repeat(1, num_frames, 1)  # [B, F, D]

        # === Step 2: Sample latents ===
        latents = torch.randn(
            (batch_size, num_frames, self.unet.config.in_channels, height // 8, width // 8),
            device=device,
            dtype=dtype,
            generator=generator,
        )

        # === Step 3: Time conditioning ===
        def get_add_time_ids(fps, motion_bucket_id, noise_aug_strength):
            add_time_ids = torch.tensor([[fps, motion_bucket_id, noise_aug_strength]], device=device, dtype=dtype)
            return add_time_ids.repeat(batch_size, 1)

        added_time_ids = get_add_time_ids(fps, motion_bucket_id, noise_aug_strength)

        # === Step 4: Denoising
        sigmas = torch.exp(torch.tensor([0.7], device=device))  # could use a sampler
        sigmas = sigmas.view(1, 1, 1, 1, 1).expand_as(latents)
        noise = torch.randn_like(latents)
        noisy_latents = latents + noise * sigmas
        timesteps = 0.25 * sigmas[:, 0, 0, 0, 0].log()

        inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

        # === Step 5: UNet forward with control input ===
        model_pred = self.unet(
            sample=inp_noisy_latents,
            timestep=timesteps.squeeze(0),
            encoder_hidden_states=image_embeds,
            added_time_ids=added_time_ids,
            control_input=control_input,
        ).sample

        # === Step 6: Decode
        c_out = -sigmas / ((sigmas**2 + 1)**0.5)
        c_skip = 1 / (sigmas**2 + 1)
        denoised_latents = model_pred * c_out + c_skip * noisy_latents
        latents = denoised_latents / self.vae.config.scaling_factor

        # === Step 7: Decode latents to video
        b, f, c, h, w = latents.shape
        latents = latents.reshape(-1, c, h, w)
        video = self.vae.decode(latents).sample
        video = video.reshape(b, f, *video.shape[1:])
        video = ((video / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8).permute(0, 1, 2, 3, 4).cpu().numpy()

        return {"frames": [video[0]]}