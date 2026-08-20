import os
import time
import secrets
import folder_paths
import torch

import numpy as np
import comfy.model_management as mm

from PIL import Image, ImageOps
from .nodes import QueueHandler_States
from ..utils.kj_tiny_vae import load_tiny_vae_decoder

from server import PromptServer
from comfy_extras.nodes_lt import LTXVSeparateAVLatent

def _tiny_vae_decode_to_pil(decoder, x0, max_frames=None, stride=1):
    # Raises on failure so the caller can disable the decoder instead of retrying every step.
    if x0.ndim == 4:
        rgb = decoder.decode(x0[:1])[0].movedim(0, -1).unsqueeze(0).contiguous()
    elif x0.ndim == 5:
        indices = list(range(0, x0.shape[2], max(1, stride)))
        if max_frames is not None and 0 < max_frames < len(indices):
            picks = np.linspace(0, len(indices) - 1, max_frames).round().astype(int).tolist()
            indices = [indices[i] for i in picks]
        rgb = decoder.decode_video(x0[:1], frame_indices=indices)
    else:
        return []
    u8 = rgb.clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
    return [Image.fromarray(u8[i]) for i in range(u8.shape[0])]

class LatentPreviewTinyVAE:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "tiny_vae": (["none"] + folder_paths.get_filename_list("vae_approx"), {"default": "none"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60, "step": 1}),
                "frames": (["all", "50%", "25%", "10%"], {"default": "all"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "generate_preview"
    CATEGORY = "lhyNodes/Preview"

    def generate_preview(self, latent, tiny_vae, fps, frames, unique_id):
        video_latent, audio_latent = LTXVSeparateAVLatent.execute(latent)
        latent_tensor = video_latent["samples"]

        # 1. 仅使用 微型 VAE (Tiny VAE) 加载与解码
        t_vae = load_tiny_vae_decoder(tiny_vae)
        if t_vae is None:
            raise RuntimeError(f"Could not load Tiny VAE: '{tiny_vae}'")

        # 通道数校验（避免通道不匹配导致的崩溃）
        if hasattr(latent_tensor, "shape") and len(latent_tensor.shape) >= 2:
            channels = int(latent_tensor.shape[1])
            if hasattr(t_vae, "latent_channels") and channels != t_vae.latent_channels:
                raise ValueError(
                    f"Tiny VAE '{tiny_vae}' decodes {t_vae.latent_channels}-channel "
                    f"latents but this model's are {channels}-channel; ignoring it."
                )

        # ============================================================
        # 计算要解码的真实帧数，并保证最低解码不小于 1 帧
        # ============================================================
        if latent_tensor.ndim == 5:
            total_t = latent_tensor.shape[2]  # 5维视频 Latent [B, C, T, H, W] 的时间轴 T
        elif latent_tensor.ndim == 4:
            total_t = latent_tensor.shape[0]  # 4维批次图片 Latent [B, C, H, W] 的批次大小 B
        else:
            total_t = 1
        
        fps_scale = 1
        if frames == "50%":
            fps_scale = 2
            max_frames = max(1, round(total_t / 2))
        elif frames == "25%":
            fps_scale = 4
            max_frames = max(1, round(total_t / 4))
        elif frames == "10%":
            fps_scale = 10
            max_frames = max(1, round(total_t / 10))
        else:  # "all"
            max_frames = None

        # 解码为 PIL 图片列表
        pil_frames = _tiny_vae_decode_to_pil(t_vae, latent_tensor, max_frames=max_frames)

        if not pil_frames:
            return (latent,)

        # 优化项 A：最大分辨率改为 384（像素点减小接近半数，速度极大提升）
        max_resolution = 512
        processed_frames = []
        for img in pil_frames:
            if img.mode != "RGB":
                img = img.convert("RGB")
            if max_resolution > 0 and (img.width > max_resolution or img.height > max_resolution):
                # 优化项 B：使用 Image.BILINEAR 代替昂贵的 Image.LANCZOS
                img = ImageOps.contain(img, (max_resolution, max_resolution), Image.BILINEAR)
            processed_frames.append(img)
            
        # 优化项 C：WebP 保存加速
        output_dir = folder_paths.get_temp_directory()
        filename = f"TinyVAE_Preview_{secrets.token_hex(8)}.webp"
        filepath = os.path.join(output_dir, filename)
        
        if latent_tensor.ndim == 5:
            effective_fps = max(1.0, fps / 3.38 / fps_scale)
        else:
            effective_fps = fps
        duration_ms = int(1000 / effective_fps)
        
        processed_frames[0].save(
            filepath,
            format="WEBP",
            save_all=True,
            append_images=processed_frames[1:],
            duration=duration_ms,
            loop=0,
            quality=70,  # 优化：降至 50，减少量化开销
            method=0    # 优化：开到 0 (最快编码模式，速度提升数倍)
        )
        
        PromptServer.instance.send_sync("tiny_vae_preview", {
            "node_id": unique_id,
            "filename": filename,
            "subfolder": "",
            "type": "temp"
        })
        
        QueueHandler_States[unique_id] = "paused"
        while QueueHandler_States.get(unique_id,"") == "paused":
            mm.throw_exception_if_processing_interrupted()
            time.sleep(0.2)
        
        return (latent,)

NODE_CLASS_MAPPINGS = {
    "LatentPreviewTinyVAE": LatentPreviewTinyVAE
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentPreviewTinyVAE": "Latent Preview (Tiny VAE)"
}