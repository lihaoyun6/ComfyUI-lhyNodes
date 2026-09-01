import os
import time
import wave
import shutil
import secrets
import subprocess

import nodes
import node_helpers
import folder_paths

import torch

import numpy as np
import comfy.model_management as mm

from PIL import Image, ImageOps
from .nodes import QueueHandler_States
from ..utils.kj_tiny_vae import load_tiny_vae_decoder

from server import PromptServer
from comfy_extras.nodes_lt import LTXVSeparateAVLatent

try:
    from comfy_extras.nodes_minimax_h3 import _empty_av_latent, _resize
except ImportError:
    pass


class MiniMaxH3ImageToVideo2Pass:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "width_1": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height_1": ("INT", {"default": 432, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "width_2": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height_2": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "length": ("INT", {
                    "default": 124, "min": 5, "max": 3600, "step": 17,
                    "tooltip": "Frame count at 24 fps, snapped up to the model's 17k+5 grid (124 = ~5s; trained range is ~124-362, longer is untested)",
                }),
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive_1", "positive_2", "latent")
    FUNCTION = "execute"
    CATEGORY = "model/conditioning/minimax"

    def execute(self, clip, vae, prompt, width_1, height_1, width_2, height_2, length, first_frame=None, last_frame=None):
        # 1. 仅生成低清阶段所需的初始空 Latent
        latent_1, frame_count = _empty_av_latent(width_1, height_1, length)

        images_for_clip = []
        keyframes_1 = []
        keyframes_2 = []

        # 2. 处理首尾帧尺寸缩放
        if first_frame is not None:
            # Pass 1 低清首帧
            img1 = _resize(first_frame[:1], width_1, height_1, "disabled")
            keyframes_1.append({"resolved_frame_index": 0, "image": img1})

            # Pass 2 高清首帧
            img2 = _resize(first_frame[:1], width_2, height_2, "disabled")
            keyframes_2.append({"resolved_frame_index": 0, "image": img2})

            # 将高清图像送入 CLIP，以提取最丰富清晰的视觉语义
            images_for_clip.append(img2)

        if last_frame is not None:
            # Pass 1 低清尾帧
            img1 = _resize(last_frame[:1], width_1, height_1, "center")
            keyframes_1.append({"resolved_frame_index": frame_count - 1, "image": img1})

            # Pass 2 高清尾帧
            img2 = _resize(last_frame[:1], width_2, height_2, "center")
            keyframes_2.append({"resolved_frame_index": frame_count - 1, "image": img2})

            # 将高清图像送入 CLIP
            images_for_clip.append(img2)

        # 3. CLIP 全局仅运行 1 次（核心提速点！）
        tokens = clip.tokenize(prompt, images=images_for_clip)
        cond_base = clip.encode_from_tokens_scheduled(tokens)

        cond_1 = cond_base
        cond_2 = cond_base

        # 4. 分别对低清与高清的首尾帧做 VAE 编码与绑定
        if keyframes_1:
            # 绑定低清阶段 Keyframes
            for kf in keyframes_1:
                kf["latent"] = vae.encode(kf.pop("image"))
            cond_1 = node_helpers.conditioning_set_values(cond_1, {
                "minimax_keyframes": keyframes_1,
                "minimax_frame_count": frame_count,
            })

            # 绑定高清阶段 Keyframes (保证第二阶段精修具有原生高清锚点)
            for kf in keyframes_2:
                kf["latent"] = vae.encode(kf.pop("image"))
            cond_2 = node_helpers.conditioning_set_values(cond_2, {
                "minimax_keyframes": keyframes_2,
                "minimax_frame_count": frame_count,
            })

        # 5. 返回元组 (positive_1, positive_2, latent)
        return (cond_1, cond_2, latent_1)


# ============================================================
# 官方原版纯净音频解码（无需任何 2D/4D 假兼容）
# ============================================================
def vae_decode_audio(vae, samples, tile=None, overlap=None):
    latent = samples["samples"]
    if hasattr(latent, "is_nested") and latent.is_nested:
        latent = latent.unbind()[-1]
        
    if tile is not None and hasattr(vae, "decode_tiled"):
        audio = vae.decode_tiled(latent, tile_x=tile, tile_y=tile, overlap=overlap).movedim(-1, 1)
    else:
        audio = vae.decode(latent).movedim(-1, 1)
        
    std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio /= std
    vae_sample_rate = getattr(vae, "audio_sample_rate_output", getattr(vae, "audio_sample_rate", 44100))
    sample_rate = vae_sample_rate if "sample_rate" not in samples else samples["sample_rate"]
    return {"waveform": audio, "sample_rate": sample_rate}


# 使用 Python 原生 wave 库写 PCM WAV（纯粹解决 torchcodec 问题）
def save_wav_builtin(filepath, waveform, sample_rate):
    # waveform 保证为 [Channels, Samples]
    audio_np = (waveform.cpu().float().clamp(-1.0, 1.0).numpy() * 32767.0).astype(np.int16)
    num_channels = audio_np.shape[0]
    raw_bytes = audio_np.T.tobytes()
    
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_bytes)
        
        
def _tiny_vae_decode_to_pil(decoder, x0, max_frames=None, stride=1):
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
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "frames": (["all", "50%", "25%", "10%"], {"default": "all"}),
                "pause": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio_vae": ("VAE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "generate_preview"
    CATEGORY = "lhyNodes/Preview"
    
    def generate_preview(self, latent, tiny_vae, fps, frames, pause, unique_id, audio_vae=None):
        video_latent, audio_latent = LTXVSeparateAVLatent.execute(latent)
        latent_tensor = video_latent["samples"]
        
        # 1. Tiny VAE 解码画面
        t_vae = load_tiny_vae_decoder(tiny_vae)
        if t_vae is None:
            raise RuntimeError(f"Could not load Tiny VAE: '{tiny_vae}'")
            
        if hasattr(latent_tensor, "shape") and len(latent_tensor.shape) >= 2:
            channels = int(latent_tensor.shape[1])
            if hasattr(t_vae, "latent_channels") and channels != t_vae.latent_channels:
                raise ValueError(
                    f"Tiny VAE '{tiny_vae}' decodes {t_vae.latent_channels}-channel "
                    f"latents but this model's are {channels}-channel; ignoring it."
                )
                
        if latent_tensor.ndim == 5:
            total_t = latent_tensor.shape[2]
        elif latent_tensor.ndim == 4:
            total_t = latent_tensor.shape[0]
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
        else:
            max_frames = None
            
        pil_frames = _tiny_vae_decode_to_pil(t_vae, latent_tensor, max_frames=max_frames)
        
        if not pil_frames:
            return (latent,)
        
        max_resolution = 512
        processed_frames = []
        for img in pil_frames:
            if img.mode != "RGB":
                img = img.convert("RGB")
            if max_resolution > 0 and (img.width > max_resolution or img.height > max_resolution):
                img = ImageOps.contain(img, (max_resolution, max_resolution), Image.BILINEAR)
            even_w = img.width - (img.width % 2)
            even_h = img.height - (img.height % 2)
            if img.width != even_w or img.height != even_h:
                img = img.crop((0, 0, even_w, even_h))
            processed_frames.append(img)
            
        # 2. 状态检测与 FPS 计算
        if latent_tensor.ndim == 5:
            effective_fps = max(1.0, fps / fps_scale)
        else:
            effective_fps = fps
            
        output_dir = folder_paths.get_temp_directory()
        rand_hex = secrets.token_hex(8)
        
        # 3. 检查 ffmpeg 与音频
        has_ffmpeg = shutil.which("ffmpeg") is not None
        audio_wav_path = None
        use_mp4 = False
        
        if has_ffmpeg:
            if audio_vae is not None and audio_latent is not None:
                try:
                    audio_dict = vae_decode_audio(audio_vae, audio_latent)
                    waveform = audio_dict["waveform"][0]  # 直接取第0批次 [Channels, Samples]
                    sample_rate = int(audio_dict["sample_rate"])
                    
                    audio_wav_path = os.path.join(output_dir, f"temp_audio_{rand_hex}.wav")
                    save_wav_builtin(audio_wav_path, waveform, sample_rate)
                except Exception as e:
                    print(f"[LatentPreviewTinyVAE] Audio decode failed: {e}")
                    audio_wav_path = None
                    
            # FFmpeg 极速视频+音频合并
            filename = f"TinyVAE_Preview_{rand_hex}.mp4"
            filepath = os.path.join(output_dir, filename)
            
            try:
                width, height = processed_frames[0].size
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "rawvideo",
                    "-vcodec", "rawvideo",
                    "-s", f"{width}x{height}",
                    "-pix_fmt", "rgb24",
                    "-r", str(effective_fps),
                    "-i", "-",  # 视频流
                ]
                
                if audio_wav_path and os.path.exists(audio_wav_path):
                    cmd.extend([
                        "-i", audio_wav_path,
                        "-map", "0:v:0",
                        "-map", "1:a:0",
                        "-c:a", "aac",
                        "-b:a", "192k",
                    ])
                else:
                    cmd.extend(["-map", "0:v:0"])
                    
                cmd.extend([
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-crf", "28",
                    "-pix_fmt", "yuv420p",
                    filepath
                ])
                
                process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                raw_video_data = b"".join(frame.tobytes() for frame in processed_frames)
                _, stderr = process.communicate(input=raw_video_data)
                
                if process.returncode == 0 and os.path.exists(filepath):
                    use_mp4 = True
                else:
                    err_msg = stderr.decode('utf-8', errors='ignore') if stderr else 'Unknown error'
                    print(f"[LatentPreviewTinyVAE] FFmpeg error (code {process.returncode}):\n{err_msg}")
            except Exception as e:
                print(f"[LatentPreviewTinyVAE] FFmpeg synthesis exception: {e}")
                use_mp4 = False
                
        # Fallback 纯 WebP (当无 ffmpeg 时)
        if not use_mp4:
            filename = f"TinyVAE_Preview_{rand_hex}.webp"
            filepath = os.path.join(output_dir, filename)
            duration_ms = int(1000 / effective_fps)
            
            processed_frames[0].save(
                filepath,
                format="WEBP",
                save_all=True,
                append_images=processed_frames[1:],
                duration=[duration_ms] * len(processed_frames),
                loop=0,
                quality=70,
                method=0
            )
            
        # 擦除临时 .wav
        if audio_wav_path and os.path.exists(audio_wav_path):
            try:
                os.remove(audio_wav_path)
            except Exception:
                pass
                
        # 4. 消息推流
        PromptServer.instance.send_sync("tiny_vae_preview", {
            "node_id": unique_id,
            "filename": filename,
            "subfolder": "",
            "type": "temp"
        })
        
        if pause:
            QueueHandler_States[unique_id] = "paused"
        else:
            QueueHandler_States.pop(unique_id, None)
            
        while QueueHandler_States.get(unique_id, "") == "paused":
            mm.throw_exception_if_processing_interrupted()
            time.sleep(0.2)
            
        return (latent,)
    
    
NODE_CLASS_MAPPINGS = {
    "LatentPreviewTinyVAE": LatentPreviewTinyVAE,
    "MiniMaxH3ImageToVideo2Pass": MiniMaxH3ImageToVideo2Pass,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentPreviewTinyVAE": "Latent Preview (Tiny VAE)",
    "MiniMaxH3ImageToVideo2Pass": "MiniMax H3 Image to Video 2-Pass",
}