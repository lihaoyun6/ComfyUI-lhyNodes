# Convert code from: https://github.com/hanfeisun/pyultrahdr
from __future__ import annotations

import io
import math
import os
import struct
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

import folder_paths

# ── 默认参数预设 ─────────────────────────────────────────────────────────────

DEFAULT_HDR_PARAMS = {
    "peak_nits": 1000.0,
    "sdr_white": 203.0,
    "shadow_boost": 1.3,
    "hi_gamma": 1.0,
    "quality": 92,
    "gainmap_quality": 85,
}

# ── 色彩与 HDR 转换底层函数 ──────────────────────────────────────────────────

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


BT709_TO_BT2020 = np.array([
    [0.6274040, 0.3292820, 0.0433136],
    [0.0690970, 0.9195400, 0.0113612],
    [0.0163916, 0.0880132, 0.8955950],
], dtype=np.float32)


def inverse_tone_map(
    rgb_lin: np.ndarray,
    sdr_white: float = 203.0,
    peak_nits: float = 1000.0,
    shadow_boost: float = 1.3,
    hi_gamma: float = 1.0,
) -> np.ndarray:
    w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    L = np.clip((rgb_lin * w).sum(-1, keepdims=True), 1e-6, 1.0)
    peak_boost = peak_nits / sdr_white
    boost = shadow_boost + (peak_boost - shadow_boost) * (L ** hi_gamma)
    return rgb_lin * sdr_white * boost


def compute_gain_map(
    sdr_lin: np.ndarray,
    hdr_nits: np.ndarray,
    sdr_white: float,
) -> tuple[np.ndarray, float, float]:
    hdr_norm = hdr_nits / sdr_white
    w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    L_sdr = np.clip((sdr_lin * w).sum(-1), 1e-4, None)
    L_hdr = np.clip((hdr_norm * w).sum(-1), 1e-4, None)
    gain = np.log2(L_hdr) - np.log2(L_sdr)
    gain = np.maximum(gain, 0.0)
    return gain.astype(np.float32), 0.0, float(gain.max())


def _xmp_hdrgm(gain_min: float, gain_max: float, hdr_capacity: float) -> bytes:
    xml = (
        '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 5.5.0">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about=""\n'
        '    xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/"\n'
        '   hdrgm:Version="1.0"\n'
        f'   hdrgm:GainMapMin="{gain_min:.6f}"\n'
        f'   hdrgm:GainMapMax="{gain_max:.6f}"\n'
        f'   hdrgm:HDRCapacityMin="0.000000"\n'
        f'   hdrgm:HDRCapacityMax="{hdr_capacity:.6f}"\n'
        '   hdrgm:OffsetHDR="0.000000"\n'
        '   hdrgm:OffsetSDR="0.000000"/>\n'
        ' </rdf:RDF>\n'
        '</x:xmpmeta>\n'
        '<?xpacket end="w"?>'
    )
    payload = b'http://ns.adobe.com/xap/1.0/\x00' + xml.encode('utf-8')
    return b'\xff\xe1' + struct.pack('>H', len(payload) + 2) + payload


def _mpf_app2(primary_size: int, gainmap_size: int, gainmap_offset: int) -> bytes:
    TIFF_HDR = b'II' + struct.pack('<H', 42) + struct.pack('<I', 8)
    MP_ENTRIES_TIFF_OFF = 50

    e_ver = struct.pack('<HHI4s', 0xB000, 7, 4, b'0100')
    e_num = struct.pack('<HHII', 0xB001, 4, 1, 2)
    e_entry = struct.pack('<HHII', 0xB002, 7, 32, MP_ENTRIES_TIFF_OFF)
    ifd = struct.pack('<H', 3) + e_ver + e_num + e_entry + struct.pack('<I', 0)

    mp1 = struct.pack('<IIIHH', 0x00030000, primary_size, 0, 0, 0)
    mp2 = struct.pack('<IIIHH', 0x00000000, gainmap_size, gainmap_offset, 0, 0)

    data = b'MPF\x00' + TIFF_HDR + ifd + mp1 + mp2
    assert len(data) == 86
    return b'\xff\xe2' + struct.pack('>H', 88) + data


def build_ultra_hdr(
    sdr_uint8: np.ndarray,
    gain_log2: np.ndarray,
    gain_min: float,
    gain_max: float,
    peak_nits: float,
    sdr_white: float,
    base_quality: int = 92,
    gainmap_quality: int = 85,
) -> bytes:
    hdr_capacity = math.log2(peak_nits / sdr_white)
    xmp_app1 = _xmp_hdrgm(gain_min, gain_max, hdr_capacity)

    buf = io.BytesIO()
    Image.fromarray(sdr_uint8, 'RGB').save(
        buf, format='JPEG', quality=base_quality, subsampling=0
    )
    base_jpeg = buf.getvalue()

    gain_range = gain_max - gain_min or 1.0
    gain_u8 = np.clip((gain_log2 - gain_min) / gain_range * 255 + 0.5, 0, 255).astype(np.uint8)
    gm_buf = io.BytesIO()
    Image.fromarray(gain_u8, 'L').save(gm_buf, format='JPEG', quality=gainmap_quality)
    gm_raw = gm_buf.getvalue()
    gainmap_jpeg = gm_raw[:2] + xmp_app1 + gm_raw[2:]

    A = len(xmp_app1)
    B = len(base_jpeg)
    G = len(gainmap_jpeg)
    primary_size = 2 + A + 90 + (B - 2)
    gainmap_offset = (A + B + 90) - (A + 10)
    mpf_app2 = _mpf_app2(primary_size, G, gainmap_offset)

    return b'\xff\xd8' + xmp_app1 + mpf_app2 + base_jpeg[2:] + gainmap_jpeg


# ── 节点 1: 参数调节节点 ──────────────────────────────────────────────────────

class UltraHDRParameters:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "peak_nits": ("FLOAT", {"default": 1000.0, "min": 300.0, "max": 10000.0, "step": 50.0, "round": 0.1}),
                "sdr_white": ("FLOAT", {"default": 203.0, "min": 80.0, "max": 500.0, "step": 1.0, "round": 0.1}),
                "shadow_boost": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 3.0, "step": 0.05, "round": 0.01}),
                "hi_gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05, "round": 0.01}),
                "quality": ("INT", {"default": 92, "min": 1, "max": 100, "step": 1}),
                "gainmap_quality": ("INT", {"default": 85, "min": 1, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("HDR_PARAMS",)
    RETURN_NAMES = ("hdr_params",)
    FUNCTION = "get_params"
    CATEGORY = "image/hdr"

    def get_params(
        self,
        peak_nits: float,
        sdr_white: float,
        shadow_boost: float,
        hi_gamma: float,
        quality: int,
        gainmap_quality: int,
    ):
        params = {
            "peak_nits": peak_nits,
            "sdr_white": sdr_white,
            "shadow_boost": shadow_boost,
            "hi_gamma": hi_gamma,
            "quality": quality,
            "gainmap_quality": gainmap_quality,
        }
        return (params,)


# ── 节点 2: 保存图像节点 ──────────────────────────────────────────────────────

class SaveUltraHDRImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename_prefix": ("STRING", {"default": "UltraHDR"}),
            },
            "optional": {
                "hdr_params": ("HDR_PARAMS",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_images"
    CATEGORY = "lhyNodes/Image"
    DESCRIPTION = "UltraHDR JPG image don't contain workflow metadata!"

    def save_images(
        self,
        images: torch.Tensor,
        filename_prefix: str = "UltraHDR",
        hdr_params: Optional[Dict[str, Any]] = None,
        prompt=None,
        extra_pnginfo=None,
    ):
        # 如果未接入 hdr_params，则继承并使用全局默认参数
        params = DEFAULT_HDR_PARAMS.copy()
        if hdr_params is not None:
            params.update(hdr_params)

        peak_nits = float(params["peak_nits"])
        sdr_white = float(params["sdr_white"])
        shadow_boost = float(params["shadow_boost"])
        hi_gamma = float(params["hi_gamma"])
        quality = int(params["quality"])
        gainmap_quality = int(params["gainmap_quality"])

        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_text = \
            folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])

        results = list()
        for batch_number, img_tensor in enumerate(images):
            # ComfyUI 的 images 格式是 [H, W, C] 浮点 (0.0 ~ 1.0)，RGB
            img_np = img_tensor.cpu().numpy()
            if img_np.shape[-1] > 3:
                img_np = img_np[..., :3]

            sdr_uint8 = np.clip(img_np * 255.0 + 0.5, 0, 255).astype(np.uint8)
            sdr_lin = srgb_to_linear(img_np)

            # iTMO 扩展到 HDR (nits)
            hdr_nits = inverse_tone_map(
                sdr_lin,
                sdr_white=sdr_white,
                peak_nits=peak_nits,
                shadow_boost=shadow_boost,
                hi_gamma=hi_gamma,
            )
            hdr_bt2020 = np.clip(hdr_nits @ BT709_TO_BT2020.T, 0.0, peak_nits)

            # 计算 Gain Map
            gain_log2, gain_min, gain_max = compute_gain_map(sdr_lin, hdr_bt2020, sdr_white)

            # 打包生成标准的 Ultra HDR JPEG
            ultra_hdr_bytes = build_ultra_hdr(
                sdr_uint8=sdr_uint8,
                gain_log2=gain_log2,
                gain_min=gain_min,
                gain_max=gain_max,
                peak_nits=peak_nits,
                sdr_white=sdr_white,
                base_quality=quality,
                gainmap_quality=gainmap_quality,
            )

            file = f"{filename}_{counter:05}_.jpg"
            file_path = os.path.join(full_output_folder, file)
            with open(file_path, "wb") as f:
                f.write(ultra_hdr_bytes)

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {"ui": {"images": results}}

NODE_CLASS_MAPPINGS = {
    "SaveUltraHDRImage": SaveUltraHDRImage,
    "UltraHDRParameters": UltraHDRParameters
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveUltraHDRImage": "Save UltraHDR Image",
    "UltraHDRParameters": "UltraHDR Parameters"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]