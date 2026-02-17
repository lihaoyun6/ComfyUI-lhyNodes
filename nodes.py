import os
import random
import cv2
import random
import json
import torch
import torch.nn.functional as F
import numpy as np

from nodes import MAX_RESOLUTION
from ultralytics import YOLO
from .utils import cqdm
import comfy.samplers

RES4 = False
current_dir = os.path.dirname(os.path.abspath(__file__))
plugin_path = os.path.join(current_dir, "..", "RES4LYF")
if os.path.exists(plugin_path):
    RES4 = True
    
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
    
any_type = AnyType("*")

def get_schedulers(remove = []):
    schedulers = comfy.samplers.KSampler.SCHEDULERS
    if RES4:
        schedulers = list(dict.fromkeys(comfy.samplers.KSampler.SCHEDULERS + ['bong_tangent', 'beta57']))
    return [x for x in schedulers if x not in remove]

class detailerKSamplerSchedulerFallback:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scheduler": (get_schedulers(), {"forceInput": True}),
                "fallback_scheduler": (get_schedulers(['beta57']) + ['AYS SDXL', 'AYS SD1', 'AYS SVD', 'GITS[coeff=1.2]', 'LTXV[default]', 'OSS FLUX', 'OSS Wan', 'OSS Chroma'],),
            },
        }
    
    RETURN_TYPES = get_schedulers(['beta57']) + ['AYS SDXL', 'AYS SD1', 'AYS SVD', 'GITS[coeff=1.2]', 'LTXV[default]', 'OSS FLUX', 'OSS Wan', 'OSS Chroma'],
    RETURN_NAMES = ("SCHEDULER",)
    FUNCTION = "main"
    CATEGORY = "lhyNodes/Utils"
    
    def main(self, scheduler, fallback_scheduler):
        if scheduler not in get_schedulers(['beta57']) + ['AYS SDXL', 'AYS SD1', 'AYS SVD', 'GITS[coeff=1.2]', 'LTXV[default]', 'OSS FLUX', 'OSS Wan', 'OSS Chroma']:
            return (fallback_scheduler,)
        return(scheduler,)

class effKSamplerSchedulerFallback:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scheduler": (get_schedulers(), {"forceInput": True}),
                "fallback_scheduler": (get_schedulers(['bong_tangent', 'beta57']) + ["AYS SD1", "AYS SDXL", "AYS SVD", "GITS"],),
            },
        }
    
    RETURN_TYPES = get_schedulers(['bong_tangent', 'beta57']) + ["AYS SD1", "AYS SDXL", "AYS SVD", "GITS"],
    RETURN_NAMES = ("SCHEDULER",)
    FUNCTION = "main"
    CATEGORY = "lhyNodes/Utils"
    
    def main(self, scheduler, fallback_scheduler):
        if scheduler not in get_schedulers(['bong_tangent', 'beta57']) + ["AYS SD1", "AYS SDXL", "AYS SVD", "GITS"]:
            return (fallback_scheduler,)
        return(scheduler,)

class KSamplerSchedulerFallback:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scheduler": (get_schedulers(), {"forceInput": True}),
                "fallback_scheduler": (get_schedulers(),),
            },
        }
    
    RETURN_TYPES = get_schedulers(),
    RETURN_NAMES = ("SCHEDULER",)
    FUNCTION = "main"
    CATEGORY = "lhyNodes/Utils"
    
    def main(self, scheduler, fallback_scheduler):
        if scheduler not in get_schedulers():
            return (fallback_scheduler,)
        return(scheduler,)

class KSamplerConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps_total": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": MAX_RESOLUTION,
                    "step": 1,
                }),
                "cfg": ("FLOAT", {
                    "default": 8.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.5,
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS + ['AYS SDXL', 'AYS SD1', 'AYS SVD', 'GITS', 'GITS[coeff=1.2]', 'LTXV[default]', 'OSS FLUX', 'OSS Wan', 'OSS Chroma'],),
            },
        }
    
    RETURN_TYPES = ("INT", "FLOAT", comfy.samplers.KSampler.SAMPLERS, get_schedulers())
    RETURN_NAMES = ("STEPS", "CFG", "SAMPLER", "SCHEDULER")
    FUNCTION = "main"
    CATEGORY = "lhyNodes/Utils"
    
    def main(self, steps_total, cfg, sampler_name, scheduler):
        return (
            steps_total,
            cfg,
            sampler_name,
            scheduler,
        )

class MaskToSAMCoords:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_regions": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
                "points_per_region": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("coordinates",)
    FUNCTION = "convert"
    CATEGORY = "lhyNodes/Mask"
    
    def convert(self, mask: torch.Tensor, threshold, max_regions, points_per_region):
        mask_np = mask[0].cpu().numpy()
        
        min_val, max_val = mask_np.min(), mask_np.max()
        if max_val <= min_val:
            print("Warning: MaskToSAMCoords received a solid color mask. Returning default coordinate (0,0).")
            coords_str = json.dumps([{"x": 0, "y": 0}])
            return (coords_str,)
        
        normalized_mask = (mask_np - min_val) / (max_val - min_val)
        binary_mask = (normalized_mask > threshold).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        coords = []
        for cnt in contours[:max_regions]:
            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 0 or h <= 0:
                continue
            
            for _ in range(points_per_region):
                px = random.randint(x, x + w - 1)
                py = random.randint(y, y + h - 1)
                
                if cv2.pointPolygonTest(cnt, (float(px), float(py)), False) >= 0:
                    coords.append({"x": int(px), "y": int(py)})
                    
        if not coords:
            print("Warning: MaskToSAMCoords did not find any regions above the threshold. Returning default coordinate (0,0).")
            coords.append({"x": 0, "y": 0})
            
        coords_str = json.dumps(coords)
        
        return (coords_str,)
    
class MaskToSAMCoordsV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Mark on the mask to generate positive conditions."}),
                "threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_regions": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
                "points_per_region": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "negative_color": (["red", "green", "blue", "magenta"], {
                    "default": "red",
                    "tooltip": "red=#FF0000, green=#00FF00, blue=#0000FF, magenta=#FF00FF."
                }),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Mark the image using a colored brush to generate negative conditions."})
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("coordinates_positive", "coordinates_negative",)
    FUNCTION = "convert"
    CATEGORY = "lhyNodes/Mask"
    
    def convert(self, mask: torch.Tensor, threshold, max_regions, points_per_region, negative_color, image: torch.Tensor = None):
        color_map = {
            "red": np.array([0, 0, 255]),
            "green": np.array([0, 255, 0]),
            "blue": np.array([255, 0, 0]),
            "magenta": np.array([255, 0, 255])
        }
        neg_color = color_map.get(negative_color, np.array([0, 0, 255]))
        
        mask_np = mask[0].cpu().numpy()
        
        min_val, max_val = mask_np.min(), mask_np.max()
        if max_val <= min_val:
            print("Warning: MaskToSAMCoords received a solid color mask. Returning default coordinate (0,0).")
            positive_coords = [{"x": 0, "y": 0}]
        else:
            normalized_mask = (mask_np - min_val) / (max_val - min_val)
            binary_mask = (normalized_mask > threshold).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            positive_coords = []
            for cnt in contours[:max_regions]:
                x, y, w, h = cv2.boundingRect(cnt)
                if w <= 0 or h <= 0:
                    continue
                
                for _ in range(points_per_region):
                    px = random.randint(x, x + w - 1)
                    py = random.randint(y, y + h - 1)
                    if cv2.pointPolygonTest(cnt, (float(px), float(py)), False) >= 0:
                        positive_coords.append({"x": int(px), "y": int(py)})
                        
            if not positive_coords:
                print("Warning: MaskToSAMCoords did not find any positive regions above the threshold. Returning default coordinate (0,0).")
                positive_coords.append({"x": 0, "y": 0})
                
        positive_coords_str = json.dumps(positive_coords)
        
        negative_coords = []
        if image is not None:
            img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            red_mask = cv2.inRange(img_bgr, neg_color, neg_color)
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours[:max_regions]:
                x, y, w, h = cv2.boundingRect(cnt)
                if w <= 0 or h <= 0:
                    continue
                
                for _ in range(points_per_region):
                    px = random.randint(x, x + w - 1)
                    py = random.randint(y, y + h - 1)
                    if cv2.pointPolygonTest(cnt, (float(px), float(py)), False) >= 0:
                        negative_coords.append({"x": int(px), "y": int(py)})
                        
        negative_coords_str = json.dumps(negative_coords)
        
        return (positive_coords_str, negative_coords_str)

class StrFormat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "format": ("STRING", {"default": "", "multiline": True}),
                "value1": ("STRING", {"default": ""}),
                "value2": ("STRING", {"default": ""}),
                "value3": ("STRING", {"default": ""}),
                "value4": ("STRING", {"default": ""}),
                "value5": ("STRING", {"default": ""}),
                "value6": ("STRING", {"default": ""}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "main"
    CATEGORY = 'lhyNodes/String'
    
    def main(self, format, value1, value2, value3, value4, value5, value6):
        return (format.format(value1, value2, value3, value4, value5, value6),)

class StrFormatAdv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "format": ("STRING", {"default": "", "multiline": True}),
                "value1": ("STRING", {"default": ""}),
                "switch1": ("BOOLEAN", {"default": True}),
                "value2": ("STRING", {"default": ""}),
                "switch2": ("BOOLEAN", {"default": True}),
                "value3": ("STRING", {"default": ""}),
                "switch3": ("BOOLEAN", {"default": True}),
                "value4": ("STRING", {"default": ""}),
                "switch4": ("BOOLEAN", {"default": True}),
                "value5": ("STRING", {"default": ""}),
                "switch5": ("BOOLEAN", {"default": True}),
                "value6": ("STRING", {"default": ""}),
                "switch6": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "main"
    CATEGORY = 'lhyNodes/String'
    
    def main(self, format, value1, switch1, value2, switch2, value3, switch3, value4, switch4, value5, switch5, value6, switch6):
        v1 = value1 if switch1 else ""
        v2 = value2 if switch2 else ""
        v3 = value3 if switch3 else ""
        v4 = value4 if switch4 else ""
        v5 = value5 if switch5 else ""
        v6 = value6 if switch6 else ""
        return (format.format(v1, v2, v3, v4, v5, v6),)

class CSVRandomPicker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "csv_string": ("STRING", {
                    "multiline": True,
                    "default": "apple,banana,cat,dog"
                }),
                "count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 1000
                }),
                "separator": ("STRING", {
                    "default": ","
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1125899906842624
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "pick_random_items"
    CATEGORY = "lhyNodes/String"
    
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return True
    
    def pick_random_items(self, csv_string, count, separator, seed):
        items = [item.strip() for item in csv_string.split(separator) if item.strip()]
        if not items:
            return ("",)
        
        actual_count = min(count, len(items))
        
        rng = random.Random()
        rng.seed(seed)
        
        selected_items = rng.sample(items, actual_count)
        result = separator.join(selected_items)
        return (result,)

class CSVRandomPickerAdv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "csv_string": ("STRING", {
                    "multiline": True,
                    "default": "apple,banana,cat,dog"
                }),
                "min_count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 1000
                }),
                "max_count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 1000
                }),
                "input_separator": ("STRING", {
                    "default": ","
                }),
                "output_separator": ("STRING", {
                    "default": ","
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1125899906842624
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "pick_random_items"
    CATEGORY = "lhyNodes/String"
    
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return True
    
    def pick_random_items(self, csv_string, min_count, max_count, input_separator, output_separator, seed):
        items = [item.strip() for item in csv_string.split(input_separator) if item.strip()]
        if not items:
            return ("",)
        
        if min_count > max_count:
            raise RuntimeError('"max_count" must be greater than "min_count"!')
        
        _min_count = min(min_count, len(items))
        _max_count = min(max_count, len(items))
        actual_count =  random.randint(_min_count, _max_count)
        
        rng = random.Random()
        rng.seed(seed)
        
        selected_items = rng.sample(items, actual_count)
        result = output_separator.join(selected_items)
        return (result,)

class YoloFaceReformer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "batch_size": ("INT", {"default": 32, "min": 1, "max": 1024, "step": 1}),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "Whether to process the image sequence."}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "lhyNode/WanAnimate"
    DESCRIPTION = "Automatically reuse the previous detected face when none is found. Optimized for large inputs via batching."
    
    def process(self, images, threshold, batch_size, enabled):
        if not enabled:
            return (images,)
        
        B, H, W, C = images.shape
        model = YOLO(os.path.join(current_dir, "models", "yolov8n-face.pt"))
        out_tensor = torch.empty((B, H, W, C))
        
        last_valid_frame = images[0] 
        invalid_face_count = 0
        
        for i in cqdm(range(0, B, batch_size)):
            batch_end = min(i + batch_size, B)
            image_batch = images[i:batch_end]
            results_batch = model(image_batch.permute(0, 3, 1, 2), conf=threshold, verbose=False)
            
            for j, result in enumerate(results_batch):
                idx = i + j
                has_detection = len(result.boxes) > 0
                
                if has_detection:
                    last_valid_frame = images[idx]
                    out_tensor[idx] = images[idx]
                else:
                    invalid_face_count += 1
                    out_tensor[idx] = last_valid_frame
            
        if invalid_face_count > 0:
            print(f"YoloFaceReformer: {invalid_face_count} frames missing faces, replaced with previous valid frames.")
            
        return (out_tensor,)

class PoseReformer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "Whether to process the image sequence."}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "lhyNode/WanAnimate"
    DESCRIPTION = "Automatically reuse the previous detected pose when none is found in current frame."
    
    def process(self, images, enabled):
        if not enabled:
            return (images,)
        
        poses = []
        for i, image in enumerate(images):
            if torch.max(image) != 0.0 or i == 0:
                poses.append(image.unsqueeze(0))
            else:
                poses.append(poses[-1])
        final_poses = torch.cat(poses, dim=0)
        return (final_poses,)

class CudaDevicePatcher:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type,),
                "device": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = (any_type, "STRING")
    RETURN_NAMES = ("any", "original")
    OUTPUT_NODE = True
    FUNCTION = "main"
    CATEGORY = "lhyNode/utils"
    
    def main(self, any, device):
        ori = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        print(f'[CUDA_VISIBLE_DEVICES] set to "{device}"')
        return (any, ori)

class noneNode:
    @classmethod
    def INPUT_TYPES(s):
        return {}
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("None",)
    FUNCTION = "main"
    CATEGORY = "lhyNode/utils"
    
    def main(self):
        return (None,)
    
class queueHandler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trigger": (any_type,),
                "any": (any_type,),
            }
        }
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    FUNCTION = "main"
    CATEGORY = "lhyNode/utils"
    
    def main(self, trigger, any):
        return (any,)

class GrowMask_lhy:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand_by": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "batch_size": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1}),
                "device": (["cpu", "gpu"], {"default": "cpu"}),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "process"
    CATEGORY = 'lhyNode/masking'
    
    def process(self, mask, expand_by, tapered_corners, batch_size, device):
        if expand_by == 0:
            return (mask,)
        
        B, H, W = mask.shape
        target_device = torch.device("cuda") if device == "gpu" else torch.device("cpu")
        final_mask = torch.empty((B, H, W), dtype=torch.float32, device="cpu")
        
        abs_expand = abs(expand_by)
        kernel_size = 2 * abs_expand + 1
        padding = abs_expand
        
        is_erosion = expand_by < 0
        
        for i in cqdm(range(0, B, batch_size)):
            chunk = mask[i:i + batch_size].to(target_device)
            chunk = chunk.unsqueeze(1)
            
            if is_erosion:
                chunk = 1.0 - chunk
                
            if not tapered_corners:
                processed = F.max_pool2d(chunk, kernel_size=kernel_size, stride=1, padding=padding)
            else:
                y, x = torch.meshgrid(torch.linspace(-1, 1, kernel_size), torch.linspace(-1, 1, kernel_size), indexing='ij')
                kernel_weight = (torch.abs(x) + torch.abs(y) <= 1.0).float().to(target_device)
                kernel_weight = kernel_weight.unsqueeze(0).unsqueeze(0)
                
                processed = F.conv2d(chunk, kernel_weight, padding=padding)
                processed = (processed > 0).float()
                
            if is_erosion:
                processed = 1.0 - processed
                
            final_mask[i:i + batch_size] = processed.squeeze(1).cpu()
            
        return (final_mask,)

class DrawMaskOnImage_lhy:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE", ),
                    "mask": ("MASK", ),
                    "color": ("STRING", {"default": "0, 0, 0"}),
                  },
                  "optional": {
                    "device": (["cpu", "gpu"], {"default": "cpu"}),
                }
        }
    
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "lhyNode/masking"
    
    def process(self, image, mask, color, device="cpu"):
        target_device = torch.device("cuda") if device == "gpu" else torch.device("cpu")
        
        B, H, W, C = image.shape
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        BM, HM, WM = mask.shape
        
        color = color.strip()
        color_values = []
        if color.startswith('#'):
            hex_color = color.lstrip('#')
            if len(hex_color) in [3, 4]:
                color_values = [int(c*2, 16) / 255.0 for c in hex_color]
            elif len(hex_color) in [6, 8]:
                color_values = [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4, 6) if i < len(hex_color)]
        else:
            color_values = [float(x.strip()) / 255.0 if float(x.strip()) > 1.0 else float(x.strip()) for x in color.split(",")]
            
        rgb = torch.tensor(color_values[:3], dtype=torch.float32, device=target_device)
        alpha_val = color_values[3] if len(color_values) == 4 else 1.0
        out_tensor = torch.empty((B, H, W, C), dtype=torch.float32, device="cpu")
        
        if HM != H or WM != W:
            mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='bilinear').squeeze(1)
            
        for i in cqdm(range(B)):
            curr_img = image[i].to(target_device)
            curr_mask = mask[i % BM].to(target_device)
            blend_factor = (curr_mask.unsqueeze(-1) * alpha_val)
            
            if C == 4:
                img_rgb = curr_img[..., :3]
                img_a = curr_img[..., 3:]
                out_rgb = img_rgb * (1.0 - blend_factor) + rgb * blend_factor
                out_a = torch.maximum(img_a, blend_factor)
                res = torch.cat((out_rgb, out_a), dim=-1)
            else:
                res = curr_img * (1.0 - blend_factor) + rgb * blend_factor
                
            out_tensor[i] = res.cpu()
            del curr_img, curr_mask, res
            
        return (out_tensor, )
    
class BlockifyMask_lhy:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "masks": ("MASK",),
                    "block_size": ("INT", {"default": 32, "min": 8, "max": 512, "step": 1}),
                },
                "optional": {
                    "device": (["cpu", "gpu"], {"default": "cpu"}),
                }
        }
    
    RETURN_TYPES = ("MASK", )
    RETURN_NAMES = ("mask",)
    FUNCTION = "process"
    CATEGORY = "lhyNode/masking"
    
    def process(self, masks, block_size, device="cpu"):
        B, H, W = masks.shape
        target_device = torch.device("cuda") if device == "gpu" else torch.device("cpu")
        
        result_masks = torch.zeros((B, H, W), dtype=torch.float32, device="cpu")
        
        for i in cqdm(range(B), desc="BlockifyMask"):
            m = masks[i].to(target_device)
            mask_bool = m > 0
            
            if not mask_bool.any():
                continue
            
            nz = torch.nonzero(mask_bool)
            y_min, x_min = nz.min(dim=0)[0]
            y_max, x_max = nz.max(dim=0)[0]
            
            bbox_h = y_max - y_min + 1
            bbox_w = x_max - x_min + 1
            h_div = max(1, bbox_h // block_size)
            w_div = max(1, bbox_w // block_size)
            region = m[y_min:y_max+1, x_min:x_max+1].unsqueeze(0).unsqueeze(0)
            
            kw = bbox_w // w_div
            kh = bbox_h // h_div
            
            pooled = F.max_pool2d(region, kernel_size=(kh, kw), stride=(kh, kw))
            block_region = F.interpolate(pooled, size=(bbox_h, bbox_w), mode='nearest').squeeze()
            result_masks[i, y_min:y_max+1, x_min:x_max+1] = block_region.cpu()
            del m, region, pooled, block_region
            
        return (result_masks,)
    
class WanAnimateMaskPreprocessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "expand_by": ("INT", {"default": 10, "min": -255, "max": 255, "step": 1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "block_size": ("INT", {"default": 32, "min": 0, "max": 512, "step": 1}),
                "color": ("STRING", {"default": "0, 0, 0"}),
                "batch_size": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1}),
                "device": (["cpu", "gpu"], {"default": "gpu"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "process"
    CATEGORY = 'lhyNode/masking'
    
    def process(self, image, mask, expand_by, tapered_corners, block_size, color, batch_size, device):
        B, H, W, C = image.shape
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        BM, HM, WM = mask.shape
        
        target_device = torch.device("cuda") if device == "gpu" and torch.cuda.is_available() else torch.device("cpu")

        color = color.strip()
        if color.startswith('#'):
            hex_c = color.lstrip('#')
            if len(hex_c) in [3, 4]:
                color_vals = [int(c*2, 16) / 255.0 for c in hex_c]
            else:
                color_vals = [int(hex_c[idx:idx+2], 16) / 255.0 for idx in (0, 2, 4, 6) if idx < len(hex_c)]
        else:
            color_vals = [float(x.strip()) / 255.0 if float(x.strip()) > 1.0 else float(x.strip()) for x in color.split(",")]
            
        rgb_fill = torch.tensor(color_vals[:3], dtype=torch.float32, device=target_device)
        alpha_mult = color_vals[3] if len(color_vals) == 4 else 1.0
        final_images = torch.empty((B, H, W, C), dtype=torch.float32, device="cpu")
        final_masks = torch.empty((B, H, W), dtype=torch.float32, device="cpu")
        
        if HM != H or WM != W:
            mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='bilinear').squeeze(1)
        
        abs_expand = abs(expand_by)
        kernel_size = 2 * abs_expand + 1
        is_erosion = expand_by < 0
        
        for i in cqdm(range(0, B, batch_size)):
            curr_batch_size = min(batch_size, B - i)
            img_chunk = image[i:i+curr_batch_size].to(target_device)
            mask_indices = [(idx % BM) for idx in range(i, i + curr_batch_size)]
            mask_chunk = mask[mask_indices].to(target_device).unsqueeze(1)
            
            if expand_by != 0:
                if is_erosion:
                    mask_chunk = 1.0 - mask_chunk
                
                if not tapered_corners:
                    mask_chunk = F.max_pool2d(mask_chunk, kernel_size=kernel_size, stride=1, padding=abs_expand)
                else:
                    y, x = torch.meshgrid(torch.linspace(-1, 1, kernel_size), torch.linspace(-1, 1, kernel_size), indexing='ij')
                    kernel_weight = (torch.abs(x) + torch.abs(y) <= 1.0).float().to(target_device).unsqueeze(0).unsqueeze(0)
                    mask_chunk = (F.conv2d(mask_chunk, kernel_weight, padding=abs_expand) > 0).float()
                    
                if is_erosion: mask_chunk = 1.0 - mask_chunk
                
            if block_size > 0:
                h_div, w_div = max(1, H // block_size), max(1, W // block_size)
                pooled = F.max_pool2d(mask_chunk, kernel_size=(H//h_div, W//w_div), stride=(H//h_div, W//w_div))
                mask_chunk = F.interpolate(pooled, size=(H, W), mode='nearest')
                
            blend_factor = mask_chunk.permute(0, 2, 3, 1) * alpha_mult
            
            if C == 4:
                img_rgb = img_chunk[..., :3]
                img_a = img_chunk[..., 3:]
                out_rgb = img_rgb * (1.0 - blend_factor) + rgb_fill * blend_factor
                out_a = torch.maximum(img_a, blend_factor)
                res_img = torch.cat((out_rgb, out_a), dim=-1)
            else:
                res_img = img_chunk * (1.0 - blend_factor) + rgb_fill * blend_factor
            
            final_images[i:i+curr_batch_size] = res_img.cpu()
            final_masks[i:i+curr_batch_size] = mask_chunk.squeeze(1).cpu()
            
            del img_chunk, mask_chunk, res_img, blend_factor
            if 'pooled' in locals():
                del pooled
            
        return (final_images, final_masks)

class ImageOverlay_lhy:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "overlay_image": ("IMAGE",),
                "blend_mode": ([
                    "normal", "multiply", "screen", "overlay", "soft_light", 
                    "hard_light", "difference", "exclusion", "linear_dodge_add", 
                    "linear_burn", "subtract"
                ], {"default": "normal"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scale_to_fill": ("BOOLEAN", {"default": True, "tooltip": "Scale the overlay image to fill the source."}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "optional_mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "lhyNode/image"
    
    def process(self, source_image, overlay_image, blend_mode, opacity, scale_to_fill, invert_mask, optional_mask=None):
        # 获取基础维度
        B, H, W, C = source_image.shape
        OB, OH, OW, OC = overlay_image.shape
        
        # 确定计算设备
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # 【内存优化 1】：预分配 CPU 容器，避免 torch.stack 导致的内存翻倍
        # 使用 source_image.dtype 保持精度一致
        out_tensor = torch.empty((B, H, W, C), dtype=source_image.dtype, device="cpu")
        
        # --- 缩放逻辑预处理 ---
        if scale_to_fill:
            scale = max(H / OH, W / OW)
            new_h, new_w = int(OH * scale), int(OW * scale)
        else:
            new_h, new_w = OH, OW
            
        # 【内存优化 2】：如果叠加图只有一张，预先在显存中缩放好，避免在循环中重复计算
        # 如果叠加图是序列，则在循环内逐帧处理以节省显存
        if OB == 1:
            ovl_ready = overlay_image.permute(0, 3, 1, 2)
            if scale_to_fill:
                ovl_ready = F.interpolate(ovl_ready, size=(new_h, new_w), mode="bilinear", align_corners=False)
            ovl_ready = ovl_ready.permute(0, 2, 3, 1).to(device) # [1, new_h, new_w, OC]
            
            if optional_mask is not None:
                mask_ready = optional_mask.unsqueeze(1)
                mask_ready = F.interpolate(mask_ready, size=(new_h, new_w), mode="bilinear").squeeze(1).to(device)
                if invert_mask:
                    mask_ready = 1.0 - mask_ready
            else:
                mask_ready = None
        else:
            # 多帧水印不在这里处理，进循环
            ovl_ready = None
            mask_ready = None
            
        # 计算居中偏移
        y_offset = (H - new_h) // 2
        x_offset = (W - new_w) // 2
        
        for i in cqdm(range(B)):
            # 获取当前帧底图（送入显存）
            base_frame = source_image[i].to(device)
            
            # --- 准备当前帧叠加图 ---
            if ovl_ready is not None:
                # 单图模式：直接引用预处理好的
                curr_ovl = ovl_ready[0]
                curr_mask = mask_ready[0] if mask_ready is not None else None
            else:
                # 多图模式：逐帧缩放处理
                raw_ovl = overlay_image[i % OB].unsqueeze(0).permute(0, 3, 1, 2)
                if scale_to_fill:
                    raw_ovl = F.interpolate(raw_ovl, size=(new_h, new_w), mode="bilinear", align_corners=False)
                curr_ovl = raw_ovl.permute(0, 2, 3, 1).squeeze(0).to(device)
                
                if optional_mask is not None:
                    raw_mask = optional_mask[i % optional_mask.shape[0]].unsqueeze(0).unsqueeze(0)
                    curr_mask = F.interpolate(raw_mask, size=(new_h, new_w), mode="bilinear").squeeze().to(device)
                    if invert_mask:
                        curr_mask = 1.0 - curr_mask
                else:
                    curr_mask = None
                    
            # --- 提取 Alpha 和 RGB ---
            if OC == 4:
                ovl_rgb = curr_ovl[..., :3]
                ovl_a = curr_ovl[..., 3] * opacity
            else:
                ovl_rgb = curr_ovl
                ovl_a = torch.ones((new_h, new_w), device=device) * opacity
                
            if curr_mask is not None:
                ovl_a = ovl_a * curr_mask
                
            # --- 裁剪区域计算 (居中对齐) ---
            # y1,y2 为底图区域; wy1,wy2 为叠加图区域
            y1, y2 = max(0, y_offset), min(H, y_offset + new_h)
            x1, x2 = max(0, x_offset), min(W, x_offset + new_w)
            wy1, wy2 = max(0, -y_offset), min(new_h, H - y_offset)
            wx1, wx2 = max(0, -x_offset), min(new_w, W - x_offset)
            
            if y1 < y2 and x1 < x2:
                target_area = base_frame[y1:y2, x1:x2, :3]
                source_area = ovl_rgb[wy1:wy2, wx1:wx2, :]
                alpha_area = ovl_a[wy1:wy2, wx1:wx2].unsqueeze(-1)
                
                # 执行混合
                blended = self.apply_blend(target_area, source_area, blend_mode)
                
                # Alpha 混合合成
                combined_rgb = blended * alpha_area + target_area * (1.0 - alpha_area)
                
                # 更新底图
                res_frame = base_frame.clone()
                res_frame[y1:y2, x1:x2, :3] = combined_rgb
                
                # 如果底图是 RGBA，则计算 Alpha 复合
                if C == 4:
                    target_a = base_frame[y1:y2, x1:x2, 3:]
                    res_frame[y1:y2, x1:x2, 3:] = torch.clamp(target_a + alpha_area * (1.0 - target_a), 0, 1)
                    
                # 【内存优化 3】：写回 CPU，并立即释放显存
                out_tensor[i] = res_frame.cpu()
            else:
                out_tensor[i] = source_image[i]
                
            del base_frame, curr_ovl, curr_mask
            
        return (out_tensor,)
    
    def apply_blend(self, b, s, mode):
        if mode == "normal": return s
        if mode == "multiply": return b * s
        if mode == "screen": return 1.0 - (1.0 - b) * (1.0 - s)
        if mode == "overlay":
            return torch.where(b < 0.5, 2.0 * b * s, 1.0 - 2.0 * (1.0 - b) * (1.0 - s))
        if mode == "difference":
            return torch.abs(b - s)
        if mode == "exclusion":
            return b + s - 2.0 * b * s
        if mode == "linear_dodge_add":
            return torch.clamp(b + s, 0, 1)
        if mode == "linear_burn":
            return torch.clamp(b + s - 1.0, 0, 1)
        if mode == "subtract":
            return torch.clamp(b - s, 0, 1)
        if mode == "soft_light":
            return (1.0 - 2.0 * s) * b**2 + 2.0 * s * b
        return s

NODE_CLASS_MAPPINGS = {
    "detailerKSamplerSchedulerFallback": detailerKSamplerSchedulerFallback,
    "effKSamplerSchedulerFallback": effKSamplerSchedulerFallback,
    "KSamplerSchedulerFallback": KSamplerSchedulerFallback,
    "KSamplerConfig": KSamplerConfig,
    "MaskToSAMCoords": MaskToSAMCoords,
    "MaskToSAMCoordsV2": MaskToSAMCoordsV2,
    "StrFormat": StrFormat,
    "StrFormatAdv": StrFormatAdv,
    "CSVRandomPicker": CSVRandomPicker,
    "CSVRandomPickerAdv": CSVRandomPickerAdv,
    "YoloFaceReformer": YoloFaceReformer,
    "PoseReformer": PoseReformer,
    "CudaDevicePatcher": CudaDevicePatcher,
    "noneNode": noneNode,
    "queueHandler": queueHandler,
    "GrowMask_lhy": GrowMask_lhy,
    "DrawMaskOnImage_lhy": DrawMaskOnImage_lhy,
    "BlockifyMask_lhy": BlockifyMask_lhy,
    "WanAnimateMaskPreprocessor": WanAnimateMaskPreprocessor,
    "ImageOverlay_lhy": ImageOverlay_lhy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "detailerKSamplerSchedulerFallback": "Scheduler Fallback (Detailer)",
    "effKSamplerSchedulerFallback": "Scheduler Fallback (Efficient)",
    "KSamplerSchedulerFallback": "Scheduler Fallback (KSampler)",
    "KSamplerConfig": "KSampler Config",
    "MaskToSAMCoords": "Mask to Coordinates (SAM2)",
    "MaskToSAMCoordsV2": "Mask to Coordinates V2 (SAM2)",
    "StrFormat": "String Format",
    "StrFormatAdv": "String Format (Advanced)",
    "CSVRandomPicker": "CSV RandomPicker",
    "CSVRandomPickerAdv": "CSV RandomPicker (Advanced)",
    "YoloFaceReformer": "WanAnimate Face Reformer",
    "PoseReformer": "WanAnimate Pose Reformer",
    "CudaDevicePatcher": "Set Cuda Device",
    "noneNode": "None",
    "queueHandler": "Queue Handler",
    "GrowMask_lhy": "Grow Mask",
    "DrawMaskOnImage_lhy": "Draw Mask On Image",
    "BlockifyMask_lhy": "Blockify Mask",
    "WanAnimateMaskPreprocessor": "WanAnimate Mask Preprocessor",
    "ImageOverlay_lhy": "Image Overlay",
}