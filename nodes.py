import random
import numpy as np
import cv2
import random
import json
import torch
import os

from nodes import MAX_RESOLUTION
import comfy.samplers

RES4 = False
plugin_path = os.path.join(os.path.dirname(__file__), "..", "RES4LYF")
if os.path.exists(plugin_path):
    RES4 = True

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
    CATEGORY = "utils"
    
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
    CATEGORY = "utils"
    
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
    CATEGORY = "utils"
    
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
    CATEGORY = "utils"
    
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
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_regions": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
                "points_per_region": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("coordinates",)
    FUNCTION = "convert"
    CATEGORY = "mask"
    
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
    CATEGORY = 'utils/mxToolkit'
    
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
    CATEGORY = 'utils/mxToolkit'
    
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
    CATEGORY = "Custom/Utils"
    
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
    CATEGORY = "Custom/Utils"
    
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

NODE_CLASS_MAPPINGS = {
    "detailerKSamplerSchedulerFallback": detailerKSamplerSchedulerFallback,
    "effKSamplerSchedulerFallback": effKSamplerSchedulerFallback,
    "KSamplerSchedulerFallback": KSamplerSchedulerFallback,
    "KSamplerConfig": KSamplerConfig,
    "MaskToSAMCoords": MaskToSAMCoords,
    "StrFormat": StrFormat,
    "StrFormatAdv": StrFormatAdv,
    "CSVRandomPicker": CSVRandomPicker,
    "CSVRandomPickerAdv": CSVRandomPickerAdv,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "detailerKSamplerSchedulerFallback": "Scheduler Fallback (Detailer)",
    "effKSamplerSchedulerFallback": "Scheduler Fallback (Efficient)",
    "KSamplerSchedulerFallback": "Scheduler Fallback (KSampler)",
    "KSamplerConfig": "KSampler Config",
    "MaskToSAMCoords": "Mask to SAM2Coordinates",
    "StrFormat": "String Format",
    "StrFormatAdv": "String Format (Advanced)",
    "CSVRandomPicker": "CSV RandomPicker",
    "CSVRandomPickerAdv": "CSV RandomPicker (Advanced)",
}