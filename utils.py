import os
import sys
import math
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

import comfy.utils
import folder_paths

class cqdm:
    def __init__(self, iterable=None, total=None, desc="Processing", disable=False, **kwargs):
        self.iterable = iterable
        self.total = total
        self.desc = desc
        
        if iterable is not None and total is None:
            try:
                self.total = len(iterable)
            except (TypeError, AttributeError):
                self.total = None
                
        self.pbar = comfy.utils.ProgressBar(self.total) if self.total is not None else None
        
        self.tqdm = tqdm(
            iterable=self.iterable, 
            total=self.total, 
            desc=self.desc, 
            disable=disable,
            dynamic_ncols=True,
            file=sys.stdout,
            **kwargs 
        )
        
    def __iter__(self):
        if self.tqdm is None:
            return
        for item in self.tqdm:
            if self.pbar:
                self.pbar.update(1)
            yield item
            
    def update(self, n=1):
        if self.tqdm:
            self.tqdm.update(n)
        if self.pbar:
            self.pbar.update(n)
            
    def set_description(self, desc):
        if self.tqdm:
            self.tqdm.set_description(desc)
            
    def set_postfix(self, *args, **kwargs):
        if self.tqdm:
            self.tqdm.set_postfix(*args, **kwargs)
            
    def close(self):
        if self.tqdm is not None:
            self.tqdm.close()
            
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def __len__(self):
        return self.total

def create_batch_preview(images, output_dir, file_name_prefix, max_size=2000):
    pil_images = []
    
    if isinstance(images, list):
        pil_images = images
    elif isinstance(images, Image.Image):
        pil_images = [images]
    else:
         return []

    if not pil_images:
        return []
    
    _max_size = 0xFFFFFF if len(pil_images) == 1 else max_size
    
    filename = f"{file_name_prefix}.webp"
    full_path = os.path.join(output_dir, filename)
    
    input_root = folder_paths.get_input_directory()
    try:
        subfolder = os.path.relpath(output_dir, input_root)
        if subfolder == ".": subfolder = ""
    except ValueError:
        subfolder = ""
        output_dir = folder_paths.get_temp_directory()
        full_path = os.path.join(output_dir, filename)

    if os.path.exists(full_path):
        return [{"filename": filename, "subfolder": subfolder, "type": "input"}]

    count = len(pil_images)
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)

    if len(pil_images) > 0:
        ref_w, ref_h = pil_images[0].size
    else:
        ref_w, ref_h = 512, 512
    
    scale_w = _max_size / (cols * ref_w)
    scale_h = _max_size / (rows * ref_h)
    scale = min(1.0, scale_w, scale_h)

    thumb_w = max(1, int(ref_w * scale))
    thumb_h = max(1, int(ref_h * scale))

    canvas_w = cols * thumb_w
    canvas_h = rows * thumb_h
    canvas = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))

    for idx, img in enumerate(pil_images):
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        
        img_thumb = ImageOps.pad(img, (thumb_w, thumb_h), method=Image.NEAREST, color=(0, 0, 0, 0), centering=(0.5, 0.5))
        
        c = idx % cols
        r = idx // cols
        x = c * thumb_w
        y = r * thumb_h
        
        canvas.paste(img_thumb, (x, y))

    canvas.save(full_path, format="WEBP", quality=80)

    return [{"filename": filename, "subfolder": subfolder, "type": "input"}]