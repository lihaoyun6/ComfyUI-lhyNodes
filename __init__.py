import os
import io
import math
import zipfile
import folder_paths
from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageFont
from aiohttp import web
from server import PromptServer

from .nodes.nodes import NODE_CLASS_MAPPINGS as MAIN_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MAIN_NAME_MAPPINGS
from .nodes.morse_nodes import NODE_CLASS_MAPPINGS as MORSE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MORSE_NAME_MAPPINGS
from .nodes.file_nodes import NODE_CLASS_MAPPINGS as FILE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as FILE_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS.update(MAIN_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(MAIN_NAME_MAPPINGS)
NODE_CLASS_MAPPINGS.update(MORSE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(MORSE_NAME_MAPPINGS)
NODE_CLASS_MAPPINGS.update(FILE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(FILE_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

routes = PromptServer.instance.routes
current_dir = os.path.dirname(os.path.realpath(__file__))

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

@routes.post("/batch_preview/gen_batch")
async def generate_batch_preview_endpoint(request):
	data = await request.json()
	batch_folder = data.get("batch_folder")
	
	input_dir = folder_paths.get_input_directory()
	target_dir = os.path.join(input_dir, "batch", batch_folder)
	
	if not os.path.exists(target_dir):
		return web.json_response({"error": "Folder not found"}, status=404)
	
	preview_filename = "__preview__grid.webp"
	preview_path = os.path.join(target_dir, preview_filename)
	
	valid_ext = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
	all_files = [f for f in os.listdir(target_dir) if os.path.splitext(f)[1].lower() in valid_ext and not f.startswith("__preview__")]
	files = sorted(all_files, key=lambda x: os.path.getmtime(os.path.join(target_dir, x)), reverse=True)
	
	if not files:
		return web.json_response({"error": "No images found"}, status=404)
	
	should_regenerate = True
	if os.path.exists(preview_path):
		preview_mtime = os.path.getmtime(preview_path)
		latest_file_mtime = max([os.path.getmtime(os.path.join(target_dir, f)) for f in files])
		
		if preview_mtime > latest_file_mtime:
			should_regenerate = False

	if not should_regenerate:
		subfolder = os.path.relpath(target_dir, input_dir)
		return web.json_response({
			"filename": preview_filename,
			"subfolder": subfolder,
			"type": "input"
		})
	
	try:
		if os.path.exists(preview_path):
			os.remove(preview_path)
	except Exception as e:
		print(f"Warning: Could not remove old preview: {e}")
		
	MAX_PREVIEWS = 9
	total_count = len(files)
	has_more = total_count > MAX_PREVIEWS
	files_to_process = files[:MAX_PREVIEWS]
	
	pil_images = []
	for idx, filename in enumerate(files_to_process):
		try:
			img_path = os.path.join(target_dir, filename)
			img = Image.open(img_path)
			img = ImageOps.exif_transpose(img)
			img = img.convert("RGBA")
			
			if has_more and idx == len(files_to_process) - 1:
				blur_radius = min(img.size) // 20
				img = img.filter(ImageFilter.GaussianBlur(radius=max(5, blur_radius)))
				
				draw = ImageDraw.Draw(img)
				text = f"+{total_count - MAX_PREVIEWS + 1}"
				font_size = int(img.height * 0.2)
				font = ImageFont.truetype(os.path.join(current_dir, "res","fonts", "Alibaba-PuHuiTi-Heavy.ttf"), font_size)
			
				if hasattr(draw, 'textbbox'):
					bbox = draw.textbbox((0, 0), text, font=font)
					tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
				else:
					tw, th = draw.textsize(text, font=font)
				
				tx = (img.width - tw) / 2
				ty = (img.height - th) / 2
				stroke_w = max(3, font_size // 10)
				try:
					draw.text((tx, ty), text, font=font, fill="black", stroke_width=stroke_w, stroke_fill="white")
				except TypeError:
					for ox, oy in [(-3,-3), (3,-3), (-3,3), (3,3), (0,-3), (0,3), (-3,0), (3,0)]:
						draw.text((tx+ox, ty+oy), text, font=font, fill="white")
					draw.text((tx, ty), text, font=font, fill="black")
			pil_images.append(img)
		except Exception as e:
			print(f"Error processing preview image: {e}")
			continue
		
	if not pil_images:
		return web.json_response({"error": "All images failed to load"}, status=500)

	preview_info = create_batch_preview(
		pil_images, 
		output_dir=target_dir, 
		file_name_prefix="__preview__grid",
		max_size=1200
	)
	del pil_images
	
	return web.json_response(preview_info[0])