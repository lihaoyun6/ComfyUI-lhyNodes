import os
import zipfile
import io
import folder_paths
from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageFont
from aiohttp import web
from server import PromptServer
from .utils import create_batch_preview

from .nodes import NODE_CLASS_MAPPINGS as MAIN_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MAIN_NAME_MAPPINGS
from .morse_nodes import NODE_CLASS_MAPPINGS as MORSE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MORSE_NAME_MAPPINGS
from .file_nodes import NODE_CLASS_MAPPINGS as FILE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as FILE_NAME_MAPPINGS

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
	files = sorted([f for f in os.listdir(target_dir) 
					if os.path.splitext(f)[1].lower() in valid_ext 
					and not f.startswith("__preview__")])
	
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
			img = img.convert("RGB")
			
			if has_more and idx == len(files_to_process) - 1:
				blur_radius = min(img.size) // 20
				img = img.filter(ImageFilter.GaussianBlur(radius=max(5, blur_radius)))
				
				draw = ImageDraw.Draw(img)
				text = f"+{total_count - MAX_PREVIEWS + 1}"
				font_size = int(img.height * 0.2)
				font = ImageFont.truetype(os.path.join(current_dir, "font", "Alibaba-PuHuiTi-Heavy.ttf"), font_size)
			
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