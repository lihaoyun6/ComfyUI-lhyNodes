# 
<p align="center">
<img src="https://avatars.githubusercontent.com/u/16348097" width="200" height="200" />
<h1 align="center">ComfyUI-lhyNodes</h1>
<h3 align="center">欢迎使用我的 ComfyUI 增效节点合集<br>[<a href="./README.md">📃English</a>]
</p>
<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./res/img/preview_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./res/img/preview.png">
  <img alt="Screenshots" src="./res/img/preview.png"/>
</picture>
</p>

## 安装指南
两种安装方式:

⭐️ 在 `ComfyUI-Manager` 中搜索 `lhyNodes` 并安装.  

或者

- 使用命令行手动安装:

	```bash
  cd ComfyUI/custom_nodes
  git clone https://github.com/lihaoyun6/ComfyUI-lhyNodes.git
  python -m pip install -r ComfyUI-lhyNodes/requirements.txt
  ``` 

## 节点列表

### Load Image Batch
> 通过浏览器直接上传多张图片，并生成图像批次或图像列表。

### Image Batch to Images
> 将 `image_batch` 对象转换为图像批次。

### Image Batch to Image List
> 将 `image_batch` 对象转换为图像列表。

### Load Image from ZIP
> 从上传的 ZIP 文件中读取图像并生成 `image_batch` 对象。

### Save Image as ZIP
> 将图像（可选附加文本）保存为 ZIP 文件并提供下载。

### String Format
> 根据变量对包含 `{}` 占位符的文本进行格式化并输出。

### String Format (Advanced)
> 根据变量对包含 `{}` 占位符的文本进行格式化并输出。  
> 并且可以单独启用或禁用任意变量。

### CSV Random Picker
> 根据随机种子从 CSV 字符串中随机选择元素。

### CSV Random Picker (Advanced)
> 根据随机种子从 CSV 字符串中随机选择元素，并提供更多选项。

### Queue Handler
> 使用触发器来控制任意节点的执行时机。

### None
> 什么也不做，仅输出 `None`。

### Set CUDA Device
> 在运行时修改环境变量 `CUDA_VISIBLE_DEVICES` 的值。

### Image Overlay
> 将一张图像叠加到另一张图像之上。

### Grow Mask
> 以极快的速度对输入遮罩进行扩展。

### Blockify Mask / Draw Mask On Image
> 功能与名称所示一致。

### WanAnimate Mask Preprocessor
> 一体化的 Wan Animate 遮罩预处理节点。

### WanAnimate Face Reformer
> 从人脸帧序列中移除无脸帧，并修复序列的一致性。

### WanAnimate Pose Reformer
> 用于移除任意帧序列中的全黑帧，并修复其一致性。

### WanAnimate Best Frame Window
> 根据总帧数计算最优的分段帧窗口大小。

### Mask to Coordinates
> 通过在 ComfyUI 的遮罩编辑器中画点来生成 SAM 条件。

### Mask to Coordinates V2
> 使用遮罩画笔和彩色画笔生成正向与负向的 SAM 条件。

## 致谢
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous

