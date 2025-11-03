## Image Quality Enhancer with CUDA

This project uses **stb_image.h** and **stb_image_write.h** from [nothings/stb](https://github.com/nothings/stb) to perform basic image processing tasks — reading, editing, and writing image files.

# 🖼️ Image Enhancer using STB and CUDA

This project uses **stb_image.h** and **stb_image_write.h** from [nothings/stb](https://github.com/nothings/stb) to perform basic image reading and writing, combined with **CUDA** for GPU-accelerated image enhancement operations such as sharpening, denoising, and contrast adjustment.

---

## ⚙️ Requirements

- **NVIDIA GPU** with CUDA support  
- **CUDA Toolkit** installed and configured in your environment  
- **stb_image.h** and **stb_image_write.h** (single-header libraries for image I/O)

## 📦 Dependencies

No external libraries are needed except for:
- [`stb_image.h`](https://github.com/nothings/stb/blob/master/stb_image.h)
- [`stb_image_write.h`](https://github.com/nothings/stb/blob/master/stb_image_write.h)

---

## ⚙️ HOW TO COMPILE AND RUN

```text
==============================================
HOW TO COMPILE AND RUN:
==============================================

1. Download stb_image libraries:
   wget https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
   wget https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h

2. Compile using nvcc:
   nvcc -o image_enhancer image_enhancer.cu -O3

3. Run:
   ./image_enhancer input.jpg output.png

4. With custom parameters:
   ./image_enhancer input.jpg output.png 2.0 50 1.5
   (sharpen=2.0, denoise=50, contrast=1.5)
==============================================


==============================================
EXAMPLE USAGE: 
==============================================

# Example 1: Standard enhancement
./image_enhancer my_photo.jpg result.png

# Example 2: Stronger sharpening (value 3.0)
./image_enhancer photo.jpg output.png 3.0 30 1.2

# Example 3: Heavy denoise (value 80)
./image_enhancer noisy_photo.jpg clean.jpg 1.0 80 1.0

# Example 4: Format conversion + enhancement
./image_enhancer image.png image.jpg
==============================================

