// image_enhancer.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Download dari: https://github.com/nothings/stb
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define TILE_WIDTH 16

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel untuk sharpening (unsharp mask)
__global__ void unsharpMaskKernel(unsigned char* input, unsigned char* output, 
                                   int width, int height, int channels, float amount) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Gaussian blur weights (5x5)
    const float gaussian[5][5] = {
        {0.003765, 0.015019, 0.023792, 0.015019, 0.003765},
        {0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
        {0.023792, 0.094907, 0.150342, 0.094907, 0.023792},
        {0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
        {0.003765, 0.015019, 0.023792, 0.015019, 0.003765}
    };
    
    for (int c = 0; c < channels; c++) {
        float blurred = 0.0f;
        float original = input[(y * width + x) * channels + c];
        
        // Apply Gaussian blur
        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {
                int ny = min(max(y + ky, 0), height - 1);
                int nx = min(max(x + kx, 0), width - 1);
                blurred += input[(ny * width + nx) * channels + c] * gaussian[ky + 2][kx + 2];
            }
        }
        
        // Unsharp mask formula
        float sharp = original + amount * (original - blurred);
        output[(y * width + x) * channels + c] = (unsigned char)fminf(fmaxf(sharp, 0.0f), 255.0f);
    }
}

// Kernel untuk noise reduction (bilateral filter)
__global__ void bilateralFilterKernel(unsigned char* input, unsigned char* output,
                                      int width, int height, int channels,
                                      float sigmaSpace, float sigmaColor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int radius = 5;
    float twoSigmaSpaceSq = 2.0f * sigmaSpace * sigmaSpace;
    float twoSigmaColorSq = 2.0f * sigmaColor * sigmaColor;
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        float wsum = 0.0f;
        float centerVal = input[(y * width + x) * channels + c];
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int ny = min(max(y + ky, 0), height - 1);
                int nx = min(max(x + kx, 0), width - 1);
                
                float val = input[(ny * width + nx) * channels + c];
                float spaceDist = kx * kx + ky * ky;
                float colorDist = (val - centerVal) * (val - centerVal);
                
                float weight = expf(-spaceDist / twoSigmaSpaceSq - colorDist / twoSigmaColorSq);
                sum += val * weight;
                wsum += weight;
            }
        }
        
        output[(y * width + x) * channels + c] = (unsigned char)(sum / wsum);
    }
}

// Kernel untuk peningkatan kontras
__global__ void contrastEnhanceKernel(unsigned char* input, unsigned char* output,
                                      int width, int height, int channels, float clipLimit) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    for (int c = 0; c < channels; c++) {
        float val = input[(y * width + x) * channels + c];
        float mean = 127.5f;
        float enhanced = mean + clipLimit * (val - mean);
        
        output[(y * width + x) * channels + c] = (unsigned char)fminf(fmaxf(enhanced, 0.0f), 255.0f);
    }
}

class ImageEnhancer {
private:
    unsigned char *d_input, *d_output, *d_temp;
    int width, height, channels;
    
public:
    ImageEnhancer(int w, int h, int c) : width(w), height(h), channels(c) {
        size_t imageSize = width * height * channels * sizeof(unsigned char);
        CUDA_CHECK(cudaMalloc(&d_input, imageSize));
        CUDA_CHECK(cudaMalloc(&d_output, imageSize));
        CUDA_CHECK(cudaMalloc(&d_temp, imageSize));
    }
    
    ~ImageEnhancer() {
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_temp);
    }
    
    void enhance(unsigned char* input, unsigned char* output,
                 float sharpenAmount = 1.5f,
                 float denoiseSpace = 3.0f, 
                 float denoiseColor = 75.0f,
                 float contrastFactor = 1.2f) {
        
        size_t imageSize = width * height * channels * sizeof(unsigned char);
        
        // Copy input image ke GPU
        CUDA_CHECK(cudaMemcpy(d_input, input, imageSize, cudaMemcpyHostToDevice));
        
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
        dim3 gridDim((width + TILE_WIDTH - 1) / TILE_WIDTH,
                     (height + TILE_WIDTH - 1) / TILE_WIDTH);
        
        printf("Memproses gambar di GPU...\n");
        
        // Step 1: Denoise
        printf("  [1/3] Mengurangi noise...\n");
        bilateralFilterKernel<<<gridDim, blockDim>>>(d_input, d_temp, width, height, channels,
                                                      denoiseSpace, denoiseColor);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Step 2: Sharpen
        printf("  [2/3] Meningkatkan ketajaman...\n");
        unsharpMaskKernel<<<gridDim, blockDim>>>(d_temp, d_output, width, height, channels,
                                                  sharpenAmount);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Step 3: Enhance contrast
        printf("  [3/3] Meningkatkan kontras...\n");
        contrastEnhanceKernel<<<gridDim, blockDim>>>(d_output, d_temp, width, height, channels,
                                                      contrastFactor);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy hasil kembali ke CPU
        CUDA_CHECK(cudaMemcpy(output, d_temp, imageSize, cudaMemcpyDeviceToHost));
        
        printf("Selesai!\n");
    }
};

int main(int argc, char** argv) {
    printf("==============================================\n");
    printf("   CUDA Image Quality Enhancer v1.0\n");
    printf("==============================================\n\n");
    
    // Cek argumen
    if (argc < 3) {
        printf("Cara penggunaan:\n");
        printf("  %s <input_gambar> <output_gambar>\n\n", argv[0]);
        printf("Contoh:\n");
        printf("  %s foto.jpg foto_enhanced.png\n", argv[0]);
        printf("  %s gambar.png hasil.jpg\n\n", argv[0]);
        printf("Parameter opsional:\n");
        printf("  %s <input> <output> <sharpen> <denoise> <contrast>\n", argv[0]);
        printf("  Contoh: %s foto.jpg hasil.png 2.0 50 1.5\n\n", argv[0]);
        return 1;
    }
    
    // Ambil parameter (dengan default values)
    float sharpenAmount = 1.5f;
    float denoiseAmount = 30.0f;
    float contrastFactor = 1.2f;
    
    if (argc >= 6) {
        sharpenAmount = atof(argv[3]);
        denoiseAmount = atof(argv[4]);
        contrastFactor = atof(argv[5]);
    }
    
    // Load gambar
    printf("Memuat gambar: %s\n", argv[1]);
    int width, height, channels;
    unsigned char* input = stbi_load(argv[1], &width, &height, &channels, 0);
    
    if (!input) {
        printf("ERROR: Gagal memuat gambar '%s'\n", argv[1]);
        printf("Pastikan file ada dan format didukung (JPG, PNG, BMP, dll)\n");
        return 1;
    }
    
    printf("  Ukuran: %d x %d pixels\n", width, height);
    printf("  Channel: %d (%s)\n", channels, 
           channels == 1 ? "Grayscale" : 
           channels == 3 ? "RGB" : 
           channels == 4 ? "RGBA" : "Unknown");
    printf("  Total pixels: %d\n\n", width * height);
    
    // Alokasi buffer output
    unsigned char* output = (unsigned char*)malloc(width * height * channels);
    if (!output) {
        printf("ERROR: Gagal alokasi memori\n");
        stbi_image_free(input);
        return 1;
    }
    
    // Proses dengan CUDA
    printf("Parameter enhancement:\n");
    printf("  Sharpness: %.2f\n", sharpenAmount);
    printf("  Denoise: %.2f\n", denoiseAmount);
    printf("  Contrast: %.2f\n\n", contrastFactor);
    
    ImageEnhancer enhancer(width, height, channels);
    enhancer.enhance(input, output, sharpenAmount, denoiseAmount / 10.0f, 
                     denoiseAmount, contrastFactor);
    
    // Simpan hasil
    printf("\nMenyimpan hasil ke: %s\n", argv[2]);
    int result = 0;
    
    // Deteksi format output dari ekstensi
    const char* ext = strrchr(argv[2], '.');
    if (ext) {
        if (strcmp(ext, ".png") == 0 || strcmp(ext, ".PNG") == 0) {
            result = stbi_write_png(argv[2], width, height, channels, output, width * channels);
        } else if (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".jpeg") == 0 || 
                   strcmp(ext, ".JPG") == 0 || strcmp(ext, ".JPEG") == 0) {
            result = stbi_write_jpg(argv[2], width, height, channels, output, 95);
        } else if (strcmp(ext, ".bmp") == 0 || strcmp(ext, ".BMP") == 0) {
            result = stbi_write_bmp(argv[2], width, height, channels, output);
        } else {
            // Default ke PNG
            result = stbi_write_png(argv[2], width, height, channels, output, width * channels);
        }
    } else {
        result = stbi_write_png(argv[2], width, height, channels, output, width * channels);
    }
    
    if (result) {
        printf("✓ Berhasil! Gambar disimpan di: %s\n\n", argv[2]);
    } else {
        printf("✗ Gagal menyimpan gambar\n\n");
    }
    
    // Cleanup
    stbi_image_free(input);
    free(output);
    
    return 0;
}

