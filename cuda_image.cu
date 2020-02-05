#include <iostream>
#include <stdint.h>
#include <fstream>
#include <sstream> 
#include <vector>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

using namespace std;


__global__ void Kernel_grea_skale(unsigned char *a, unsigned char *b, int N) 
{
	int rows = blockIdx.y * blockDim.y + threadIdx.y;
  	int cols = blockIdx.x * blockDim.x + threadIdx.x;
	int i ;
	
	
	i = (a[rows * N * 3 + cols*3] + a[rows * N * 3 + cols*3 + 1] + a[rows * N * 3 + cols*3 + 2])/3;
	b[rows * N + cols] = i;

}

__global__ void Kernel_neg(unsigned char *a, unsigned char *b, int N) 
{
	int rows = blockIdx.y * blockDim.y + threadIdx.y;
	  int cols = blockIdx.x * blockDim.x + threadIdx.x;
	
	b[rows * N * 3 + cols * 3] = (uint8_t)fmin(0.393 * a[rows * N * 3 + cols * 3] +  0.769 * a[rows * N * 3 + cols * 3 + 1] + 0.189 * a[rows * N * 3 + cols * 3 + 2] , 255.0);
	b[rows * N * 3 + cols * 3 + 1] = (uint8_t)fmin(0.349 * a[rows * N * 3 + cols * 3] +  0.686 * a[rows * N * 3 + cols * 3 + 1] + 0.168 * a[rows * N * 3 + cols * 3 + 2] , 255.0);
	b[rows * N * 3 + cols * 3 + 2] = (uint8_t)fmin(0.272 * a[rows * N * 3 + cols * 3] +  0.534 * a[rows * N * 3 + cols * 3 + 1] + 0.131 * a[rows * N * 3 + cols * 3 + 2] , 255.0);
	
}

__global__ void Kernel_negatiw(unsigned char *a, unsigned char *b, int N) 
{
	int rows = blockIdx.y * blockDim.y + threadIdx.y;
  	int cols = blockIdx.x * blockDim.x + threadIdx.x;
	
	b[rows * N * 3 + cols * 3] = (uint8_t)(255 - a[rows * N * 3 + cols * 3]);
	b[rows * N * 3 + cols * 3 + 1] = (uint8_t)(255 - a[rows * N * 3 + cols * 3 + 1]);
	b[rows * N * 3 + cols * 3 + 2] = (uint8_t)(255 - a[rows * N * 3 + cols * 3 + 2]);
}

__global__ void Kernel_detection(unsigned char *a, unsigned char *b, int col , int row) 
{
	int rows = blockIdx.y * blockDim.y + threadIdx.y;
  	int cols = blockIdx.x * blockDim.x + threadIdx.x;

	float Gx,Gy;

	if(rows == row and cols == col)
	{
		b[rows * col + cols] = 0;
	}
	else if(rows == row){
		Gx = a[rows * col + cols];
		Gy = a[rows * col + (cols + 1)] ;
	}
	else if(cols == col){
		Gx = a[rows * col + cols];
		Gy = a[(rows + 1) * col + (cols + 1)] ;
	}
	else
	{
		Gx = a[rows * col + cols] - a[(rows + 1) * col + (cols + 1)];
		Gy = a[rows * col + (cols + 1)] - a[(rows + 1) * col + cols];
	}
	b[rows * col + cols] = uint8_t(sqrt(Gx*Gx + Gy + Gy));

}


__global__ void Kernel_blur(unsigned char *a, unsigned char *b, double *Gkernel, int col , int row) 
{
	int rows = blockIdx.y * blockDim.y + threadIdx.y;
  	int cols = blockIdx.x * blockDim.x + threadIdx.x;
	
	double sumR = 0;
	double sumG = 0;
	double sumB = 0;

	
	for(int i = -25 ; i <= 26; i++){
		if( (rows + i>=0) and (rows+i <= row)){
			for(int j = -25 ; j <= 26; j++){
				if( (cols + i>=0) and (cols+i <= col)){
					//              |----x-----|  |---y---|    |---------x----------|    |------y------|  
					sumR += Gkernel[(i + 25)*51 + (j + 25)] * a[(rows + i) * (col * 3) + (cols*3 + j*3)];
					sumG += Gkernel[(i + 25)*51 + (j + 25)] * a[(rows + i) * (col * 3) + (cols*3 + j*3 + 1)];
					sumB += Gkernel[(i + 25)*51 + (j + 25)] * a[(rows + i) * (col * 3) + (cols*3 + j*3 + 2)];
				}
					
			}

		}
		
	}
	
	b[rows * col * 3 + cols * 3] = (uint8_t)(sumR);
	b[rows * col * 3 + cols * 3 +1] = (uint8_t)(sumG);
	b[rows * col * 3 + cols * 3 + 2] = (uint8_t)(sumB);
}


void FilterCreation(vector<double> &GKernel)
{
	double sigma = 10.0;
	double r, s = 2.0 * sigma * sigma;

	double sum = 0.0;

	// generating 51x51 kernel 
	for (int x = -25; x <= 25; x++) {
		for (int y = -25; y <= 25; y++) {
			r = sqrt(x * x + y * y);
			GKernel[(x + 25)*51 + (y + 25)] = (exp(-(r * r) / s)) / (6.28 * s);
			sum += GKernel[(x + 25)*51 + (y + 25)];
		}
	}

	// normalising the Kernel 
	for (int i = 0; i < 51; ++i)
		for (int j = 0; j < 51; ++j)
			GKernel[ i*51 + j] /= sum;
			
}

int main(void) {


	// Gausian kernel
	size_t mat_size_GKernel = 51*51 * sizeof(double);
	vector<double> GKernel(mat_size_GKernel);
	
	FilterCreation(GKernel);

	/////////////////////////////////////////////////////
	
	int width, height, channels;
	unsigned char *img = stbi_load("image.jpg", &width, &height, &channels, 0);

	
	//Size for vector
	int col,row;
	col = width;
	row = height;

	size_t mat_size_color = width * height * 3 * sizeof(unsigned char);
	size_t mat_size_L = width * height * 3 * sizeof(unsigned char);
	
	
	
	//GPU
	
	unsigned char *filter_grea =   new unsigned char[mat_size_L];
	unsigned char *filter_detection =   new unsigned char[mat_size_L];

	unsigned char *filter_neg =   new unsigned char[mat_size_color];
	unsigned char *filter_blur =   new unsigned char[mat_size_color];
	unsigned char *filter_sep =   new unsigned char[mat_size_color];


	unsigned char *GPU_image = NULL;
	unsigned char *GPU_szare = NULL;
	unsigned char *GPU_neg = NULL;
	unsigned char *GPU_detection = NULL;
	unsigned char *GPU_blur = NULL;
	unsigned char *GPU_sep = NULL;

	double *GPU_GKernel;


	cout<< "type 1 "<< endl;
	//Cuda Malloc

	cudaMalloc((void ** )&GPU_image, mat_size_color);
	cudaMalloc((void ** )&GPU_blur, mat_size_color);
	cudaMalloc((void ** )&GPU_sep, mat_size_color);
	cudaMalloc((void ** )&GPU_neg, mat_size_color);
	cudaMalloc((void ** )&GPU_szare, mat_size_L);
	cudaMalloc((void ** )&GPU_detection, mat_size_L);
	
	cudaMalloc(&GPU_GKernel, mat_size_GKernel);

		
	//Cuda Memcpy Host To Device

	cudaMemcpy(GPU_image , img, mat_size_color, cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_GKernel, GKernel.data(), mat_size_GKernel, cudaMemcpyHostToDevice);

	//////////////////////////////////////////////////////////////////////

	int THREADS = 32;

  	int BLOCKS_COL = col / THREADS;
	int BLOCKS_ROW = row / THREADS;

	dim3 threads(32, 32);
  	dim3 blocks(BLOCKS_COL, BLOCKS_ROW);
	  
	cout<< "Start grea GPU"<<endl;
	Kernel_grea_skale<<<blocks, threads>>>(GPU_image, GPU_szare, col);

	cout<< "Start neg GPU"<<endl;
	Kernel_negatiw<<<blocks, threads>>>(GPU_image, GPU_neg, col);

	cout<< "Start neg GPU"<<endl;
	Kernel_neg<<<blocks, threads>>>(GPU_image, GPU_sep, col);

	cout<< "Start detection GPU"<<endl;
	Kernel_detection<<<blocks, threads>>>(GPU_szare, GPU_detection, col , row);

	cout<< "Start blur GPU"<<endl;
	Kernel_blur<<<blocks, threads>>>(GPU_image, GPU_blur, GPU_GKernel ,col , row);

	//////////////////////////////////////////////////////////////////////
	cudaMemcpy(filter_grea, GPU_szare, mat_size_L, cudaMemcpyDeviceToHost);
	cudaMemcpy(filter_detection, GPU_detection, mat_size_L, cudaMemcpyDeviceToHost);

	cudaMemcpy(filter_neg, GPU_neg, mat_size_color, cudaMemcpyDeviceToHost);
	cudaMemcpy(filter_blur, GPU_blur, mat_size_color, cudaMemcpyDeviceToHost);
	cudaMemcpy(filter_sep, GPU_sep, mat_size_color, cudaMemcpyDeviceToHost);

	cout<< "zapis do pliku" << endl;


	stbi_write_jpg("filter_neg.jpg", width, height, channels, filter_neg, 0);
	stbi_write_jpg("filter_blur.jpg", width, height, 3, filter_blur, 0);
	stbi_write_jpg("filter_sep.jpg", width, height, 3, filter_sep, 0);

	int gray_channels = channels == 4 ? 2 : 1;
	stbi_write_jpg("filter_grea.jpg", width, height,gray_channels, filter_grea, 100);
	stbi_write_jpg("filter_detection.jpg", width, height,gray_channels, filter_detection, 0);


	delete filter_grea;
	delete filter_detection;
	delete filter_neg;
	delete filter_blur;
	delete filter_sep;
	
	cudaFree(GPU_image);
	cudaFree(GPU_szare);
	cudaFree(GPU_neg);
	cudaFree(GPU_detection);
	cudaFree(GPU_blur);
	cudaFree(GPU_sep);
	cudaFree(GPU_GKernel);

	return 0;
}