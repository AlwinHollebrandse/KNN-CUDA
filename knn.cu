#include <stdio.h>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"

#define TILE_WIDTH 16

__global__ void matrixMul(float *A, float *B, float *C, int width)
{
	int column = ( blockDim.x * blockIdx.x ) + threadIdx.x;
	int row    = ( blockDim.y * blockIdx.y ) + threadIdx.y;

	if (row < width && column < width)
	{
		float sum = 0;

		for(int k = 0; k < width; k++)
			sum += A[row * width + k] * B[k * width + column];

		C[row*width + column] = sum;
	}
}

__global__ void matrixMulTiled(float *A, float *B, float *C, int width)
{
    int column = ( blockDim.x * blockIdx.x ) + threadIdx.x;
	int row    = ( blockDim.y * blockIdx.y ) + threadIdx.y;
	
    float sum = 0;

    // Loop over the A and B tiles required to compute the submatrix
    for (int t = 0; t < width/TILE_WIDTH; t++)
    {
        __shared__ float sub_A[TILE_WIDTH][TILE_WIDTH];
        __shared__ float sub_B[TILE_WIDTH][TILE_WIDTH];
        
        // Coolaborative loading of A and B tiles into shared memory
        sub_A[threadIdx.y][threadIdx.x] = A[row*width + (t*TILE_WIDTH + threadIdx.x)];
        sub_B[threadIdx.y][threadIdx.x] = B[column + (t*TILE_WIDTH + threadIdx.y)*width];
        
        __syncthreads();
    
        // Loop within shared memory
        for (int k = 0; k < TILE_WIDTH; k++)
          sum += sub_A[threadIdx.y][k] * sub_B[k][threadIdx.x];
      
        __syncthreads();
    }
    
    C[row*width + column] = sum;
}

void MatrixMultiplicationHost(float *A, float *B, float *C, int width)
{
	for (int i = 0; i < width; i++)
		for (int j = 0; j < width; j++)
		{
			float sum = 0;

			for (int k = 0; k < width; k++)
				sum += A[i * width + k] * B[k * width + j];

			C[i * width + j] = sum;
		}
}

int* computeConfusionMatrix(int* predictions, ArffData* dataset) {
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matrix size numberClasses x numberClasses
    
    for(int i = 0; i < dataset->num_instances(); i++) { // for each instance compare the true class and predicted class
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];
        
        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }
    
    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset) {
    int successfulPredictions = 0;
    
    for(int i = 0; i < dataset->num_classes(); i++) {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagonal are correct predictions
    }
    
    return successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char* argv[])
{
	if(argc != 3) {
        cout << "Usage: ./main datasets/datasetFile.arff kValue" << endl;
        exit(0);
    }
    
    // Open the dataset
    ArffParser parser(argv[1]);
	ArffData *dataset = parser.parse();
	int globalK = atoi(argv[2]);
	
	int matrixSize = 512; // square matrix matrixSize * matrixSize
	int numElements = matrixSize * matrixSize;

	// Allocate host memory
	float *h_A = (float *)malloc(numElements * sizeof(float));
	float *h_B = (float *)malloc(numElements * sizeof(float));
	float *h_C = (float *)malloc(numElements * sizeof(float));
	float *h_C_CPUres = (float *)malloc(numElements * sizeof(float));

	// Initialize the host input matrixs
	for (int i = 0; i < numElements; ++i)
	{
		h_A[i] = rand()/(float)RAND_MAX;
		h_B[i] = rand()/(float)RAND_MAX;
	}

	// Allocate the device input matrix A
	float *d_A, *d_B, *d_C;

	cudaMalloc(&d_A, numElements * sizeof(float));
	cudaMalloc(&d_B, numElements * sizeof(float));
	cudaMalloc(&d_C, numElements * sizeof(float));

	// Copy the host input matrixs A and B in host memory to the device input matrixs in
	cudaMemcpy(d_A, h_A, numElements * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, numElements * sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	int threadsPerBlockDim = 16;
	int gridDimSize = (matrixSize + threadsPerBlockDim - 1) / threadsPerBlockDim;

	dim3 blockSize(threadsPerBlockDim, threadsPerBlockDim);
	dim3 gridSize (gridDimSize, gridDimSize);

	printf("CUDA kernel launch with %dx%d blocks of %dx%d threads\n", gridDimSize, gridDimSize, threadsPerBlockDim, threadsPerBlockDim);
	
	cudaEventRecord(start);

	matrixMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, matrixSize);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU time to multiple matrixes %f ms\n", milliseconds);

	cudaEventRecord(start);

	// matrixMulTiled<<<gridSize, blockSize>>>(d_A, d_B, d_C, matrixSize);

	// // Compute the confusion matrix
	// int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
	// // Calculate the accuracy
	// float accuracy = computeAccuracy(confusionMatrix, dataset);

	// printf("The KNN classifier for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);


	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	// printf("The KNN classifier for %lu instances required %f ms CPU time, accuracy was %.4f\n", dataset->num_instances(), milliseconds, accuracy);

	// printf("GPU time to multiple matrixes tiled %f ms\n", milliseconds);

	// Copy the device result matrix in device memory to the host result matrix
	cudaMemcpy(h_C, d_C, numElements * sizeof(float), cudaMemcpyDeviceToHost);

	cudaError_t cudaError = cudaGetLastError();

	if(cudaError != cudaSuccess)
	{
		fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
		exit(EXIT_FAILURE);
	}

	// Compute CPU time
	cudaEventRecord(start);

	MatrixMultiplicationHost(h_A, h_B, h_C_CPUres, matrixSize);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("CPU time to sum the matrixes %f ms\n", milliseconds);

	// Verify that the result matrix is correct
	for (int i = 0; i < numElements; i++)
		if (fabs(h_C[i] - h_C_CPUres[i]) > 1e-3)
		{
			fprintf(stderr, "Result verification failed at element %d, %f vs %f!\n", i, h_C[i], h_C_CPUres[i]);
			exit(EXIT_FAILURE);
		}

	printf("Multiplication of the matrixes was OK\n");

	// Free device global memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);
	free(h_C_CPUres);

	return 0;
}