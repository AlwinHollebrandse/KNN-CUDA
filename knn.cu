#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <map>
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

// A comparator function used by qsort 
int compare(const void * arg1, const void * arg2) { 
    int const *lhs = static_cast<int const*>(arg1);
    int const *rhs = static_cast<int const*>(arg2);
    return (lhs[0] < rhs[0]) ? -1
        :  ((rhs[0] < lhs[0]) ? 1
        :  (lhs[1] < rhs[1] ? -1
        :  ((rhs[1] < lhs[1] ? 1 : 0))));
} 

// Performs majority voting using the first k first elements of an array
int kVoting(int k, float (*shortestKDistances)[2]) {
    std::map<float, int> classCounter;
    for (int i = 0; i < k; i++) {
        classCounter[shortestKDistances[i][1]]++;
    }

    int voteResult = -1;
    int numberOfVotes = -1;
    for (auto i : classCounter) {
        if (i.second > numberOfVotes) {
            numberOfVotes = i.second;
            voteResult = i.first;
        }
    }

    return voteResult;
}

// TODO convert to TILED
__global__ void fillDistanceMatrix(float *d_datasetArray, float *d_distanceMatrix, int width, int numberOfAttributes) { // d_distanceMatrix is a square matrix
	int column = ( blockDim.x * blockIdx.x ) + threadIdx.x; // TODO which is outer and inner?
	int row    = ( blockDim.y * blockIdx.y ) + threadIdx.y;

	if (row < width && column < width) {
		if (row == column) {
			d_distanceMatrix[row*width + column] = INT_MAX; // cant compare to to self TODO make large number? 
		} else {
			float distance = 0;

			for(int k = 0; k < numberOfAttributes - 1; k++) { // compute the distance between the two instances
				// dataset->get_instance(i)->get(k)->operator float() - dataset->get_instance(j)->get(k)->operator float();
				float diff = d_datasetArray[row * numberOfAttributes + k] - d_datasetArray[column * numberOfAttributes + k]; // one instance minus the other
				distance += diff * diff;
			}

			d_distanceMatrix[row*width + column] = sqrt(distance);
		}
	}
}

// __global__ void findKNN(float *d_distanceMatrix, float *d_predictions, int width, int k) { // d_distanceMatrix is a square matrix
// 	int column = ( blockDim.x * blockIdx.x ) + threadIdx.x; // TODO which is outer and inner?
// 	int row    = ( blockDim.y * blockIdx.y ) + threadIdx.y;

// 	if (row < width && column < width) {
// 		if (row == column) {
// 			d_distanceMatrix[row*width + column] = 0; // cant compare to to self TODO make large number? 
// 		} else {
// 			float distance = 0;

// 			for(int k = 0; k < numberOfAttributes - 1; k++) { // compute the distance between the two instances
// 				// dataset->get_instance(i)->get(k)->operator float() - dataset->get_instance(j)->get(k)->operator float();
// 				float diff = d_datasetArray[row * numberOfAttributes + k] - d_datasetArray[column * numberOfAttributes + k]; // one instance minus the other
// 				distance += diff * diff;
// 			}

// 			d_distanceMatrix[row*width + column] = sqrt(distance);
// 		}
// 	}
// }

void hostFindKNN(float *h_distanceMatrix, float *h_datasetArray, int *h_predictions, int width, int numberOfAttributes, int k) { // h_distanceMatrix is a square matrix
	for (int i = 0; i < width; i++) {
		float distancesAndClasses[width][2];

		for (int j = 0; j < width; j++) {
			// printf("i: %d, j: %d, distance: %f, class: %f\n", i, j, h_distanceMatrix[i*width + j], h_datasetArray[j*numberOfAttributes + numberOfAttributes -1]);
			distancesAndClasses[j][0] = sqrt(h_distanceMatrix[i*width + j]); // distance
			distancesAndClasses[j][1] = h_datasetArray[j*numberOfAttributes + numberOfAttributes -1]; // class
		}
		// for (int j = 0; j < width; j++) {
		// 	printf("distance: %f, class: %f\n", distancesAndClasses[j][0], distancesAndClasses[j][1]);
		// }

		qsort(distancesAndClasses, width, (2 * sizeof(float)), compare); // TODO dont need to sort, need to find "k" shortest. Due to programming time contraints, this wasnt done yet

		float shortestKDistances[k][2];
		for(int j = 0; j < k; j++) {
			shortestKDistances[j][0] = distancesAndClasses[j][0];
			shortestKDistances[j][1] = distancesAndClasses[j][1];
		}

		h_predictions[i] = kVoting(k, shortestKDistances);
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
        printf("Usage: ./main datasets/datasetFile.arff kValue");
        exit(0);
    }
    
    // Open the dataset
    ArffParser parser(argv[1]);
	ArffData *dataset = parser.parse();
	int k = atoi(argv[2]);
	
	int datasetMatrixLength = dataset->num_instances();// TODO are tehse needed?
	int datasetMatrixWidth = dataset->num_attributes();
	int numElements = datasetMatrixLength * datasetMatrixWidth;

	// Allocate host memory
	float *h_datasetArray = (float *)malloc(numElements * sizeof(float));
	float *h_distanceMatrix = (float *)malloc(dataset->num_instances() * dataset->num_instances() * sizeof(float)); // used to find all distances in parallel
	int *h_predictions = (int *)malloc(dataset->num_instances() * sizeof(int));
	int *h_predictions_CPUres = (int *)malloc(dataset->num_instances() * sizeof(int)); // TODO do I need a cpu version?

	// Initialize the host input matrixs
	for (int i = 0; i < datasetMatrixLength; ++i) {
		for (int j = 0; j < datasetMatrixWidth; ++j) {
			h_datasetArray[i * datasetMatrixWidth + j] = dataset->get_instance(i)->get(j)->operator float(); // TODO how to handle class?
		}
	}

	// Allocate the device input matrix A
	float *d_datasetArray;
	float *d_distanceMatrix;
	int *d_predictions;

	cudaMalloc(&d_datasetArray, numElements * sizeof(float));
	cudaMalloc(&d_distanceMatrix, dataset->num_instances() * dataset->num_instances() * sizeof(float));
	cudaMalloc(&d_predictions, dataset->num_instances() * sizeof(int));

	// Copy the host input matrixs A and B in host memory to the device input matrixs in
	cudaMemcpy(d_datasetArray, h_datasetArray, numElements * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_distanceMatrix, h_distanceMatrix, dataset->num_instances() * dataset->num_instances() * sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	int threadsPerBlockDim = 16; // dataset->num_attributes(); // so each thread handles 1 attribute. TODO handle class
	int gridDimSize = (dataset->num_instances() + threadsPerBlockDim - 1) / threadsPerBlockDim; //512 / threadsPerBlockDim; // TODO try other values?  // TODO no clue. maybe match num attributes? was matrixSize

	// this is all for the distance matrix
	dim3 blockSize(threadsPerBlockDim, threadsPerBlockDim);
	dim3 gridSize (gridDimSize, gridDimSize);

	printf("CUDA kernel launch with %dx%d blocks of %dx%d threads\n", gridDimSize, gridDimSize, threadsPerBlockDim, threadsPerBlockDim);
	
	cudaEventRecord(start);

	fillDistanceMatrix<<<gridSize, blockSize>>>(d_datasetArray, d_distanceMatrix, dataset->num_instances(), dataset->num_attributes());

	// knn(dataset, k, d_datasetArray, d_predictions);

	// matrixMul<<<gridSize, blockSize>>>(d_datasetArray, d_distanceMatrix, d_predictions, matrixSize);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU time to fill Distance Matrix %f ms\n", milliseconds);

	cudaMemcpy(h_distanceMatrix, d_distanceMatrix, dataset->num_instances() * dataset->num_instances() * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < dataset->num_instances(); i++) {
		for (int j = 0; j < dataset->num_instances(); j++) {
			printf("%f, ", h_distanceMatrix[i * dataset->num_instances() + j]);
		}
		printf("\n");
	}

	
	hostFindKNN(h_distanceMatrix, h_datasetArray, h_predictions, dataset->num_instances(), dataset->num_attributes(), k);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("time to make predictions %f ms\n", milliseconds);

	for (int i = 0; i < dataset->num_instances(); i++) {
		printf("actual: %d, predicted: %d\n", dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32(), h_predictions[i]);
	}

	cudaEventRecord(start);

	// matrixMulTiled<<<gridSize, blockSize>>>(d_datasetArray, d_distanceMatrix, d_predictions, matrixSize);

	// Compute the confusion matrix
	int* confusionMatrix = computeConfusionMatrix(h_predictions, dataset);
	// Calculate the accuracy
	float accuracy = computeAccuracy(confusionMatrix, dataset);

	printf("The KNN classifier for %lu instances with k=%d had an accuracy of %.4f\n", dataset->num_instances(), k, accuracy);


	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	// printf("The KNN classifier for %lu instances required %f ms CPU time, accuracy was %.4f\n", dataset->num_instances(), milliseconds, accuracy);

	// printf("GPU time to multiple matrixes tiled %f ms\n", milliseconds);

	// Copy the device result matrix in device memory to the host result matrix
	cudaMemcpy(h_predictions, d_predictions, dataset->num_instances() * sizeof(int), cudaMemcpyDeviceToHost);

	cudaError_t cudaError = cudaGetLastError();

	if(cudaError != cudaSuccess)
	{
		fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
		exit(EXIT_FAILURE);
	}

	// Compute CPU time
	cudaEventRecord(start);

	// MatrixMultiplicationHost(h_datasetArray, h_distanceMatrix, h_predictions_CPUres, matrixSize); // TODO need a cpu version?

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("CPU time to sum the matrixes %f ms\n", milliseconds);

	// Verify that the result matrix is correct
	for (int i = 0; i < numElements; i++)
		if (fabs(h_predictions[i] - h_predictions_CPUres[i]) > 1e-3)
		{
			fprintf(stderr, "Result verification failed at element %d, %f vs %f!\n", i, h_predictions[i], h_predictions_CPUres[i]);
			exit(EXIT_FAILURE);
		}

	printf("Multiplication of the matrixes was OK\n");

	// Free device global memory
	cudaFree(d_datasetArray);
	cudaFree(d_predictions);

	// Free host memory
	free(h_datasetArray);
	free(h_predictions);
	free(h_predictions_CPUres);

	return 0;
}