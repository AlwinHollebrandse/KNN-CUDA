#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <map>
#include <climits> 
#include <cfloat>
#include <vector>
#include <algorithm> // for heap
#include <numeric> // std::iota
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"

#define THREADSPERBLOCK 256

__global__ void fillDistanceMatrix(float *d_datasetArray, float *d_distanceMatrix, int width, int numberOfAttributes) { // d_distanceMatrix is a square matrix
	int column = ( blockDim.x * blockIdx.x ) + threadIdx.x; // TODO which is outer and inner?
	int row    = ( blockDim.y * blockIdx.y ) + threadIdx.y;

	if (row < width && column < width) {
		if (row == column) {
			d_distanceMatrix[row*width + column] = FLT_MAX; // cant compare to to self
		} else {
			float distance = 0;

			for(int k = 0; k < numberOfAttributes - 1; k++) { // compute the distance between the two instances
				float diff = d_datasetArray[row * numberOfAttributes + k] - d_datasetArray[column * numberOfAttributes + k]; // one instance minus the other
				distance += diff * diff;
			}

			d_distanceMatrix[row*width + column] = sqrt(distance);
		}
	}
}

// Uses a shared memory reduction approach to find the smallest k values
__global__ void deviceFindMinK(float *d_smallestK, int *d_smallestKClasses, float *d_distanceMatrix, int *d_actualClasses, int numInstances, int k) {
	__shared__ float sharedDistanceMemory[4][THREADSPERBLOCK];
	__shared__ int sharedClassMemory[4][THREADSPERBLOCK];

	int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
	int tid_y  = blockIdx.y*blockDim.y + threadIdx.y;
	int startingDistanceIndex = tid_y * numInstances; // NOTE startingDistanceIndex is used to accesss the proper "row" of the matrix

	if (tid_x == 0) {
		printf("\nin deviceFindMinK tid_y: %d\n", tid_y);
	}

	sharedDistanceMemory[tid_y][threadIdx.x] = (tid_x < numInstances) ? d_distanceMatrix[startingDistanceIndex + tid_x] : FLT_MAX;
	sharedClassMemory[tid_y][threadIdx.x] = (tid_x < numInstances) ? d_actualClasses[tid_x] : -1;

	__syncthreads();

	if (tid_x == 0 && tid_y == 0) {
		printf("\n pre shared mem. k = %d:\n", k);
		for (int i = 0; i < THREADSPERBLOCK; i++) {
			printf("%f, ", sharedDistanceMemory[tid_y][i]);
		}

		printf("\n pre shared class mem. k = %d:\n", k);
		for (int i = 0; i < THREADSPERBLOCK; i++) {
			printf("%d, ", sharedClassMemory[tid_y][i]);
		}
		printf("\n\n");
	}

    __syncthreads();

	// do reduction in shared memory
	// for (int s = 0; s < blockDim.x; s += k) {
	// ((ceiling division of blockDim.x / k) / 2) is number of "chunks" of size k that barely spill over the halfway point
	// mulitply it by k to get the actual max s value to start at
	int prevS = blockDim.x; // TODO works at max k?

	for (int s = (((blockDim.x + k - 1) / k) / 2) * k; s < prevS; s = (((s / k) + 2 - 1) / 2) * k) { // (ceil(blocksSizeK left / 2) * k)  TODO what happens when k > blockDim?
		if (threadIdx.x < s && threadIdx.x % k == 0) {
			int leftIndex = threadIdx.x;
			int rightIndex = leftIndex + s;
			// printf("s: %d, leftIndex: %d, rightIndex: %d\n", s, leftIndex, rightIndex);
			float* result = new float[k]; // TODO does something need to be freed?
			int* resultClasses = new int[k]; // TODO does something need to be freed?

			// if on first iteration
			if (prevS == blockDim.x) {
				thrust::sort_by_key(thrust::seq, sharedDistanceMemory[tid_y] + leftIndex, sharedDistanceMemory[tid_y] + leftIndex + k, sharedClassMemory[tid_y] + leftIndex);
				int actualEndingIndex = rightIndex + k;
				if (actualEndingIndex >= THREADSPERBLOCK)
					actualEndingIndex = THREADSPERBLOCK;
				thrust::sort_by_key(thrust::seq, sharedDistanceMemory[tid_y] + rightIndex, sharedDistanceMemory[tid_y] + actualEndingIndex, sharedClassMemory[tid_y] + rightIndex);
			}

			for (int i = 0; i < k; i++) {
				if (rightIndex < blockDim.x && sharedDistanceMemory[tid_y][rightIndex] < sharedDistanceMemory[tid_y][leftIndex]) {
					result[i] = sharedDistanceMemory[tid_y][rightIndex];
					resultClasses[i] = sharedClassMemory[tid_y][rightIndex];
					rightIndex++;
				} else {
					result[i] = sharedDistanceMemory[tid_y][leftIndex];
					resultClasses[i] = sharedClassMemory[tid_y][leftIndex];
					leftIndex++;
				}
			}

			for (int i = 0; i < k; i++) {
				sharedDistanceMemory[tid_y][threadIdx.x + i] = result[i];
				sharedClassMemory[tid_y][threadIdx.x + i] = resultClasses[i];
			}
		}
		
		prevS = s;

		__syncthreads();
	}

	if (tid_x == 0 && tid_y == 0) {
		printf("\n block final smallest k:\n");
		for (int i = 0; i < k; i++) {
			printf("%f, ", sharedDistanceMemory[tid_y][i]);
		}
		printf("\n\n");
		printf("\n block final classes of smallest k:\n");
		for (int i = 0; i < k; i++) {
			printf("%d, ", sharedClassMemory[tid_y][i]);
		}
		printf("\n");
	}

	// write your nearestK to global mem
	if (tid_x == 0) {
		int startingKIndex = k * tid_y;
		int endingKIndex = startingKIndex + k;
		int j = 0;
		for (int i = startingKIndex; i < endingKIndex; i++) {
			d_smallestK[i] = sharedDistanceMemory[tid_y][j];
			d_smallestKClasses[i] = sharedClassMemory[tid_y][j];
			j++;
		}
	}
}

// TODO call this with the correct dimensions
__global__ void makePredicitions(int *d_predictions, float *d_smallestK, int *d_smallestKClasses, int sizeOfSmallest, int startingDistanceIndex, int yIndex, int k) {
	if (threadIdx.x == 0) { // TODO remove startingDistanceIndex and replace yIndex with tidY replace startingDistanceIndex with tidY * k
		printf("\nin makePredicitions %d:\n", yIndex);
	}

	int tid_x = blockIdx.x*blockDim.x + threadIdx.x;

	int prevS = gridDim.x;//blockDim.x; // TODO works at max k?

	for (int s = (((blockDim.x + k - 1) / k) / 2) * k; s < prevS; s = (((s / k) + 2 - 1) / 2) * k) { // (ceil(blocksSizeK left / 2) * k)  TODO what happens when k > blockDim?
		if (threadIdx.x < s && threadIdx.x % k == 0 && threadIdx.x < sizeOfSmallest) {
			int leftIndex = threadIdx.x + startingDistanceIndex;
			int rightIndex = leftIndex + s;
			// printf("s: %d, leftIndex: %d, rightIndex: %d\n", s, leftIndex, rightIndex);
			float* result = new float[k]; // TODO does something need to be freed?
			int* resultClasses = new int[k]; // TODO does something need to be freed?

			for (int i = 0; i < k; i++) {
				if (rightIndex < blockDim.x && d_smallestK[rightIndex] < d_smallestK[leftIndex]) {
					result[i] = d_smallestK[rightIndex];
					resultClasses[i] = d_smallestKClasses[rightIndex];
					rightIndex++;
				} else {
					result[i] = d_smallestK[leftIndex];
					resultClasses[i] = d_smallestKClasses[leftIndex];
					leftIndex++;
				}
			}

			for (int i = 0; i < k; i++) {
				d_smallestK[threadIdx.x + i] = result[i];
				d_smallestKClasses[threadIdx.x + i] = resultClasses[i];
			}
		}
		
		prevS = s;
		__syncthreads();
	}

	if (tid_x == 0) {
		int endingDistanceIndex = startingDistanceIndex + k;
		printf("\n d_smallestK:\n");
		for (int i = startingDistanceIndex; i < endingDistanceIndex; i++) {
			printf("%f, ", d_smallestK[i]);
		}
		printf("\n\n");

		printf("\n d_smallestKClasses:\n");
		for (int i = startingDistanceIndex; i < endingDistanceIndex; i++) {
			printf("%d, ", d_smallestKClasses[i]);
		}
		printf("\n\n");
	}

	// make predictions
	// get max class
	if (threadIdx.x == 0) {
		int endingDistanceIndex = startingDistanceIndex + k;
		printf("making prediction, \n");
		int maxClass = 0;
		for (int i = startingDistanceIndex; i < endingDistanceIndex; i++) {  // TODO when 2d make startingDistanceIndex be the start
			if (d_smallestKClasses[i] > maxClass)
				maxClass = d_smallestKClasses[i];
		}
		printf("maxClass: %d\n", maxClass);

		int* classCounter = new int[maxClass + 1]; // TODO does something need to be freed?
		for (int i = startingDistanceIndex; i < endingDistanceIndex; i++) { // TODO when 2d make startingDistanceIndex be the start
			classCounter[d_smallestKClasses[i]]++;
		}

		printf("\n classCounter:\n");
		for (int i = 0; i <= maxClass; i++) {
			printf("%d, ", classCounter[i]);
		}
		printf("\n\n");
		
		int voteResult = -1;
		int numberOfVotes = -1;
		for (int i = 0; i <= maxClass; i++) {
			if (classCounter[i] > numberOfVotes) {
				numberOfVotes = classCounter[i];
				voteResult = i;
			}
		}
		
		d_predictions[yIndex] = voteResult;
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
	if (k > dataset->num_instances())
		k = dataset->num_instances();
	
	int datasetMatrixLength = dataset->num_instances();// TODO are tehse needed?
	int datasetMatrixWidth = dataset->num_attributes();
	int numElements = datasetMatrixLength * datasetMatrixWidth;

	// Allocate host memory
	float *h_datasetArray = (float *)malloc(numElements * sizeof(float));
	float *h_distanceMatrix = (float *)malloc(dataset->num_instances() * dataset->num_instances() * sizeof(float)); // used to find all distances in parallel
	int *h_predictions = (int *)malloc(dataset->num_instances() * sizeof(int));

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
	dim3 gridSize(gridDimSize, gridDimSize);

	printf("CUDA kernel launch with %dx%d blocks of %dx%d threads\n", gridDimSize, gridDimSize, threadsPerBlockDim, threadsPerBlockDim);
	
	cudaEventRecord(start);

	fillDistanceMatrix<<<gridSize, blockSize>>>(d_datasetArray, d_distanceMatrix, dataset->num_instances(), dataset->num_attributes());

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU time to fill Distance Matrix %f ms\n", milliseconds);



	// this is all for the matrix reduction
	int blocksPerGrid = (dataset->num_instances() + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
	dim3 reductionBlockSize(threadsPerBlockDim, 4); // 256 * 4 <= 1024
	dim3 reductionGridSize(blocksPerGrid, blocksPerGrid); // TODOwould grid also be 2D?


	// in addition, my kernel seems to only be getting called once and making tid_y 0-4 instead of  the 0-numInstances like I was expecting. 


	// dim3 blockSize(threadsPerBlockDim, threadsPerBlockDim);
	// dim3 gridSize(gridDimSize, gridDimSize);

	int *h_actualClasses = (int *)malloc(dataset->num_instances() * sizeof(int));
	for (int i = 0; i < dataset->num_instances(); i++) {
		h_actualClasses[i] = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
	}

	int *d_actualClasses;
	cudaMalloc(&d_actualClasses, dataset->num_instances() * sizeof(int));

	cudaMemcpy(d_actualClasses, h_actualClasses, dataset->num_instances() * sizeof(int), cudaMemcpyHostToDevice);

	// float **d_smallestK;
	float *d_smallestK;
	cudaMalloc(&d_smallestK, k * blocksPerGrid * dataset->num_instances() * sizeof(float));
	int *d_smallestKClasses;
	cudaMalloc(&d_smallestKClasses, k * blocksPerGrid * dataset->num_instances() * sizeof(int));

	// cudadevicesynchronize(); // wait for distanceMAtrix to be filled TODO needed?

	deviceFindMinK<<<blocksPerGrid, reductionBlockSize>>>(d_smallestK, d_smallestKClasses, d_distanceMatrix, d_actualClasses, dataset->num_instances(), k);

	cudaError_t cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess) {
		fprintf(stderr, "post deviceFindMinK cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
		exit(EXIT_FAILURE);
	}

	// cudadevicesynchronize(); // wait for smallestK's to be filled TODO needed?

	int sizeOfSmallest = k * blocksPerGrid * dataset->num_instances();
	// for (int i = 0; i < dataset->num_instances(); i++) { // dataset->num_instances() // TODO replace loop with tid_y. see slack.
	// 	// TODO change dimensions
	// 	makePredicitions<<<blocksPerGrid, THREADSPERBLOCK>>>(d_predictions, d_smallestK, d_smallestKClasses, sizeOfSmallest, i * k, i, k);

	// 	cudaError_t cudaError = cudaGetLastError();
	// 	if(cudaError != cudaSuccess) {
	// 		fprintf(stderr, "post makePredicitions cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
	// 		exit(EXIT_FAILURE);
	// 	}
	// }

	cudaMemcpy(h_predictions, d_predictions, dataset->num_instances() * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("time to make predictions %f ms\n", milliseconds);


	for (int i = 0; i < dataset->num_instances(); i++) {
		printf("actual: %d, predicted: %d\n", h_actualClasses[i], h_predictions[i]);
	}

	// Compute the confusion matrix
	int* confusionMatrix = computeConfusionMatrix(h_predictions, dataset);
	// Calculate the accuracy
	float accuracy = computeAccuracy(confusionMatrix, dataset);

	printf("The KNN classifier for %lu instances with k=%d had an accuracy of %.4f\n", dataset->num_instances(), k, accuracy);

	// Free device global memory
	cudaFree(d_datasetArray);
	cudaFree(d_predictions);
	cudaFree(d_smallestK);
	cudaFree(d_smallestKClasses);
	cudaFree(d_actualClasses);

	// Free host memory
	free(h_datasetArray);
	free(h_predictions);
	free(h_actualClasses);

	return 0;
}