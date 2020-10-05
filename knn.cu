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

#define DISTANCETHREADSPERBLOCKDIM 16  // TODO change from 32 to 16? why isnt distance 32?
#define REDUCTIONTHREADSBLOCKDIMX 256
#define REDUCTIONTHREADSBLOCKDIMY 4

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
	__shared__ float sharedDistanceMemory[REDUCTIONTHREADSBLOCKDIMY][REDUCTIONTHREADSBLOCKDIMX];
	__shared__ int sharedClassMemory[REDUCTIONTHREADSBLOCKDIMY][REDUCTIONTHREADSBLOCKDIMX];

	int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
	int tid_y  = blockIdx.y*blockDim.y + threadIdx.y;
	int startingDistanceIndex = tid_y * numInstances; // NOTE startingDistanceIndex is used to access the proper "row" of the distance matrix

	// if (tid_x == 0 && tid_y == 0) {
	// 	printf("\nin deviceFindMinK blockDim.x: %d, blockDim.y: %d, gridDim.x: %d, gridDim.y: %d:\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);
	// }

	sharedDistanceMemory[threadIdx.y][threadIdx.x] = (tid_x < numInstances) && (tid_y < numInstances) ? d_distanceMatrix[startingDistanceIndex + tid_x] : FLT_MAX;
	sharedClassMemory[threadIdx.y][threadIdx.x] = (tid_x < numInstances) && (tid_y < numInstances) ? d_actualClasses[tid_x] : -1;

	__syncthreads();

	// do reduction in shared memory
	// for (int s = 0; s < blockDim.x; s += k) {
	// ((ceiling division of blockDim.x / k) / 2) is number of "chunks" of size k that barely spill over the halfway point
	// mulitply it by k to get the actual max s value to start at
	int prevS = blockDim.x; // TODO works at max k?

	for (int s = (((blockDim.x + k - 1) / k) / 2) * k; s < prevS; s = (((s / k) + 2 - 1) / 2) * k) { // (ceil(blocksSizeK left / 2) * k)  TODO what happens when k > blockDim?
		if (threadIdx.x < s && threadIdx.x % k == 0 && tid_y < numInstances) { // TODO check if tid_y < numInstances? to prevent precitions for the excess 'y' dim blocks?
			int leftIndex = threadIdx.x;
			int rightIndex = leftIndex + s;
			float* result = new float[k]; // TODO does something need to be freed?
			int* resultClasses = new int[k]; // TODO does something need to be freed?

			// if on first iteration
			if (prevS == blockDim.x) {
				thrust::sort_by_key(thrust::seq, sharedDistanceMemory[threadIdx.y] + leftIndex, sharedDistanceMemory[threadIdx.y] + leftIndex + k, sharedClassMemory[threadIdx.y] + leftIndex);
				int actualEndingIndex = rightIndex + k;
				if (actualEndingIndex >= REDUCTIONTHREADSBLOCKDIMX)
					actualEndingIndex = REDUCTIONTHREADSBLOCKDIMX;
				thrust::sort_by_key(thrust::seq, sharedDistanceMemory[threadIdx.y] + rightIndex, sharedDistanceMemory[threadIdx.y] + actualEndingIndex, sharedClassMemory[threadIdx.y] + rightIndex);
			}

			for (int i = 0; i < k; i++) {
				if (rightIndex < blockDim.x && sharedDistanceMemory[threadIdx.y][rightIndex] < sharedDistanceMemory[threadIdx.y][leftIndex]) {
					result[i] = sharedDistanceMemory[threadIdx.y][rightIndex];
					resultClasses[i] = sharedClassMemory[threadIdx.y][rightIndex];
					rightIndex++;
				} else {
					result[i] = sharedDistanceMemory[threadIdx.y][leftIndex];
					resultClasses[i] = sharedClassMemory[threadIdx.y][leftIndex];
					leftIndex++;
				}
			}

			for (int i = 0; i < k; i++) {
				sharedDistanceMemory[threadIdx.y][threadIdx.x + i] = result[i];
				sharedClassMemory[threadIdx.y][threadIdx.x + i] = resultClasses[i];
			}
		}
		
		prevS = s;

		__syncthreads();
	}

	// write your nearestK to global mem
	if (threadIdx.x == 0 && tid_y < numInstances) {
		int startingKIndex = ((tid_y * gridDim.x) + (tid_x/blockDim.x)) * k;
		int endingKIndex = startingKIndex + k;
		// printf("tid_y: %d, startingKIndex: %d\n", tid_y, startingKIndex);

		int j = 0;
		for (int i = startingKIndex; i < endingKIndex; i++) {
			d_smallestK[i] = sharedDistanceMemory[threadIdx.y][j];
			d_smallestKClasses[i] = sharedClassMemory[threadIdx.y][j];
			j++;
		}
	}
}

__global__ void makePredictions(int *d_predictions, float *d_smallestK, int *d_smallestKClasses, int numInstances, int k) {
	int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
	int tid_y  = blockIdx.y*blockDim.y + threadIdx.y;
	int startingDistanceIndex = ((tid_y * gridDim.x) * (blockDim.x/k)) * k; // NOTE startingDistanceIndex is used to access the proper "row" of the smallestK matrix
	// TODO test at higher instances values

	// if (tid_x == 0 && tid_y == 0) {
	// 	printf("\nin makePredictions blockDim.x: %d, blockDim.y: %d, gridDim.x: %d, gridDim.y: %d:\n\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);
	// }

	int prevS = blockDim.x; // TODO works at max k?

	for (int s = (((blockDim.x + k - 1) / k) / 2) * k; s < prevS; s = (((s / k) + 2 - 1) / 2) * k) { // (ceil(blocksSizeK left / 2) * k)  TODO what happens when k > blockDim?
		if (threadIdx.x < s && threadIdx.x % k == 0 && tid_y < numInstances) {
			int leftIndex = threadIdx.x + startingDistanceIndex;
			int rightIndex = leftIndex + s;

			float* result = new float[k]; // TODO does something need to be freed?
			int* resultClasses = new int[k]; // TODO does something need to be freed?

			for (int i = 0; i < k; i++) {
				if (d_smallestK[rightIndex] < d_smallestK[leftIndex]) {
					result[i] = d_smallestK[rightIndex]; // TODO bug, when I print these result arrays after the fact, they only have values of 0, even though the d_smallestK have non zeroes
					resultClasses[i] = d_smallestKClasses[rightIndex];
					rightIndex++;
				} else {
					result[i] = d_smallestK[leftIndex];
					resultClasses[i] = d_smallestKClasses[leftIndex];
					leftIndex++;
				}
			}
			
			for (int i = 0; i < k; i++) {
				d_smallestK[threadIdx.x + startingDistanceIndex + i] = result[i];
				d_smallestKClasses[threadIdx.x + startingDistanceIndex + i] = resultClasses[i];
			}
		}
		
		prevS = s;
		__syncthreads();
	}

	// make predictions
	// get max class
	if (threadIdx.x == 0 && threadIdx.y < numInstances) {
		int endingDistanceIndex = startingDistanceIndex + k;
		// printf("making prediction, \n");
		int maxClass = 0;
		for (int i = startingDistanceIndex; i < endingDistanceIndex; i++) {
			if (d_smallestKClasses[i] > maxClass)
				maxClass = d_smallestKClasses[i];
		}

		int* classCounter = new int[maxClass + 1]; // TODO does something need to be freed?
		for (int i = startingDistanceIndex; i < endingDistanceIndex; i++) {
			classCounter[d_smallestKClasses[i]]++;
		}

		int voteResult = -1;
		int numberOfVotes = -1;
		for (int i = 0; i <= maxClass; i++) {
			if (classCounter[i] > numberOfVotes) {
				numberOfVotes = classCounter[i];
				voteResult = i;
			}
		d_predictions[tid_y] = voteResult;
		}
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
	float distanceMilliseconds = 0;
	float predictionMilliseconds = 0;

	// distance matrix
	int gridDimSize = (dataset->num_instances() + DISTANCETHREADSPERBLOCKDIM - 1) / DISTANCETHREADSPERBLOCKDIM;

	dim3 blockSize(DISTANCETHREADSPERBLOCKDIM, DISTANCETHREADSPERBLOCKDIM);
	dim3 gridSize(gridDimSize, gridDimSize);

	// printf("CUDA kernel launch with %dx%d blocks of %dx%d threads\n", gridDimSize, gridDimSize, DISTANCETHREADSPERBLOCKDIM, DISTANCETHREADSPERBLOCKDIM);
	
	cudaEventRecord(start);

	fillDistanceMatrix<<<gridSize, blockSize>>>(d_datasetArray, d_distanceMatrix, dataset->num_instances(), dataset->num_attributes());

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&distanceMilliseconds, start, stop);
	printf("GPU time to fill Distance Matrix %f ms\n", distanceMilliseconds);



	// matrix reduction
	cudaEventRecord(start);
	// reductionBlocksPerGridX is number of blocks (where block has 256 threads) to do all elements in row (dataset->num_instances())
	int reductionBlocksPerGridX = (dataset->num_instances() + REDUCTIONTHREADSBLOCKDIMX - 1) / REDUCTIONTHREADSBLOCKDIMX;
	int reductionBlocksPerGridY = (dataset->num_instances() + REDUCTIONTHREADSBLOCKDIMY - 1) / REDUCTIONTHREADSBLOCKDIMY;

	// printf("\nreduction dims: REDUCTIONTHREADSBLOCKDIMX: %d, REDUCTIONTHREADSBLOCKDIMY: %d, reductionBlocksPerGridX: %d, reductionBlocksPerGridY: %d\n",
	// REDUCTIONTHREADSBLOCKDIMX, REDUCTIONTHREADSBLOCKDIMY, reductionBlocksPerGridX, reductionBlocksPerGridY);

	dim3 reductionBlockSize(REDUCTIONTHREADSBLOCKDIMX, REDUCTIONTHREADSBLOCKDIMY); // 256 * 4 <= 1024
	dim3 reductionGridSize(reductionBlocksPerGridX, reductionBlocksPerGridY);

	int *h_actualClasses = (int *)malloc(dataset->num_instances() * sizeof(int));
	for (int i = 0; i < dataset->num_instances(); i++) {
		h_actualClasses[i] = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
	}

	int *d_actualClasses;
	cudaMalloc(&d_actualClasses, dataset->num_instances() * sizeof(int));

	cudaMemcpy(d_actualClasses, h_actualClasses, dataset->num_instances() * sizeof(int), cudaMemcpyHostToDevice);

	// float **d_smallestK;
	float *d_smallestK;
	cudaMalloc(&d_smallestK, k * reductionBlocksPerGridX * dataset->num_instances() * sizeof(float));
	int *d_smallestKClasses;
	cudaMalloc(&d_smallestKClasses, k * reductionBlocksPerGridX * dataset->num_instances() * sizeof(int));

	// cudadevicesynchronize(); // wait for distanceMAtrix to be filled TODO needed?

	deviceFindMinK<<<reductionGridSize, reductionBlockSize>>>(d_smallestK, d_smallestKClasses, d_distanceMatrix, d_actualClasses, dataset->num_instances(), k);

	cudaError_t cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess) {
		fprintf(stderr, "post deviceFindMinK cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
		exit(EXIT_FAILURE);
	}

	// cudadevicesynchronize(); // wait for smallestK's to be filled TODO needed?



	// prediction matrix
	int numberOfKSegmentsPerRow = k * reductionBlocksPerGridX;
	int maxNumberOFInstancesToPredictAtOnce = 1024 / numberOfKSegmentsPerRow;

	int predictionBlocksPerGridX = (numberOfKSegmentsPerRow + 1024 - 1) / 1024;
	int predictionBlocksPerGridY = (dataset->num_instances() + maxNumberOFInstancesToPredictAtOnce - 1) / maxNumberOFInstancesToPredictAtOnce;

	// printf("\nprediction dims: numberOfKSegmentsPerRow: %d, maxNumberOFInstancesToPredictAtOnce: %d, predictionBlocksPerGridX: %d, predictionBlocksPerGridY: %d\n",
		// numberOfKSegmentsPerRow, maxNumberOFInstancesToPredictAtOnce, predictionBlocksPerGridX, predictionBlocksPerGridY);

	dim3 predictionBlockSize(numberOfKSegmentsPerRow, maxNumberOFInstancesToPredictAtOnce);
	dim3 predictionGridSize(predictionBlocksPerGridX, predictionBlocksPerGridY);


	makePredictions<<<predictionGridSize, predictionBlockSize>>>(d_predictions, d_smallestK, d_smallestKClasses, dataset->num_instances(), k);

	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess) {
		fprintf(stderr, "post makePredictions cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
		exit(EXIT_FAILURE);
	}

	cudaMemcpy(h_predictions, d_predictions, dataset->num_instances() * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&predictionMilliseconds, start, stop);
	printf("time to make predictions %f ms\n", predictionMilliseconds);


	printf("total GPU time: %f ms\n", distanceMilliseconds + predictionMilliseconds);






	// for (int i = 0; i < dataset->num_instances(); i++) {
	// 	printf("actual: %d, predicted: %d\n", h_actualClasses[i], h_predictions[i]);
	// }

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