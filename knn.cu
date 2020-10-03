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

#define TILE_WIDTH 16

// TODO not needed?
__global__ void fillDistanceMatrixTiled(float *A, float *B, float *C, int width) {
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

struct DistanceAndClass {
	float distance;
	int assignedClass; // class is a reserved word
};

DistanceAndClass* newDistanceAndClass(float distance, int assignedClass) { 
    DistanceAndClass* temp = new DistanceAndClass; 
	temp->distance = distance; 
	temp->assignedClass = assignedClass; 
    return temp; 
}

struct DistanceAndClass_rank_greater_than {
    bool operator()(DistanceAndClass* const a, DistanceAndClass* const b) const {
        return a->distance > b->distance;
    }
};

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

// Function to return k'th smallest element in a given array 
void kthSmallest(std::vector<DistanceAndClass*> distanceAndClassVector, int k, float (*shortestKDistances)[2]) {
	// build a min heap
	std::make_heap(distanceAndClassVector.begin(), distanceAndClassVector.end(), DistanceAndClass_rank_greater_than());
  
    // Extract min (k) times 
	for (int i = 0; i < k; i++) {
		shortestKDistances[i][0] = distanceAndClassVector.front()->distance;
		shortestKDistances[i][1] = (float)distanceAndClassVector.front()->assignedClass;
		std::pop_heap (distanceAndClassVector.begin(), distanceAndClassVector.end(), DistanceAndClass_rank_greater_than());
		distanceAndClassVector.pop_back();
	}
	// printf("final: shortestKDistances[0]: %f, shortestKDistances[1]: %f\n", shortestKDistances[0], shortestKDistances[1]);
}

// void* smallestKMerge(float *result, float *distanceArr, int leftStartingIndex, int leftEndingIndex, int rightStartingIndex, int rightEndingIndex, int k) { // TODO include class array
// 	int leftIndex = leftStartingIndex;
// 	int rightIndex = rightStartingIndex;
	
// 	for (int i = 0; i < k; i++) {
// 		if (distanceArr[leftIndex] < distanceArr[rightIndex]) {
// 			result[i] = distanceArr[leftIndex];
// 			leftIndex++;
// 		} else {
// 			result[i] = distanceArr[rightIndex];
// 			rightIndex++;
// 		}
// 	}
// }

// Uses a shared memory reduction approach to find the smallest k values
__global__ void deviceFindMinKNonOptimized(float *d_smallestK, float *d_distanceMatrix, int *d_actualClasses, int width, int k) { // NOTE, rn im assuming a row is passed in at once

	__shared__ float sharedDistanceMemory[128];
	// __shared__ int sharedClassMemory[128];

    int tid = blockIdx.x*blockDim.x + threadIdx.x;
	sharedDistanceMemory[threadIdx.x] = (tid < width) ? d_distanceMatrix[tid] : FLT_MAX;
	// sharedClassMemory[threadIdx.x] = (tid < width) ? d_actualClasses[tid] : INT_MAX;

    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			float minDistance = sharedDistanceMemory[threadIdx.x];
			// int minClass = sharedClassMemory[threadIdx.x];

			if (sharedDistanceMemory[threadIdx.x + s] < minDistance) {
				minDistance = sharedDistanceMemory[threadIdx.x + s];
				// minClass = sharedClassMemory[threadIdx.x + s];
			}

			sharedDistanceMemory[threadIdx.x] = minDistance; 
			// sharedClassMemory[threadIdx.x] = minClass; 
		}

		__syncthreads();
	}

    // write result for this block to global memory
    // if (threadIdx.x == 0)
	//     atomicAdd(d_smallestK, sharedDistanceMemory[0]); // TODO how to map class to this..... but also need to set the min to a max value now...
}

// Uses a shared memory reduction approach to find the smallest k values
__global__ void deviceFindMinK(float *smallestK, float *d_distanceMatrix, int *d_actualClasses, int width, int k) { // NOTE, rn im assuming a row is passed in at once
	if (threadIdx.x == 0) {
		printf("\nin deviceFindMinK:\n");
	}
	__shared__ float sharedDistanceMemory[128];// or do width? my thinking was to do this as per row of the matrix, so load a whole row (TODO SCALABLE?)
	__shared__ int sharedClassMemory[128];


	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	sharedDistanceMemory[threadIdx.x] = (tid < width) ? d_distanceMatrix[tid] : FLT_MAX;
	sharedClassMemory[threadIdx.x] = (tid < width) ? d_actualClasses[tid] : -1;
	
	if (tid == 0) {
		printf("\n pre shared mem. k = %d:\n", k);
		for (int i = 0; i < 128; i++) {
			printf("%f, ", sharedDistanceMemory[i]);
		}
		
		printf("\n pre shared class mem. k = %d:\n", k); // TODO why are these all FLT_MAX?????
		for (int i = 0; i < 128; i++) {
			printf("%f, ", sharedClassMemory[i]);
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
		// printf("threadIdx.x: %d\n", threadIdx.x);
		if (tid == 0) {
			printf("startingS: %d, currentS: %d, prevS: %d, blockDim.x: %d\n", (((blockDim.x + k - 1) / k) / 2) * k, s, prevS, blockDim.x);
		}
		if (threadIdx.x < s && threadIdx.x % k == 0) {
			int leftIndex = threadIdx.x;
			int rightIndex = threadIdx.x + s;
			printf("s: %d, leftIndex: %d, rightIndex: %d\n", s, leftIndex, rightIndex);
			float result[5]; // TODO k malloc
			int resultClasses[5]; // TODO k malloc

			// float *result;
			// cudaMalloc(&result, k * sizeof(float));

			// if on first iteration
			if (prevS == blockDim.x) {
				thrust::sort_by_key(thrust::seq, sharedDistanceMemory + leftIndex, sharedDistanceMemory + leftIndex + k, sharedClassMemory + leftIndex);
				int actualEndingIndex = rightIndex + k;
				if (actualEndingIndex >= 128)
					actualEndingIndex = 128;
				thrust::sort_by_key(thrust::seq, sharedDistanceMemory + rightIndex, sharedDistanceMemory + actualEndingIndex, sharedClassMemory + rightIndex); // TODO 128 is not /5 so theres excess
			}

			// smallestKMerge(result, sharedDistanceMemory, leftStartingIndex, leftEndingIndex, leftStartingIndex + s, leftEndingIndex + s, k);
			for (int i = 0; i < k; i++) {
				if (rightIndex < blockDim.x && sharedDistanceMemory[rightIndex] < sharedDistanceMemory[leftIndex]) {
					result[i] = sharedDistanceMemory[rightIndex];
					resultClasses[i] = sharedClassMemory[rightIndex];
					rightIndex++;
				} else {
					result[i] = sharedDistanceMemory[leftIndex];
					resultClasses[i] = sharedClassMemory[leftIndex];
					leftIndex++;
				}
			}

			// for (int i = 0; i < k; i++) {
			// 	idx[i] = i;
			// 	printf("%d, ", idx[i]);
			// }

			// thrust::sort_by_key(thrust::seq, idx, idx + k, result);
			// // initialize original index locations
			// std::vector<int> idx(k);
			// std::iota(idx.begin(), idx.end(), 0);

			// // sort indexes based on comparing values in v
			// // using std::stable_sort instead of std::sort
			// // to avoid unnecessary index re-orderings
			// // when v contains elements of equal values 
			// std::stable_sort(idx.begin(), idx.end(),
			// 	[&](int i1, int i2) {return result[i1] < result[i2];});

			if (tid == 0) {
				// printf("\n internal idx[i]:\n");
				// for (int i = 0; i < k; i++) {
				// 	printf("%f, ", idx[i]);
				// }
				// printf("\n\n");

				printf("\n internal result[i]:\n");
				for (int i = 0; i < k; i++) {
					printf("%f, ", result[i]);
				}
				printf("\n\n");

				printf("\n internal resultClasses[i]:\n");
				for (int i = 0; i < k; i++) {
					printf("%f, ", resultClasses[i]);
				}
				printf("\n\n");
			}

			for (int i = 0; i < k; i++) {
				sharedDistanceMemory[threadIdx.x + i] = result[i];
				sharedClassMemory[threadIdx.x + i] = resultClasses[i];
			}
		}

		__syncthreads();
		
		prevS = s;

		if (tid == 0) {
			printf("\nchanging prevS! %d\n", prevS);
		}

		if (tid == 0) {
			printf("\n post shared mem. k = %d:\n", k);
			for (int i = 0; i < 128; i++) {
				printf("%f, ", sharedDistanceMemory[i]);
			}
			printf("\n\n");
		}
		__syncthreads();
	}

	// if (tid == 0) {
	// 	// printf("\n post block 1 shared mem. k = %d:\n", k);
	// 	for (int i = 0; i < 128; i++) {
	// 		printf("%f, ", sharedDistanceMemory[i]);
	// 	}
	// 	// printf("\n");
	// }

	// write result for this block to global memory
	// if (threadIdx.x == 0) {
	// 	for (int i = 0; i < k; i++) {
	// 		atomicMin(result[i], sharedDistanceMemory[i]); // dont I need an atomic merge?
	// 	}
	// }

    // for (int s = blockDim.x/2; s > 0; s >>= 1) {
	// 	if (threadIdx.x < s)
	// 		sharedDistanceMemory[threadIdx.x] += sharedDistanceMemory[threadIdx.x + s];

	// 	__syncthreads();
	// }

    // // write result for this block to global memory
    // if (threadIdx.x == 0)
	//     atomicAdd(result, sharedDistanceMemory[0]);  // TODO how to map class to this..... and how to custom atomic merge, just a critical section that uses all blocks? how to accss all blocks?
	// *mutex should be 0 before calling this function


	// __global__ void kernelFunction(..., unsigned long long* mutex) 
	// {
	//     bool isSet = false; 
	//     do 
	//     {
	//         if (isSet = atomicCAS(mutex, 0, 1) == 0) 
	//         {
	//             // critical section goes here
	//         }
	//         if (isSet) 
	//         {
	//             mutex = 0;
	//         }
	//     } 
	//     while (!isSet);
	// }
}


void hostFindKNN(float *h_distanceMatrix, float *h_datasetArray, int *h_predictions, int width, int numberOfAttributes, int k) { // h_distanceMatrix is a square matrix
	for (int i = 0; i < width; i++) {
		std::vector<DistanceAndClass*> distanceAndClassVector;

		for (int j = 0; j < width; j++) {
			float distance = sqrt(h_distanceMatrix[i*width + j]);
			int assignedClass = (int)h_datasetArray[j*numberOfAttributes + numberOfAttributes -1];
			DistanceAndClass* distanceAndClass = newDistanceAndClass(distance, assignedClass);
			distanceAndClassVector.push_back(distanceAndClass);
		}

		float shortestKDistances[k][2];
		kthSmallest(distanceAndClassVector, k, shortestKDistances);

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
	if (k > dataset->num_instances())
		k = dataset->num_instances();
	
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
	dim3 gridSize(gridDimSize, gridDimSize);

	printf("CUDA kernel launch with %dx%d blocks of %dx%d threads\n", gridDimSize, gridDimSize, threadsPerBlockDim, threadsPerBlockDim);
	
	cudaEventRecord(start);

	fillDistanceMatrix<<<gridSize, blockSize>>>(d_datasetArray, d_distanceMatrix, dataset->num_instances(), dataset->num_attributes());





	// this is all for the matrix reduction TODO delete?
	int threadsPerBlock = 128;
	int blocksPerGrid = (dataset->num_instances() + threadsPerBlock - 1) / threadsPerBlock;
	printf("threadsPerBlock: %d, blocksPerGrid: %d, blockSize: %d, gridSize: %d\n", threadsPerBlock, blocksPerGrid, blockSize, gridSize);
	printf("blockSize: %d, gridSize: %d, threadsPerBlock: %d, blocksPerGrid: %d\n", blockSize, gridSize, threadsPerBlock, blocksPerGrid);


	// dim3 blockSize(threadsPerBlockDim, threadsPerBlockDim);
	// dim3 gridSize(gridDimSize, gridDimSize);

	// TODO avoid this copy if possible
	cudaMemcpy(h_distanceMatrix, d_distanceMatrix, dataset->num_instances() * dataset->num_instances() * sizeof(int), cudaMemcpyDeviceToHost);

	printf("\nh_distanceMatrix:\n");
	for (int i = 0; i < dataset->num_instances(); i++) {
		for (int j = 0; j < dataset->num_instances(); j++) {
			printf("%f, ", h_distanceMatrix[i*dataset->num_instances() + j]);
		}
		printf("\n");
	}

	printf("actual classes host:\n");
	int *h_actualClasses = (int *)malloc(dataset->num_instances() * sizeof(int));
	for (int i = 0; i < dataset->num_instances(); i++) {
		h_actualClasses[i] = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
		printf("%d, ", h_actualClasses[i]);
	}

	int *d_actualClasses;
	cudaMalloc(&d_actualClasses, dataset->num_instances() * sizeof(int));

	cudaMemcpy(d_actualClasses, h_actualClasses, dataset->num_instances() * sizeof(int), cudaMemcpyHostToDevice);

	// float **d_smallestK;
	float *d_smallestK;
	cudaMalloc(&d_smallestK, k * sizeof(float));
	//	TODO openMP for loop
	for (int i = 0; i < 1; i++) { // dataset->num_instances()
		float *h_distanceRow = (float *)malloc(dataset->num_instances() * sizeof(float));
		//	TODO openMP for loop

		printf("\nh_distanceRow:\n");
		for (int j = 0; j < dataset->num_instances(); j++) {
			h_distanceRow[j] = h_distanceMatrix[i*dataset->num_instances() + j];
			printf("%f, ", h_distanceRow[j]);
		}
		printf("\n");

		float *d_distanceRow;
		cudaMalloc(&d_distanceRow, dataset->num_instances() * sizeof(float));
		// for (int j = 0; j < dataset->num_instances(); j++) {
		// 	d_distanceRow[j] = d_distanceMatrix[i*dataset->num_instances() + j];
		// }

		cudaMemcpy(d_distanceRow, h_distanceRow, dataset->num_instances() * sizeof(float), cudaMemcpyHostToDevice);

		// for (int j = 0; j < k; j++) {
		// 	deviceFindMinKNonOptimized<<<gridSize, blockSize>>>(d_smallestK[j], d_distanceRow, d_actualClasses, dataset->num_instances(), k);
		// }
		deviceFindMinK<<<blocksPerGrid, threadsPerBlock>>>(d_smallestK, d_distanceRow, d_actualClasses, dataset->num_instances(), k);
		// deviceFindMinK<<<gridSize, blockSize>>>(d_smallestK, d_distanceRow, d_actualClasses, dataset->num_instances(), k);

		cudaError_t cudaError = cudaGetLastError();

		if(cudaError != cudaSuccess) {
			fprintf(stderr, "post deviceFindMinK cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
			exit(EXIT_FAILURE);
		}

		float *h_smallestK = (float *)malloc(k * sizeof(float));
		cudaMemcpy(h_smallestK, d_smallestK, k * sizeof(float), cudaMemcpyDeviceToHost);

		printf("\n smallest distances:\n");
		for (int j = 0; j < k; j++) {
			printf("j: %d, distance: %f, class: %d\n", j, h_smallestK[j], 0);
		}

		cudaError = cudaGetLastError();

		if(cudaError != cudaSuccess) {
			fprintf(stderr, "post memcopy cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
			exit(EXIT_FAILURE);
		}

		cudaFree(d_distanceRow);
	}

	cudaFree(d_smallestK);
	cudaFree(d_actualClasses);
	free(h_actualClasses);





	// knn(dataset, k, d_datasetArray, d_predictions);

	// matrixMul<<<gridSize, blockSize>>>(d_datasetArray, d_distanceMatrix, d_predictions, matrixSize);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU time to fill Distance Matrix %f ms\n", milliseconds);

	
	hostFindKNN(h_distanceMatrix, h_datasetArray, h_predictions, dataset->num_instances(), dataset->num_attributes(), k);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("time to make predictions %f ms\n", milliseconds);

	for (int i = 0; i < dataset->num_instances(); i++) {
		printf("actual: %d, predicted: %d\n", dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32(), h_predictions[i]);
	}

	cudaEventRecord(start);

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