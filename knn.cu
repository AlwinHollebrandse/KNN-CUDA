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

// Performs majority voting using the first k first elements of an array
__global__ void deviceKVoting(int &prediction, int k, int *d_smallestKClasses, int startingDistanceIndex) {
	// get max class
	int maxClass = 0;
	for (int i = startingDistanceIndex; i < k; i++) {
		if (d_smallestKClasses[i] > maxClass)
			maxClass = d_smallestKClasses[i];
	}

	int* classCounter = new int[maxClass + 1]; // TODO does something need to be freed?
	// thrust::device_vector<int> classCounter(maxClass + 1);
	for (int i = startingDistanceIndex; i < k; i++) {
        classCounter[d_smallestKClasses[i]]++;
	}
	
	int voteResult = -1;
    int numberOfVotes = -1;
    for (int i = 0; i <= maxClass; i++) {
        if (classCounter[i] > numberOfVotes) {
            numberOfVotes = classCounter[i];
            voteResult = i;
        }
    }

	
    // std::map<int, int> classCounter; // TODO somehow use a map in device
    // for (int i = startingDistanceIndex; i < k; i++) {
    //     classCounter[d_smallestKClasses[i]]++;
    // }

    // int voteResult = -1;
    // int numberOfVotes = -1;
    // for (auto i : classCounter) {
    //     if (i.second > numberOfVotes) {
    //         numberOfVotes = i.second;
    //         voteResult = i.first;
    //     }
    // }

    prediction = voteResult;
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

// Uses a shared memory reduction approach to find the smallest k values
__global__ void deviceFindMinK(float *d_smallestK, int *d_smallestKClasses, float *d_distanceMatrix, int startingDistanceIndex, int yIndex, int *d_actualClasses, int numInstances, int k) {
	if (threadIdx.x == 0) { // TODO remove startingDistanceIndex and replace yIndex with tidY replace startingDistanceIndex with tidY * numInstances
		printf("\nin deviceFindMinK %d:\n", yIndex);
	}
	__shared__ float sharedDistanceMemory[THREADSPERBLOCK];
	__shared__ int sharedClassMemory[THREADSPERBLOCK];

	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	sharedDistanceMemory[threadIdx.x] = (tid < numInstances) ? d_distanceMatrix[startingDistanceIndex + tid] : FLT_MAX; // NOTE startingDistanceIndex is used to accesss the proper "row" of the matrix
	sharedClassMemory[threadIdx.x] = (tid < numInstances) ? d_actualClasses[tid] : -1;
	
	__syncthreads();
	
	if (tid == 0) {
		printf("\n pre shared mem. k = %d:\n", k);
		for (int i = 0; i < THREADSPERBLOCK; i++) {
			printf("%f, ", sharedDistanceMemory[i]);
		}
		
		printf("\n pre shared class mem. k = %d:\n", k); // TODO why are these all FLT_MAX?????
		for (int i = 0; i < THREADSPERBLOCK; i++) {
			printf("%d, ", sharedClassMemory[i]);
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
				thrust::sort_by_key(thrust::seq, sharedDistanceMemory + leftIndex, sharedDistanceMemory + leftIndex + k, sharedClassMemory + leftIndex);
				int actualEndingIndex = rightIndex + k;
				if (actualEndingIndex >= THREADSPERBLOCK)
					actualEndingIndex = THREADSPERBLOCK;
				thrust::sort_by_key(thrust::seq, sharedDistanceMemory + rightIndex, sharedDistanceMemory + actualEndingIndex, sharedClassMemory + rightIndex);
			}

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

			for (int i = 0; i < k; i++) {
				sharedDistanceMemory[threadIdx.x + i] = result[i];
				sharedClassMemory[threadIdx.x + i] = resultClasses[i];
			}
		}
		
		prevS = s;

		__syncthreads();
	}

	if (tid == 0) {
		printf("\n block final smallest k:\n");
		for (int i = 0; i < k; i++) {
			printf("%f, ", sharedDistanceMemory[i]);
		}
		printf("\n\n");
		printf("\n block final classes of smallest k:\n");
		for (int i = 0; i < k; i++) {
			printf("%d, ", sharedClassMemory[i]);
		}
		printf("\n");
	}

	// write your nearestK to global mem
	if (tid == 0) {
		int startingKIndex = k * yIndex;
		int endingKIndex = startingKIndex + k;
		int j = 0;
		for (int i = startingKIndex; i < endingKIndex; i++) {
			d_smallestK[i] = sharedDistanceMemory[j];
			d_smallestKClasses[i] = sharedClassMemory[j];
			j++;
		}
	}
}


// TODO call this with the correct dimensions
__global__ void makePredicitions(int *d_predictions, float *d_smallestK, int *d_smallestKClasses, int sizeOfSmallest, int startingDistanceIndex, int yIndex, int k) {
	if (threadIdx.x == 0) { // TODO remove startingDistanceIndex and replace yIndex with tidY replace startingDistanceIndex with tidY * k
		printf("\nin makePredicitions %d:\n", yIndex);
	}

	int tid = blockIdx.x*blockDim.x + threadIdx.x;

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

	if (tid == 0) {
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

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU time to fill Distance Matrix %f ms\n", milliseconds);



	// this is all for the matrix reduction TODO delete?
	int blocksPerGrid = (dataset->num_instances() + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
	printf("THREADSPERBLOCK: %d, blocksPerGrid: %d, blockSize: %d, gridSize: %d\n", THREADSPERBLOCK, blocksPerGrid, blockSize, gridSize);
	printf("blockSize: %d, gridSize: %d, THREADSPERBLOCK: %d, blocksPerGrid: %d\n", blockSize, gridSize, THREADSPERBLOCK, blocksPerGrid);


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
	// cudaMallocHost(&h_actualClasses, dataset->num_instances() * sizeof(int));
	for (int i = 0; i < dataset->num_instances(); i++) {
		h_actualClasses[i] = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
	}
	for (int i = 0; i < dataset->num_instances(); i++) {
		printf("%d, ", h_actualClasses[i]);
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

	//	TODO openMP for loop
	for (int i = 0; i < dataset->num_instances(); i++) { // dataset->num_instances() // TODO replace loop with tid_y. see slack.
		deviceFindMinK<<<blocksPerGrid, THREADSPERBLOCK>>>(d_smallestK, d_smallestKClasses, d_distanceMatrix, i * dataset->num_instances(), i, d_actualClasses, dataset->num_instances(), k);

		cudaError_t cudaError = cudaGetLastError();
		if(cudaError != cudaSuccess) {
			fprintf(stderr, "post deviceFindMinK cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
			exit(EXIT_FAILURE);
		}
		// printf("\n smallest distances:\n");
		// for (int j = 0; j < k; j++) {
		// 	printf("j: %d, distance: %f, class: %d\n", j, h_smallestK[j], 0);
		// }
	}

	// cudadevicesynchronize(); // wait for smallestK's to be filled TODO needed?

	int sizeOfSmallest = k * blocksPerGrid * dataset->num_instances();
	for (int i = 0; i < dataset->num_instances(); i++) { // dataset->num_instances() // TODO replace loop with tid_y. see slack.
		// TODO change dimensions
		makePredicitions<<<blocksPerGrid, THREADSPERBLOCK>>>(d_predictions, d_smallestK, d_smallestKClasses, sizeOfSmallest, i * k, i, k);

		cudaError_t cudaError = cudaGetLastError();
		if(cudaError != cudaSuccess) {
			fprintf(stderr, "post makePredicitions cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
			exit(EXIT_FAILURE);
		}
	}

	cudaMemcpy(h_predictions, d_predictions, dataset->num_instances() * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("time to make predictions %f ms\n", milliseconds);






	// knn(dataset, k, d_datasetArray, d_predictions);

	// matrixMul<<<gridSize, blockSize>>>(d_datasetArray, d_distanceMatrix, d_predictions, matrixSize);

	
	// hostFindKNN(h_distanceMatrix, h_datasetArray, h_predictions, dataset->num_instances(), dataset->num_attributes(), k);

	for (int i = 0; i < dataset->num_instances(); i++) {
		printf("actual: %d, predicted: %d\n", h_actualClasses[i], h_predictions[i]);
	}

	// Compute the confusion matrix
	int* confusionMatrix = computeConfusionMatrix(h_predictions, dataset);
	// Calculate the accuracy
	float accuracy = computeAccuracy(confusionMatrix, dataset);

	printf("The KNN classifier for %lu instances with k=%d had an accuracy of %.4f\n", dataset->num_instances(), k, accuracy);

	// // Verify that the result matrix is correct
	// for (int i = 0; i < numElements; i++)
	// 	if (fabs(h_predictions[i] - h_predictions_CPUres[i]) > 1e-3)
	// 	{
	// 		fprintf(stderr, "Result verification failed at element %d, %f vs %f!\n", i, h_predictions[i], h_predictions_CPUres[i]);
	// 		exit(EXIT_FAILURE);
	// 	}

	// printf("Multiplication of the matrixes was OK\n");

	// Free device global memory
	cudaFree(d_datasetArray);
	cudaFree(d_predictions);
	cudaFree(d_smallestK);
	cudaFree(d_smallestKClasses);
	cudaFree(d_actualClasses);

	// Free host memory
	free(h_datasetArray);
	free(h_predictions);
	free(h_predictions_CPUres);
	free(h_actualClasses);

	return 0;
}