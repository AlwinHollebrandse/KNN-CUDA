# 2020--603-A2-Hollebrandse-
CUDA KNN
NOTE: this kNN classifier predicts each data element in the provided dataset by using the remainder of the dataset.
NOTE: If the provided k is larger than the number of elements in the dataset, it gets set to the size of the dataset.
### Sample Input Arguments:
./datasets/small.arff 5
### Sample Output:
GPU time to fill Distance Matrix 0.053024 ms
time to make predictions 75.288063 ms
total GPU time: 75.341087 ms
The KNN classifier for 336 instances with k=5 had an accuracy of 0.8125
## Implementation Details:
This algorithm splits the kNN into 2 major tasks: computing a distance matrix, and then computing the kNN voting. More details can be found in the respective sections.
## Distance Matrix:
This part of the project was done with the GPU. The used block size and grid size are defined as follows: a 16 by 16 block and (dataset->num_instances() + threadsPerBlockDim - 1) / threadsPerBlockDim by (dataset->num_instances() + threadsPerBlockDim - 1) / threadsPerBlockDim grid. The `fillDistanceMatrix` method makes each thread get a row and column value, where each variable represents a different data instance. If the two variables are the same, then that distance is set to the highest possible value as to not be picked during the kNN portion of the program. If they are different, then the euclidean distance between the two is computed through the use of a flattened dataset array that the device was given access to.
 
## Making Predictions:
The process to fill predictions is two fold. Given a complete distance matrix, the first step is to reduce each row into the nearest k neighbors. The second step would then be to do voting. Inorder to reduce a distance matrix row into the nearest k neighbors, the kernel `deviceFindMinK` was set up. In the 2d kernel implementation, 256 by 4 blocks are created and fit into a  (reductionBlocksPerGridX, reductionBlocksPerGridY) grid (reductionBlocksPerGridX = (dataset->num_instances() + 256 - 1) / 256 and reductionBlocksPerGridY = (dataset->num_instances() + 4 - 1) / 4).The values 256 were chosen because they would in theory maximize the amount of work that could be done per instance while still looking at several data points at once. This would allow for decent parallelization in cases when the number of data points was either large or small. This kernel looks at 256 neighbors at a time within a single block. These neighbors are broken into "chunks" of size k. These "chunks" are then counted and the mid point is computed (where an odd number of "chunks" will be on the left side of the midpoint if they are not evenly divisible by two.) Each thread within the block is assigned to one of these left hand neighbor "chunks". These threads then merge their left cluster with its respective right hand side side cluster. For example, if a thread was assigned the first "chunk", its right cluster partner would be the first "chunk" past the midway point (s). If this is the first iteration of the merging process, each cluster is sorted in ascending order. This ensures that the merging process results in the smallest distances (and the attached classes). Merging happens by looking at the left most indexes of both chunks and then selecting the smaller of the two values. This repeats "k" times. Because these chunks are already sorted, the output will also be in order. The output is saved into a  temporary array and then overrides the values that the left "chunk" had. A new midpoint is then found within the former left hand clusters. This repeats per block until only a single cluster remains with the smallest distances and classes of its 256 neighbors. This happens in parallel for every block. All of the final block results per distance row, aka data instance, are then written to global memory. This is done by writing in adjacency matrix locations of the `d_smallestK` and `d_smallestKClasses`. For example, if a dataset had 300 instances it would require 2 horizontal blocks to cover all neighbors (ceiling division of 300/256 = 2). The first block of the first data instance would write values 0,(k-1) of the respective d_smallestK arrays. The second block, which also operates on the first data instance, would write its k nearest neighbors to the k,(k+k) positions of the arrays. These positions were calculated using tid_x and tid_y and the grid/block dimensions. Once this is finished for every kernel the `makePredictions` kernel would leave the GPU queue. Being called with the following dimensions: (predictionBlocksPerGridX (which is (numberOfKSegmentsPerRow + 1024 - 1) / 1024), and predictionBlocksPerGridY (which is (dataset->num_instances() + maxNumberOFInstancesToPredictAtOnce - 1) / maxNumberOFInstancesToPredictAtOnce)), this kernel combined the results of all blocks associated with a given data point. Continuing the previous example, this kernel would look at the first k * number of reductionBlocks needed to cover all neighbors. In this case it would look at the first 2k values in global memory and then perform the same merge functionality as the `deviceFindMinK` kernel. The final results of this process yield the actual nearest k neighbors of a given data instance. The classes with these values are then counted and used to vote and make the prediction for the data point in question. The class with the most votes is predicted. In the event of a tie, the first class encountered that had that vote amount is returned. This finalPredicitons array is then returned from the kNN method and compared to the dataset's actual classes to compute and report a final accuracy.
 
## Methods:
This code was first made with  1D kernel and then extended to a 2D version for considerable speedup. IN order to compute results, the accuracy of each run was compared to the results of a previous project that used the same datasets and a k-value of 5. To report time, a timer was set before and after each important GPU kernel. These times were reported independently and added together.
 
## Results:
Small Dataset:
GPU time to fill Distance Matrix 0.039104 ms
time to make predictions 82.289726 ms
total GPU time: 82.328827 ms
The KNN classifier for 336 instances with k=5 had an accuracy of 0.8125
 
Medium Dataset:
error
 
Large Dataset:
error
 
## Conclusion:
Given the lackluster results of this project, it is hard to make definitive claims. It can be said that given the large time discrepancy between the distance matrix filling and making predictions, it seems probable that the bug that caused the larger datasets to fail is within the one of the `Making Predictions` kernels. This is further supported by the fact that the distance matrix has the correct and expected values while the final predictions do not.
 
## Improvements:
The first thing that could be improved by this project would be finishing it. The biggest time sinks experienced programming wise were forgetting to sync_threads() when needed and reading the incorrect parts of both shared and global memory several times. Time was also lost to unfamiliar errors. For example, "failure to launch due to too many requested resources" when only print statements were added took a while to figure out. Another thing that went wrong was not sufficiently testing the 1D kernel version before converting to a 2d kernel. I assumed that because the 1D kernel had worked on the `superSmall` dataset, that it would work on the larger ones as a correctly programmed CUDA program is all about scalability. Given that the errors occur with larger dataset values the current culprit is most likely to do with memory accesses. My leading theory is that something got lost in translation with saving the results of each block to contiguous global memory. Another task to do is to experiment with different kernel size setups to compare timings with various dataset sizes. It should be noted that the `improvements` section of my last project is no longer there, as those changes were made after submission. 
 
 

