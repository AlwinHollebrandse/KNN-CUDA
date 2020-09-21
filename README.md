# 2020--603-A2-Hollebrandse-
CUDA KNN
 
NOTE: this kNN classifier predicts each data element in the provided dataset by using the remainder of the dataset.
NOTE: If the provided k is larger than the number of elements in the dataset, it gets set to the size of the dataset.
 
### Sample Input Arguments:
./datasets/large.arff 5
 
### Sample Output:
SOMETHING TODO
 
## Implementation Details:
This algorithm splits the kNN into 2 major tasks: computing a distance matrix, and then finding the kNN voting. More details can be found in the respective sections.
 
## Distance Matrix:
This part of the project was done with the GPU. The used block size and grid size are defined as follows: a 16 by 16 block and (dataset->num_instances() + threadsPerBlockDim - 1) / threadsPerBlockDim by (dataset->num_instances() + threadsPerBlockDim - 1) / threadsPerBlockDim grid. The magic happens in the `fillDistanceMatrix` method. Each thread gets a row and column value, where each variable represents a different data instance. If the two variables are the same, then that distance is set to the highest possible value as to not be picked during the kNN portion of the program. If they are different, then the euclidean distance between the two is computed through the use of a flattened dataset array that the device was given access to.
 
## kNN:
The kNN call (hostFindKNN) is computed as follows: for a given dataset instance, calculate the euclidean distance compared to each other dataset instances. Record these distances and the attached class of each value. Sort this resulting array in ascending order according to distance. Take the first `k` pairs of the sorted list and perform `kVoting` to get a prediction. Voting works but getting the count of each class in the kNN values. The class with the most votes is predicted. In the event of a tie, the first class encountered that had that vote amount is returned. This "finalPredicitons" array is then returned from the kNN method and compared to the dataset's actual classes to compute and report a final accuracy. In addition to reporting the accuracy, this code also reports the number of instances in the dataset and the amount of CPU time the kNN classifier took. The CPU time is computed by using "clock_gettime" functionality before and after the kNN call.
 
## Results:
TODO

