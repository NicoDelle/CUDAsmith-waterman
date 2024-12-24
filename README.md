# GPU 101 project: implementation of the smith-waterman algorithm in CUDA C
This repository contains a sequential and a parallel implementation of the smith-waterman algorithm.
The latter is meant to run on NVIDIA GPUs, and is written making use of the CUDA C framework.
Both implementations reside in the <code>src</code> folder and can be compiled as static libraries (the makefile should take care of everything). 
<code>main.cpp</code> shows the usage for both versions, and tests the results yielded by the parallel algorithm against the ones produced sequentially.

## How does the parallel algorithm work?
### First layer of parallelism
Since the input for the algorithm is given as <code> N = 1000</code> couples of <code> S_LEN = 512</code> characters long sequences, a first level of parallelism can be introduced computing the algorithm independently for each couple.
This is achieved by dedicating one block each, and launching 1000 blocks. The efficiency of this approach mainly depends on the device capabilities: a GPU with more streaming multiprocessors will be able to execute more blocks in parallel.

### Second layer of parallelism
Further parallelization is introduced within each block.
Since the value of element <code>i,j</code> of the direction and score matrices depend only on values in position <code>i-1,j</code>, <code>i,j-1</code>, <code>i-1,j-1</code>, these matrices can be computed diagonally.
The matrix is computed with <code>2*S_LEN=1024</code> iterations:
- The first row and column of the matrices are set to 0 and are used to provide room for the algorithm: the computation starts from the element in position <code>1,1</code>
- During the first iteration, only one thread will be considered valid, and it will compute value <code>1,1</code>
- During the second iteration, the now two valid threads will compute values <code>2,1</code> and <code>1,2</code>
- And so on. This way, every element on the same diagonal is computed at the same time , and the matrix is completed from the top left corner to the bottom right one

The rest of the algorithm consists of retrieving the path for the best match, and is computed sequentially

## Performance comparison
With the settings discussed above, the parallel algorithm takes around 2.11 s on a GTX 1650 mobile GPU paired with a ryzen 5 5600H, while the sequential one takes approximately 4.01 seconds to compute.
This means the parallel algorithm gives almost a 50% performance gain.
