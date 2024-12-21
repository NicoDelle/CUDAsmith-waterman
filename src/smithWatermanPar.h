#include <cuda_runtime.h>
#include <iostream>
#include <tuple>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUDA_KERNEL_CHECK() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA kernel launch error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
        err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA kernel execution error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define S_LEN 512
#define N 1000

#define NUM_THREADS 512
#define NUM_BLOCKS 1000
#define MATCH 1
#define MISMATCH -1
#define DEL -2
#define INS -2

//function definitions

template <typename T>
__host__ void allocateMatrix(T**& matrix, int rows, int cols);
template <typename T>
__host__ void allocateTensor(T***& tensor, int depth, int rows, int cols);
template <typename T>
__host__ std::tuple<int,int> maxElement(T**& matrix);
__host__ void backtraceP(u_int16_t *simple_rev_cigar, u_int16_t **dir_mat, int i, int j, int max_cigar_len);
__host__ u_int16_t ***smithWatermanPar(char **h_query, char **h_reference, u_int16_t **cigar);

//device only functions

__device__ __forceinline__ int mapToElement(int tid, int iteration);
__device__ __forceinline__ int leftNeighbor(int index);
__device__ __forceinline__ int upNeighbor(int index);
__device__ __forceinline__ int upLeftNeighbor(int index);
__device__ __forceinline__ int getActiveThreads(int iteration);
__device__ int max4(int a, int b, int c, int d);