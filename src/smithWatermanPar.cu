#include "smithWatermanPar.h"
#include <fstream>

template <typename T>
__host__ void writeMatrixToFile(T** matrix, int rows, int cols, const char* filename)
{
    std::ofstream file(filename);
    if (file.is_open())
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                file << matrix[i][j];
                if (j < cols - 1)
                    file << ",";
            }
            file << "\n";
        }
        file.close();
    }
    else
    {
        std::cerr << "Unable to open file " << filename << std::endl;
    }
}

__global__ void smithWatermanKernel(
    char* d_query, 
    char* d_reference, 
    u_int32_t *d_score_tensor, 
    u_int16_t *d_direction_tensor
    )
{
    int tid = threadIdx.x;
    __shared__ u_int32_t scoreNeighbors[S_LEN];
    if (tid < S_LEN)
    {
        scoreNeighbors[tid] = 0;
    }
    __syncthreads();

    int index, comparison, tmp;
    int max = INS;
    u_int32_t upNeighbor, leftNeighbor, upLeftNeighbor;

    for (int iteration = 1; iteration < S_LEN * 2; iteration++)
    {
        if (tid < getActiveThreads(iteration)) 
        {
            index = mapToElement(tid, iteration) + blockIdx.x * (S_LEN+1) * (S_LEN+1);

            upLeftNeighbor = d_score_tensor[getUpLeftNeighbor(index)];

            if (iteration < S_LEN)
            {
                upNeighbor = scoreNeighbors[tid];
                leftNeighbor = d_score_tensor[getLeftNeighbor(index)];
            }
            else
            {
                upNeighbor = d_score_tensor[getUpNeighbor(index)];
                leftNeighbor = scoreNeighbors[tid];
            }
            
            //compute the algorithm given the neighbors
            comparison = ((d_query[mapToQueryIndex(tid, iteration) + S_LEN * blockIdx.x] == d_reference[mapToReferenceIndex(tid, iteration) + S_LEN * blockIdx.x]) ? MATCH : MISMATCH);
            tmp = max4(
                upLeftNeighbor + comparison, 
                upNeighbor + DEL,
                leftNeighbor + INS,
                0
            );

            if (tmp == (upLeftNeighbor + comparison))
                d_direction_tensor[index] = (comparison == MATCH) ? 1 : 2;
            else if (tmp == (upNeighbor + DEL))
                d_direction_tensor[index] = 3;
            else if (tmp == (leftNeighbor + INS))
                d_direction_tensor[index] = 4;
            else
                d_direction_tensor[index] = 0;
            
            d_score_tensor[index] = tmp;

            scoreNeighbors[tid] = tmp;
        }
        __syncthreads();
    }    
}

u_int16_t ***smithWatermanPar(char **h_query, char **h_reference, u_int16_t **cigar)
{
    u_int16_t ***h_direction_tensor;
    u_int32_t ***h_score_tensor;

    allocateTensor(h_direction_tensor, N, S_LEN + 1, S_LEN + 1);
    allocateTensor(h_score_tensor, N, S_LEN+1, S_LEN+1);

    char *d_query;
    char *d_reference;
    u_int32_t *d_score_tensor;
    u_int16_t *d_direction_tensor;

    CUDA_CHECK(cudaMalloc(&d_query, N*S_LEN*sizeof(char))); // 512 KB
    CUDA_CHECK(cudaMalloc(&d_reference, N*S_LEN*sizeof(char))); // 512 KB
    //Tensors to store all score/direction matrices at once
    CUDA_CHECK(cudaMalloc(&d_score_tensor, N*(S_LEN+1)*(S_LEN+1)*sizeof(u_int32_t))); // 1.052 GB
    CUDA_CHECK(cudaMalloc(&d_direction_tensor, N*(S_LEN+1)*(S_LEN+1)*sizeof(u_int16_t))); // 263 MB -> tot 1.365 GB

    CUDA_CHECK(cudaMemcpy(d_query, h_query[0], N*S_LEN*sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_reference, h_reference[0], N*S_LEN*sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_score_tensor, 0, N * (S_LEN + 1) * (S_LEN + 1) * sizeof(u_int32_t))); // problem if N > 1
    CUDA_CHECK(cudaMemset(d_direction_tensor, 0, N * (S_LEN + 1) * (S_LEN + 1) * sizeof(u_int16_t)));

    dim3 threadsPerBlock(NUM_THREADS, 1, 1);
    dim3 blocksPerGrid(NUM_BLOCKS, 1, 1);
    smithWatermanKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_query, 
        d_reference, 
        d_score_tensor, 
        d_direction_tensor
    );
    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_direction_tensor[0][0], d_direction_tensor, N*(S_LEN+1)*(S_LEN+1)*sizeof(u_int16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_score_tensor[0][0], d_score_tensor, N*(S_LEN+1)*(S_LEN+1)*sizeof(u_int32_t), cudaMemcpyDeviceToHost));


    int maxRow, maxCol;
    for (int i = 0; i < N; i++)
    {
        std::tie(maxRow, maxCol) = maxElement(h_score_tensor[i]);
        std::cout << "Max coords: " << maxRow << ", " << maxCol << std::endl;
        std::cout << "Max value: " << h_score_tensor[i][maxRow][maxCol] << std::endl; 
        backtraceP(cigar[i], h_direction_tensor[i], maxRow, maxCol, S_LEN*2);
    }
    writeMatrixToFile(h_score_tensor[N-1], S_LEN+1, S_LEN+1, "scoreMatrix.txt");

    //CLEANUP
    cudaFree(d_direction_tensor);
    cudaFree(d_score_tensor);
    cudaFree(d_query);
    cudaFree(d_reference);

    // Free the memory allocated for the tensors
    delete[] h_score_tensor[0][0];
    delete[] h_score_tensor[0];
    delete[] h_score_tensor;

    return h_direction_tensor;
}

template <typename T>
__host__ void allocateMatrix(T**& matrix, int rows, int cols)
{
    matrix = new T*[rows];
    matrix[0] = new T[rows*cols];
    for (int i = 1; i < rows; i++)
    {
        matrix[i] = matrix[i-1] + cols;
    }
}

template <typename T>
__host__ void allocateTensor(T***& tensor, int depth, int rows, int cols)
{
    // Allocate memory for the depth pointers
    tensor = new T**[depth];
    
    // Allocate memory for the row pointers for all depths
    for (int d = 0; d < depth; ++d)
    {
        tensor[d] = new T*[rows];
    }
    
    // Allocate memory for all elements in a single contiguous block
    T* dataBlock = new T[depth * rows * cols];

    // Set the pointers for each depth and row
    for (int d = 0; d < depth; ++d)
    {
        for (int r = 0; r < rows; ++r)
        {
            tensor[d][r] = dataBlock + (d * rows * cols) + (r * cols);
        }
    }
}

template <typename T>
__host__ std::tuple<int, int> maxElement(T**& matrix)
{
    T max = 0;
    int maxRow, maxCol;
    for (int i = 1; i < S_LEN + 1; i++)
    {
        for (int j = 1; j < S_LEN + 1; j++)
        {
            if (matrix[i][j] > max)
            {
                max = matrix[i][j];
                maxRow = i;
                maxCol = j;
            }
        }
    }
    return std::make_tuple(maxRow, maxCol);
}

void backtraceP(u_int16_t *simple_rev_cigar, u_int16_t **dir_mat, int i, int j, int max_cigar_len)
{
	int n;
	for (n = 0; n < max_cigar_len && dir_mat[i][j] != 0; n++)
	{
		int dir = dir_mat[i][j];
		if (dir == 1 || dir == 2)
		{
			i--;
			j--;
		}
		else if (dir == 3)
			i--;
		else if (dir == 4)
			j--;

		simple_rev_cigar[n] = dir;
	}
}

/**
 * @brief Given an iteration, maps a thread id to an element of the diagonal of sc_matrix and dir_matrix 
 */
__device__ __forceinline__ int mapToElement(int tid, int iteration)
{
    return (iteration <= S_LEN) ? tid + 1 + (S_LEN + 1) * (iteration - tid) : (S_LEN - tid) * (S_LEN + 1) + iteration - S_LEN + tid + 1;
}

__device__ __forceinline__ int mapToQueryIndex(int tid, int iteration)
{
    return ((iteration <= S_LEN) ? getActiveThreads(iteration) - tid - 1 : S_LEN - tid - 1);
}

__device__ __forceinline__ int mapToReferenceIndex(int tid, int iteration)
{
    return ((iteration <= S_LEN) ? tid : tid + iteration - S_LEN);
}

__device__ __forceinline__ int getLeftNeighbor(int index)
{
    return index - 1;
}

__device__ __forceinline__ int getUpNeighbor(int index)
{
    return index - S_LEN - 1;
}

__device__ __forceinline__ int getUpLeftNeighbor(int index)
{
    return index - S_LEN - 2;
}

/**
 * @brief Active threads for an iteration range from 1 to 512 and then back to 1
 */
__device__ __forceinline__ int getActiveThreads(int iteration)
{
    return (iteration <= S_LEN) ? iteration : 2 * S_LEN - iteration;
}

__device__ int max4(int n1, int n2, int n3, int n4)
{
	int tmp1, tmp2;
	tmp1 =  n1 > n2 ? n1 : n2;
	tmp2 = n3 > n4 ? n3 : n4;
	tmp1 = tmp1 > tmp2 ? tmp1 : tmp2;
	return tmp1;
}