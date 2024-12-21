#include "src/smithWatermanSeq.h"
#include "src/smithWatermanPar.h"
#include <tuple>
#include <fstream>


template <typename T>
void allocateMatrix(T**& matrix, int rows, int cols);
template <typename T>
void allocateTensor(T***& tensor, int depth, int rows, int cols);
std::tuple<int, int> compareCigars(u_int16_t **cigarSeq, u_int16_t **cigarPar);
void saveToFile(const std::string filename, u_int16_t **matrix, int totRows, int totCols);

int main()
{
	char **h_query;
	char **h_reference;
    u_int16_t **cigarSeq, **cigarPar;
	allocateMatrix(h_query, N, S_LEN);
	allocateMatrix(h_reference, N, S_LEN);
	allocateMatrix(cigarSeq, N, 2*S_LEN);
	allocateMatrix(cigarPar, N, 2*S_LEN);
    
	char alphabet[] = {'A', 'T', 'C', 'G', 'N'};
	for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < S_LEN; j++)
        {
            h_query[i][j] = alphabet[rand() % 5];
            h_reference[i][j] = alphabet[rand() % 5];
        }
    }

	u_int16_t ***directionTensorPar = smithWatermanPar(h_query, h_reference, cigarPar);
    u_int16_t **directionMatrixSeq = smithWatermanSeq(h_query, h_reference, cigarSeq);

    saveToFile("CigarSeq.txt", cigarSeq, N, 2*S_LEN);
    saveToFile("CigarPar.txt", cigarPar, N, 2*S_LEN);
    saveToFile("LastDirectionMatrixPar.txt", directionTensorPar[0], S_LEN+1, S_LEN +1);
    saveToFile("LastDirectionMatrixSeq.txt", directionMatrixSeq, S_LEN+1, S_LEN+1);

    int errSeq, errIdx;
    //std::tie(errSeq, errIdx) = compareCigars(cigarSeq, cigarPar);
    //std::cout << "Mismatch found in sequence " << errSeq << " at " <<  errIdx << std::endl;


	delete[] h_query[0];
	delete[] h_query;
	delete[] h_reference[0];
	delete[] h_reference;
	delete[] cigarSeq[0];
	delete[] cigarSeq;
	delete[] cigarPar[0];
	delete[] cigarPar;
}

template <typename T>
void allocateMatrix(T**& matrix, int rows, int cols)
{
    matrix = new T*[rows];
    matrix[0] = new T[rows*cols];
    for (int i = 1; i < rows; i++)
    {
        matrix[i] = matrix[i-1] + cols;
    }
}

template <typename T>
void allocateTensor(T***& tensor, int depth, int rows, int cols)
{
    tensor = new T**[depth];
    tensor[0] = new T*[depth * rows];
    tensor[0][0] = new T[depth * rows * cols];

    for (int d = 0; d < depth; ++d)
    {
        if (d > 0)
        {
            tensor[d] = tensor[d - 1] + rows;
        }
        for (int r = 0; r < rows; ++r)
        {
            if (d == 0 && r > 0)
            {
                tensor[0][r] = tensor[0][r - 1] + cols;
            }
            else if (d > 0)
            {
                tensor[d][r] = tensor[d - 1][r] + cols;
            }
        }
    }
}

std::tuple<int, int> compareCigars(u_int16_t **cigarSeq, u_int16_t **cigarPar)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < 2*S_LEN; j++)
        {
            if (cigarSeq[i][j] != cigarPar[i][j])
            {
                std::cout << "Expected " << cigarSeq[i][j] << ", found " << cigarPar[i][j] << std::endl;
                return std::make_tuple(i, j);
            }
        }
    }
    return std::make_tuple(-1, -1);
}

void saveToFile(const std::string filename, u_int16_t **cigar, int totRows, int totCols)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file " << filename << std::endl;
    }

    for (int i = 0; i < totRows; i++)
    {
        for (int j = 0; j < totCols; j++)
        {
            file << cigar[i][j] << ", ";
        }
        file << std::endl;
    }

    file.close();
    std::cout << "Matrix saved sucessfully to " << filename << std::endl;
}