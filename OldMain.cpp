#include "smithWatermanSeq.h"
#include "smithWatermanPar.h"
#include <tuple>
#include <fstream>
#include <chrono>
#include <iostream>

template <typename T>
void allocateMatrix(T**& matrix, int rows, int cols);
std::tuple<int, int> compareCigars(u_int16_t **cigarSeq, u_int16_t **cigarPar);

int main()
{
	char **h_query;
	char **h_reference;
    u_int16_t **cigarSeq, **cigarPar;
	allocateMatrix(h_query, N, S_LEN);
	allocateMatrix(h_reference, N, S_LEN);
	allocateMatrix(cigarSeq, N, 2*S_LEN);
	allocateMatrix(cigarPar, N, 2*S_LEN);
    
	char alphabet[] = {'T', 'A', 'C', 'G', 'N'};
	for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < S_LEN; j++)
        {
            h_query[i][j] = alphabet[rand() % 5];
            h_reference[i][j] = alphabet[rand() % 5];
        }
    }

    auto startSeq = std::chrono::high_resolution_clock::now();
    smithWatermanSeq(h_query, h_reference, cigarSeq);
    auto endSeq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSeq = endSeq - startSeq;
    std::cout << "Sequential execution time: " << durationSeq.count() << " seconds" << std::endl;


    auto startPar = std::chrono::high_resolution_clock::now();
	smithWatermanPar(h_query, h_reference, cigarPar);
    auto endPar = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationPar = endPar - startPar;
    int errSeq, errIdx;
    std::tie(errSeq, errIdx) = compareCigars(cigarSeq, cigarPar);
    std::cout << "Parallel execution time: " << durationPar.count() << " seconds" << std::endl;

    if (errSeq != -1)
        std::cout << "Mismatch found in sequence " << errSeq << " at " <<  errIdx << std::endl;
    else std::cout << "All cigars match" << std::endl;

    // CLEANUP
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