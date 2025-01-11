#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

const int MATRIX_SIZE = 513;

void readMatrixFromFile(const char* filename, std::vector<std::vector<int>>& matrix)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Unable to open file " << filename << std::endl;
        return;
    }

    std::string line;
    for (int i = 0; i < MATRIX_SIZE; ++i)
    {
        std::getline(file, line);
        std::stringstream ss(line);
        std::string value;
        for (int j = 0; j < MATRIX_SIZE; ++j)
        {
            std::getline(ss, value, ',');
            matrix[i][j] = std::stoi(value);
        }
    }

    file.close();
}

void compareMatrices(const std::vector<std::vector<int>>& matrix1, const std::vector<std::vector<int>>& matrix2)
{
    int diffCount = 0;
    for (int i = 0; i < MATRIX_SIZE; ++i)
    {
        for (int j = 0; j < MATRIX_SIZE; ++j)
        {
            if (matrix1[i][j] != matrix2[i][j])
            {
                std::cout << "Difference at (" << i << ", " << j << "): " << matrix1[i][j] << " != " << matrix2[i][j] << std::endl;
                diffCount++;
            }
        }
    }
    std::cout << "Total number of differences: " << diffCount << std::endl;
}

int main()
{
    std::vector<std::vector<int>> matrix1(MATRIX_SIZE, std::vector<int>(MATRIX_SIZE));
    std::vector<std::vector<int>> matrix2(MATRIX_SIZE, std::vector<int>(MATRIX_SIZE));

    readMatrixFromFile("scoreMatrix.txt", matrix1);
    readMatrixFromFile("scoring_matrix.txt", matrix2);

    compareMatrices(matrix1, matrix2);

    return 0;
}