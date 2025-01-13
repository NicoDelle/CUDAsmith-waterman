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

int main()
{
    std::vector<std::vector<int>> matrix(MATRIX_SIZE, std::vector<int>(MATRIX_SIZE));

    readMatrixFromFile("scoreMatrix.txt", matrix);

    int row1, col1, row2, col2;
    std::cout << "Enter coordinates for the first element (row and column): ";
    std::cin >> row1 >> col1;
    std::cout << "Enter coordinates for the second element (row and column): ";
    std::cin >> row2 >> col2;

    if (row1 >= 0 && row1 < MATRIX_SIZE && col1 >= 0 && col1 < MATRIX_SIZE &&
        row2 >= 0 && row2 < MATRIX_SIZE && col2 >= 0 && col2 < MATRIX_SIZE)
    {
        std::cout << "Element at (" << row1 << ", " << col1 << "): " << matrix[row1][col1] << std::endl;
        std::cout << "Element at (" << row2 << ", " << col2 << "): " << matrix[row2][col2] << std::endl;
    }
    else
    {
        std::cerr << "Invalid coordinates!" << std::endl;
    }

    return 0;
}