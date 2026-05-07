#pragma once

#include <vector>
#include <cstddef>

namespace machineLearning {

class Matrix {
private:
    size_t rows;
    size_t cols;

    std::vector<double> data;

public:
    Matrix(size_t rows_, size_t cols_, std::vector<double> data_);

    Matrix(size_t rows_, size_t cols_);

    size_t rowCount() const;
    size_t colCount() const;

    double operator()(size_t row, size_t col) const;
    double& operator()(size_t row, size_t col);

};
}