#include "ml/matrix.h"

#include <utility>
#include <stdexcept>

namespace machineLearning {

Matrix::Matrix(size_t rows_, size_t cols_, std::vector<double> data_)
    : rows(rows_), cols(cols_), data(std::move(data_))
{
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Invalid row or col count");
    }
    if (rows * cols != data.size()) {
        throw std::invalid_argument("Data size does not match row and column count");
    }
}

Matrix::Matrix(size_t rows_, size_t cols_)
    : rows(rows_), cols(cols_)
{
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Invalid row or col count");
    }
    data.resize(rows_ * cols_, 0.0);
}


size_t Matrix::rowCount() const {
    return rows;
}

size_t Matrix::colCount() const {
    return cols;
}


double Matrix::operator()(size_t row, size_t col) const {
    if (row > rowCount() - 1) {
        throw std::out_of_range("Row out of bounds of matrix");
    }
    if (col > colCount() - 1) {
        throw std::out_of_range("Col out of bounds of matrix");
    }
    return data[(row * colCount()) + col];
}

double& Matrix::operator()(size_t row, size_t col) {
    if (row > rowCount() - 1) {
        throw std::out_of_range("Row out of bounds of matrix");
    }
    if (col > colCount() - 1) {
        throw std::out_of_range("Col out of bounds of matrix");
    }
    return data[(row * colCount()) + col];
}

}