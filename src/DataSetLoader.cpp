#include "ml/DataSetLoader.h"
#include "ml/Matrix.h"
#include "ml/Vector.h"

#include <sstream>
#include <fstream>


namespace machineLearning {


std::vector<double> DataSetLoader::parseCSVLine(const std::string& line) {
    std::vector<double> results;

    std::stringstream lineStream(line);
    std::string cell;

    while (std::getline(lineStream, cell, ',')) {
        if (cell.empty()) {
            throw std::invalid_argument("CSV contains an empty value");
        }

        results.push_back(std::stod(cell));
    }

    if (results.empty()) {
        throw std::invalid_argument("CSV line is empty");
    }

    return results;
}

Dataset DataSetLoader::loadCSV(const std::string& path, size_t targetColumn) {
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open CSV file");
    }

    std::vector<std::vector<double>> featureRows;
    std::vector<double> targets;

    std::string line;
    size_t expectedColumnCount = 0;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::vector<double> row = parseCSVLine(line);

        if (expectedColumnCount == 0) {
            expectedColumnCount = row.size();
        }

        if (row.size() != expectedColumnCount) {
            throw std::invalid_argument("CSV rows have inconsistent column counts");
        }

        if (targetColumn >= row.size()) {
            throw std::invalid_argument("Target column is out of range");
        }

        std::vector<double> featureRow;
        featureRow.reserve(row.size() - 1);

        double target = 0.0;

        for (size_t i = 0; i < row.size(); ++i) {
            if (i == targetColumn) {
                target = row[i];
            } else {
                featureRow.push_back(row[i]);
            }
        }

        featureRows.push_back(std::move(featureRow));
        targets.push_back(target);
    }

    if (featureRows.empty()) {
        throw std::invalid_argument("CSV file contains no data");
    }

    const size_t rowCount = featureRows.size();
    const size_t colCount = featureRows[0].size();

    std::vector<double> flatData;
    flatData.reserve(rowCount * colCount);

    for (const auto& featureRow : featureRows) {
        if (featureRow.size() != colCount) {
            throw std::invalid_argument("Feature rows have inconsistent column counts");
        }

        for (double value : featureRow) {
            flatData.push_back(value);
        }
    }

    return Dataset{
        Matrix(rowCount, colCount, std::move(flatData)),
        Vector(std::move(targets))
    };
}

}