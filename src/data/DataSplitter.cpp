#include "ml/data/DataSplitter.h"

#include <cmath>
#include <vector>
#include <random>
#include <stdexcept>

namespace machineLearning {

DatasetSplit DataSplitter::split(const Dataset& data, double trainRatio, double valRatio, double testRatio) {
    const double totalRatio = trainRatio + valRatio + testRatio;
    if (trainRatio <= 0.0 || valRatio < 0.0 || testRatio <= 0.0 || std::abs(totalRatio - 1.0) > 1e-9) {
        throw std::invalid_argument("Ratios must be positive and sum to 1.0");
    }

    size_t n = data.features.rowCount();
    size_t cols = data.features.colCount();

   
    std::vector<size_t> indices(n);
    for (size_t i = 0; i < n; ++i) {
        indices[i] = i;
    }

    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);

    
    size_t trainSize = static_cast<size_t>(n * trainRatio);
    size_t valSize = static_cast<size_t>(n * valRatio);
    size_t testSize = n - trainSize - valSize;

    std::vector<double> trainData;
    std::vector<double> valData;
    std::vector<double> testData;
    std::vector<double> trainTargets;
    std::vector<double> valTargets;
    std::vector<double> testTargets;

    trainData.reserve(trainSize * cols);
    valData.reserve(valSize * cols);
    testData.reserve(testSize * cols);

    
    for (size_t i = 0; i < n; ++i) {
        size_t idx = indices[i];

        if (i < trainSize) {
            
            for (size_t j = 0; j < cols; ++j) {
                trainData.push_back(data.features(idx, j));
            }
            trainTargets.push_back(data.targets[idx]);
        } else if (i < trainSize + valSize) {
            
            for (size_t j = 0; j < cols; ++j) {
                valData.push_back(data.features(idx, j));
            }
            valTargets.push_back(data.targets[idx]);
        } else {
            
            for (size_t j = 0; j < cols; ++j) {
                testData.push_back(data.features(idx, j));
            }
            testTargets.push_back(data.targets[idx]);
        }
    }

    Dataset trainSet{
        Matrix(trainSize, cols, std::move(trainData)),
        Vector(std::move(trainTargets))
    };

    Dataset valSet{
        Matrix(valSize, cols, std::move(valData)),
        Vector(std::move(valTargets))
    };

    Dataset testSet{
        Matrix(testSize, cols, std::move(testData)),
        Vector(std::move(testTargets))
    };

    return {std::move(trainSet), std::move(valSet), std::move(testSet)};
}

}
