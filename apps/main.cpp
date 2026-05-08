#include "ml/data/DataSetLoader.h"
#include "ml/data/DataSplitter.h"
#include "ml/layers/DenseLayer.h"
#include "ml/activations/ReLUActivation.h"
#include "ml/activations/SigmoidActivation.h"
#include "ml/core/Matrix.h"
#include "ml/core/Vector.h"
#include "ml/training/StandardScaler.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace machineLearning;

Vector rowToVector(const Matrix& matrix, size_t row) {
    std::vector<double> values;
    values.reserve(matrix.colCount());

    for (size_t j = 0; j < matrix.colCount(); ++j) {
        values.push_back(matrix(row, j));
    }

    return Vector(std::move(values));
}

double binaryCrossEntropy(double prediction, double target) {
    const double eps = 1e-9;
    prediction = std::clamp(prediction, eps, 1.0 - eps);
    return -(target * std::log(prediction) + (1.0 - target) * std::log(1.0 - prediction));
}

double positiveRate(const Vector& targets) {
    size_t positives = 0;

    for (size_t i = 0; i < targets.size(); ++i) {
        if (targets[i] == 1.0) {
            ++positives;
        }
    }

    return static_cast<double>(positives) / targets.size();
}

double averagePrediction(DenseLayer& layer1,
                         ReLUActivation& relu,
                         DenseLayer& layer2,
                         SigmoidActivation& sigmoid,
                         const Matrix& features) {
    if (features.rowCount() == 0) {
        return 0.0;
    }

    double sum = 0.0;

    for (size_t i = 0; i < features.rowCount(); ++i) {
        Vector x = rowToVector(features, i);
        Vector z1 = layer1.forward(x);
        Vector a1 = relu.forward(z1);
        Vector z2 = layer2.forward(a1);
        Vector p = sigmoid.forward(z2);
        sum += p[0];
    }

    return sum / static_cast<double>(features.rowCount());
}

double percentPredictedOne(DenseLayer& layer1,
                           ReLUActivation& relu,
                           DenseLayer& layer2,
                           SigmoidActivation& sigmoid,
                           const Matrix& features) {
    if (features.rowCount() == 0) {
        return 0.0;
    }

    size_t predictedOnes = 0;

    for (size_t i = 0; i < features.rowCount(); ++i) {
        Vector x = rowToVector(features, i);
        Vector z1 = layer1.forward(x);
        Vector a1 = relu.forward(z1);
        Vector z2 = layer2.forward(a1);
        Vector p = sigmoid.forward(z2);

        if (p[0] >= 0.5) {
            ++predictedOnes;
        }
    }

    return static_cast<double>(predictedOnes) / features.rowCount();
}

double minPrediction(DenseLayer& layer1,
                     ReLUActivation& relu,
                     DenseLayer& layer2,
                     SigmoidActivation& sigmoid,
                     const Matrix& features) {
    if (features.rowCount() == 0) {
        return 0.0;
    }

    double minValue = std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < features.rowCount(); ++i) {
        Vector x = rowToVector(features, i);
        Vector z1 = layer1.forward(x);
        Vector a1 = relu.forward(z1);
        Vector z2 = layer2.forward(a1);
        Vector p = sigmoid.forward(z2);
        minValue = std::min(minValue, p[0]);
    }

    return minValue;
}

double maxPrediction(DenseLayer& layer1,
                     ReLUActivation& relu,
                     DenseLayer& layer2,
                     SigmoidActivation& sigmoid,
                     const Matrix& features) {
    if (features.rowCount() == 0) {
        return 0.0;
    }

    double maxValue = -std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < features.rowCount(); ++i) {
        Vector x = rowToVector(features, i);
        Vector z1 = layer1.forward(x);
        Vector a1 = relu.forward(z1);
        Vector z2 = layer2.forward(a1);
        Vector p = sigmoid.forward(z2);
        maxValue = std::max(maxValue, p[0]);
    }

    return maxValue;
}

double evaluateAccuracy(DenseLayer& layer1,
                        ReLUActivation& relu,
                        DenseLayer& layer2,
                        SigmoidActivation& sigmoid,
                        const Matrix& features,
                        const Vector& targets) {
    size_t correct = 0;
    const size_t rows = features.rowCount();

    for (size_t i = 0; i < rows; ++i) {
        Vector x = rowToVector(features, i);
        Vector z1 = layer1.forward(x);
        Vector a1 = relu.forward(z1);
        Vector z2 = layer2.forward(a1);
        Vector yhatVec = sigmoid.forward(z2);
        double yhat = yhatVec[0];
        int prediction = yhat >= 0.5 ? 1 : 0;
        int target = static_cast<int>(targets[i]);

        if (prediction == target) {
            ++correct;
        }
    }

    return static_cast<double>(correct) / rows;
}

int main() {
    try {
        const std::string path = "assets/User_Data.csv";
        const size_t targetColumn = 2;
        const double trainRatio = 0.6;
        const double valRatio = 0.2;
        const double testRatio = 0.2;
        const double learningRate = 0.005;
        const size_t epochs = 1000;
        const size_t hiddenSize = 16;

        std::cout << "Loading dataset from: " << path << "\n";
        Dataset dataset = DataSetLoader::loadCSV(path, targetColumn);

        std::cout << "Splitting dataset (train/val/test): " << trainRatio << "/"
                  << valRatio << "/" << testRatio << "\n";
        DatasetSplit split = DataSplitter::split(dataset, trainRatio, valRatio, testRatio);

        std::cout << "Train samples: " << split.train.features.rowCount() << "\n";
        std::cout << "Validation samples: " << split.validation.features.rowCount() << "\n";
        std::cout << "Test samples: " << split.test.features.rowCount() << "\n";
        std::cout << "Feature count: " << split.train.features.colCount() << "\n";
        const double trainPosRate = positiveRate(split.train.targets);
        const double trainNegRate = 1.0 - trainPosRate;
        const double classWeightPos = trainNegRate;
        const double classWeightNeg = trainPosRate;

        std::cout << "Train positive rate: " << std::fixed << std::setprecision(2)
                  << trainPosRate * 100.0 << "%\n";
        std::cout << "Val positive rate: " << std::fixed << std::setprecision(2)
                  << positiveRate(split.validation.targets) * 100.0 << "%\n";
        std::cout << "Class weights: pos=" << classWeightPos << " neg=" << classWeightNeg << "\n\n";

        StandardScaler scaler;
        scaler.fit(split.train.features);

        Matrix X_train = scaler.transform(split.train.features);
        Matrix X_val = scaler.transform(split.validation.features);
        Matrix X_test = scaler.transform(split.test.features);

        const size_t inputSize = X_train.colCount();

        DenseLayer layer1(inputSize, hiddenSize);
        ReLUActivation relu(hiddenSize);
        DenseLayer layer2(hiddenSize, 1);
        SigmoidActivation sigmoid(1);

        for (size_t epoch = 1; epoch <= epochs; ++epoch) {
            double epochLoss = 0.0;

            for (size_t i = 0; i < X_train.rowCount(); ++i) {
                Vector x = rowToVector(X_train, i);
                double target = split.train.targets[i];

                Vector z1 = layer1.forward(x);
                Vector a1 = relu.forward(z1);
                Vector z2 = layer2.forward(a1);
                Vector yhatVec = sigmoid.forward(z2);
                double yhat = yhatVec[0];

                const double sampleWeight = target == 1.0 ? classWeightPos : classWeightNeg;
                double loss = -(classWeightPos * target * std::log(std::clamp(yhat, 1e-9, 1.0))
                                + classWeightNeg * (1.0 - target) * std::log(std::clamp(1.0 - yhat, 1e-9, 1.0)));
                epochLoss += loss;

                Vector dZ2(std::vector<double>{(yhat - target) * sampleWeight});
                Vector dA1 = layer2.backward(dZ2);
                Vector dZ1 = relu.backward(dA1);

                layer2.applyGradients(learningRate);
                layer1.applyGradients(learningRate);

                if (epoch == 1 && i == 0) {
                    std::cout << "--- First sample details ---\n";
                    std::cout << "Target: " << target << "\n";
                    std::cout << "Z1: ";
                    for (size_t j = 0; j < z1.size(); ++j) {
                        std::cout << z1[j] << " ";
                    }
                    std::cout << "\nA1: ";
                    for (size_t j = 0; j < a1.size(); ++j) {
                        std::cout << a1[j] << " ";
                    }
                    std::cout << "\nZ2: " << z2[0] << "\n";
                    std::cout << "Prediction: " << yhat << "\n";
                    std::cout << "Backward gradients: dZ2=" << dZ2[0] << "\n";
                    std::cout << "----------------------------\n";
                }
            }

            const double averageLoss = epochLoss / split.train.features.rowCount();
            if (epoch % 50 == 0 || epoch == 1 || epoch == epochs) {
                double valAccuracy = evaluateAccuracy(layer1, relu, layer2, sigmoid,
                                                      X_val,
                                                      split.validation.targets);
                double valPredictedOne = percentPredictedOne(layer1, relu, layer2, sigmoid, X_val);
                double valAverageProb = averagePrediction(layer1, relu, layer2, sigmoid, X_val);
                double valMinProb = minPrediction(layer1, relu, layer2, sigmoid, X_val);
                double valMaxProb = maxPrediction(layer1, relu, layer2, sigmoid, X_val);

                std::cout << "Epoch " << epoch
                          << " | avg loss = " << std::fixed << std::setprecision(4) << averageLoss
                          << " | val accuracy = " << std::setprecision(2) << valAccuracy * 100.0 << "%"
                          << " | predicted 1 rate = " << std::setprecision(2) << valPredictedOne * 100.0 << "%"
                          << " | avg prob = " << std::setprecision(4) << valAverageProb
                          << " | min prob = " << std::setprecision(4) << valMinProb
                          << " | max prob = " << std::setprecision(4) << valMaxProb
                          << "\n";
            }
        }

        double testAccuracy = evaluateAccuracy(layer1, relu, layer2, sigmoid,
                                               X_test,
                                               split.test.targets);

        std::cout << "\nTraining complete." << "\n";
        std::cout << "Final test accuracy: " << std::fixed << std::setprecision(2)
                  << testAccuracy * 100.0 << "%\n";

        std::cout << "\nSample prediction on test set:\n";
        if (X_test.rowCount() > 0) {
            Vector sample = rowToVector(X_test, 0);
            Vector z1 = layer1.forward(sample);
            Vector a1 = relu.forward(z1);
            Vector z2 = layer2.forward(a1);
            double yhat = sigmoid.forward(z2)[0];
            std::cout << "  raw output = " << z2[0]
                      << ", sigmoid = " << yhat
                      << ", class = " << (yhat >= 0.5 ? 1 : 0)
                      << "\n";
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }
}
