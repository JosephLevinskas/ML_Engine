#include "ml/Vector.h"
#include "ml/Matrix.h"
#include "ml/LinearModel.h"
#include "ml/LogisticModel.h"
#include "ml/Trainer.h"

#include <iostream>
#include <vector>
#include <iomanip>

using namespace machineLearning;

void printDivider() {
    std::cout << "----------------------------------------\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(4);

    printDivider();
    std::cout << "LINEAR REGRESSION DEMO\n";
    printDivider();

    // y = 3x1 + 5x2 + 2
    Matrix linearX(
        5,
        2,
        {
            1.0, 2.0,
            2.0, 1.0,
            3.0, 4.0,
            4.0, 3.0,
            5.0, 6.0
        }
    );

    Vector linearY({15.0, 13.0, 31.0, 29.0, 43.0});

    LinearModel linearModel(Vector({0.0, 0.0}), 0.0);

    Trainer trainer(0.01, 1000);

    TrainingResults linearResults = trainer.train(linearModel, linearX, linearY);

    std::cout << "Final Linear Loss: " << linearResults.losses.back() << "\n";

    Vector linearPred = linearResults.model.predict(linearX);

    std::cout << "Predictions vs Targets:\n";
    for (size_t i = 0; i < linearPred.size(); ++i) {
        std::cout << "Pred: " << linearPred[i]
                  << " | Target: " << linearY[i] << "\n";
    }

    printDivider();

    std::cout << "LOGISTIC REGRESSION DEMO\n";
    printDivider();

    // Simple separable dataset
    Matrix logisticX(
        6,
        1,
        {
            0.0,
            1.0,
            2.0,
            8.0,
            9.0,
            10.0
        }
    );

    Vector logisticY({0.0, 0.0, 0.0, 1.0, 1.0, 1.0});

    LogisticModel logisticModel(Vector({0.0}), 0.0);

    Trainer logisticTrainer(0.1, 2000);

    LogisticTrainingResults logisticResults =
        logisticTrainer.train(logisticModel, logisticX, logisticY);

    std::cout << "Final Logistic Loss: " << logisticResults.losses.back() << "\n";

    std::cout << "\nPredictions:\n";

    for (size_t i = 0; i < logisticX.rowCount(); ++i) {
        Vector x({logisticX(i, 0)});

        double prob = logisticResults.model.predictProbability(x);
        int cls = logisticResults.model.predictClass(x);

        std::cout << "x = " << x[0]
                  << " | prob = " << prob
                  << " | class = " << cls
                  << " | target = " << logisticY[i]
                  << "\n";
    }

    printDivider();

    std::cout << "GENERALIZATION TEST\n";
    printDivider();

    Vector test1({1.5});
    Vector test2({9.5});

    std::cout << "x = 1.5 -> prob: "
              << logisticResults.model.predictProbability(test1)
              << "\n";

    std::cout << "x = 9.5 -> prob: "
              << logisticResults.model.predictProbability(test2)
              << "\n";

    printDivider();

    std::cout << "DONE\n";

    return 0;
}