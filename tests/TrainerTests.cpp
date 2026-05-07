#include "ml/training/Trainer.h"
#include "ml/models/LinearModel.h"
#include "ml/training/LossFunctions.h"

#include <cassert>
#include <stdexcept>
#include <iostream>

using namespace machineLearning;

void test_constructor_rejects_zero_learning_rate() {
    bool threw = false;

    try {
        Trainer trainer(0.0, 10);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_constructor_rejects_negative_learning_rate() {
    bool threw = false;

    try {
        Trainer trainer(-0.01, 10);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_constructor_rejects_zero_epochs() {
    bool threw = false;

    try {
        Trainer trainer(0.01, 0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_training_reduces_mse() {
    Matrix X(2, 2, {
        1.0, 2.0,
        3.0, 4.0
    });

    Vector targets({8.0, 18.0});
    LinearModel initial(Vector({1.0, 2.0}), 0.0);

    double beforeLoss = meanSquaredError(initial.predict(X), targets);

    Trainer trainer(0.01, 10);
    TrainingResults result = trainer.train(initial, X, targets);

    double afterLoss = meanSquaredError(result.model.predict(X), targets);

    assert(afterLoss < beforeLoss);
}

void test_training_changes_weights_and_bias() {
    Matrix X(2, 2, {
        1.0, 2.0,
        3.0, 4.0
    });

    Vector targets({8.0, 18.0});
    LinearModel initial(Vector({1.0, 2.0}), 0.0);

    Trainer trainer(0.01, 10);
    TrainingResults result = trainer.train(initial, X, targets);

    const LinearModel& trained = result.model;

    assert(trained.getWeights()[0] != initial.getWeights()[0]);
    assert(trained.getWeights()[1] != initial.getWeights()[1]);
    assert(trained.getBias() != initial.getBias());
}

void test_zero_error_training_keeps_model_same() {
    Matrix X(2, 2, {
        1.0, 2.0,
        3.0, 4.0
    });

    LinearModel initial(Vector({1.0, 2.0}), 0.0);
    Vector targets = initial.predict(X);

    Trainer trainer(0.01, 10);
    TrainingResults result = trainer.train(initial, X, targets);

    const LinearModel& trained = result.model;

    assert(trained.getWeights()[0] == initial.getWeights()[0]);
    assert(trained.getWeights()[1] == initial.getWeights()[1]);
    assert(trained.getBias() == initial.getBias());
}

void test_training_records_loss_history() {
    Matrix X(2, 2, {
        1.0, 2.0,
        3.0, 4.0
    });

    Vector targets({8.0, 18.0});
    LinearModel initial(Vector({1.0, 2.0}), 0.0);

    Trainer trainer(0.01, 10);
    TrainingResults result = trainer.train(initial, X, targets);

    assert(result.trainingObjectiveLosses.size() == 10);
    assert(result.trainingObjectiveLosses.back() < result.trainingObjectiveLosses.front());
}

int main() {
    test_constructor_rejects_zero_learning_rate();
    test_constructor_rejects_negative_learning_rate();
    test_constructor_rejects_zero_epochs();

    test_training_reduces_mse();
    test_training_changes_weights_and_bias();
    test_zero_error_training_keeps_model_same();
    test_training_records_loss_history();

    std::cout << "All Trainer tests passed.\n";
    return 0;
}
