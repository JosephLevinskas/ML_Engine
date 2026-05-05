#include "ml/LinearRegressionPipeline.h"

#include <vector>
#include <stdexcept>

namespace machineLearning {

LinearRegressionPipeline::LinearRegressionPipeline(const LinearModel& initialModel) 
    : model(initialModel) {}


TrainingResults LinearRegressionPipeline::train(const Matrix& X, const Vector& y, Trainer& trainer) {
    
    Matrix scaledX = scaler.fitTransform(X);

    TrainingResults result = trainer.train(model, scaledX, y);

    model = result.model;
    trained = true;

    return result;
}

Vector LinearRegressionPipeline::predict(const Matrix& X) const {
    if (!trained) {
        throw std::invalid_argument("Matrix must be trained before it can predict");
    }

    Matrix fittedM = scaler.transform(X);
    return model.predict(fittedM);

}

}