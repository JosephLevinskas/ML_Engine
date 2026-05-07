#pragma once 

#include "ml/core/Vector.h"
#include "ml/core/Matrix.h"

namespace machineLearning {

double dot(const Vector& v1, const Vector& v2);

Vector operator*(const Matrix&, const Vector&);


}

