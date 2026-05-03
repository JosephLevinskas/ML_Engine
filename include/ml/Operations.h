#pragma once 

#include "ml/Vector.h"
#include "ml/Matrix.h"

namespace machineLearning {

double dot(const Vector& v1, const Vector& v2);

Vector operator*(const Matrix&, const Vector&);

}
