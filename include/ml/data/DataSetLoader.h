#pragma once

#include "ml/core/Matrix.h"
#include "ml/core/Vector.h"

#include <string>
#include <vector>


namespace machineLearning {

struct Dataset {
    Matrix features;
    Vector targets;
};

class DataSetLoader {
public:
    static std::vector<double> parseCSVLine(const std::string& line);
    static Dataset loadCSV(const std::string& path, size_t targetColumn);
};
}
