#pragma once

#include "ml/DataSetLoader.h"

namespace machineLearning {

struct DatasetSplit {
    Dataset train;
    Dataset test;
};

class DataSplitter {
public:
    static DatasetSplit split(const Dataset& data, double trainRatio);
};

}