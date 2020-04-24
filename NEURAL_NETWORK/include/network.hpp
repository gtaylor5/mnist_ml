#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "data.h"
#include "neuron.hpp"
#include "layer.hpp"
#include "input_layer.hpp"
#include "common.hpp"

class Network : public CommonData
{
  private:
    InputLayer *inputLayer;
    std::vector<Layer *> layers;
    double eta;
    int iteration = 0;
  public:
    int target;
    double testPerformance;
    Network(std::vector<int> hiddenLayerSpec, int, int);
    ~Network();
    void fprop(Data *data);
    void bprop(Data *data);
    void updateWeights(Data *data);
    void threadedTrain(std::vector<Data *>, int start, int end);
    void train();
    void test();
    void validate();
};

#endif
