#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "data.h"
#include "neuron.hpp"
#include "layer.hpp"
#include "hidden_layer.hpp"
#include "input_layer.hpp"
#include "output_layer.hpp"
#include "common.hpp"

class Network : public common_data
{
  private:
    InputLayer *inputLayer;
    OutputLayer *outputLayer;
    std::vector<HiddenLayer *> hiddenLayers;
    double eta;
  public:
    Network(std::vector<int> hiddenLayerSpec, int, int);
    ~Network();
    void fprop(data *data);
    void bprop(data *data);
    void updateWeights();
    void train();
    void test();
    void validate();
};

#endif
