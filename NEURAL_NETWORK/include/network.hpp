#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "data.h"
#include "neuron.hpp"
#include "layer.hpp"
#include "input_layer.hpp"
#include "common.hpp"

class Network : public CommonData
{
  public:
    std::vector<Layer *> layers;
    double eta;
    double testPerformance;
    Network(std::vector<int> hiddenLayerSpec, int, int, double);
    ~Network();
    std::vector<double> fprop(Data *data);
    double activate(std::vector<double>, std::vector<double>);
    double transfer(double);
    double transferDerivative(double);
    void bprop(Data *data);
    void updateWeights(Data *data);
    int predict(Data *data);
    void train(int);
    double test();
    void validate();
};

#endif
