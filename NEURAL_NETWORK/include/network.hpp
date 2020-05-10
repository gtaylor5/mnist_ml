#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "data.h"
#include "neuron.hpp"
#include "layer.hpp"
#include "common.hpp"

class Network : public CommonData
{
  public:
    std::vector<Layer *> layers;
    double learningRate;
    double testPerformance;
    Network(std::vector<int> spec, int, int, double);
    ~Network();
    std::vector<double> fprop(Data *data);
    double activate(std::vector<double>, std::vector<double>); // dot product
    double transfer(double);
    double transferDerivative(double); // used for backprop
    void bprop(Data *data);
    void updateWeights(Data *data);
    int predict(Data *data); // return the index of the maximum value in the output array.
    void train(int); // num iterations
    double test();
    void validate();
};

#endif
