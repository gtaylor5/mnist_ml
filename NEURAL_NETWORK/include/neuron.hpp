#ifndef _NEURON_HPP_
#define _NEURON_HPP_
#include <stdio.h>
#include <vector>
#include <cmath>

class Neuron {
  public:
    std::vector<double> weights;
    double bias;
    double delta;
    double output;
    double activation;
    Neuron(int, int);

    void initializeWeights(int, int);
    double activate(std::vector<double>);
    double transfer(double value);
    double transferDerivative(double value);

};

#endif
