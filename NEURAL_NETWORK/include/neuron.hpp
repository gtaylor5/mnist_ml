#ifndef _NEURON_HPP_
#define _NEURON_HPP_
#include <stdio.h>
#include <vector>
#include <cmath>

class Neuron {
  public:
    double output;
    double delta;
    std::vector<double> weights;
    Neuron(int, int);
    void initializeWeights(int);
};

#endif
