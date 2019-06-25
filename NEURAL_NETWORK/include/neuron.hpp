#ifndef _NEURON_HPP_
#define _NEURON_HPP_
#include <stdio.h>
#include <vector>
#include <cmath>

class Neuron {
  private:
    std::vector<double> weights;
    double bias;
    double preActivation;
    double activatedOutput;
    double outputDerivative;
    double error;
    double alpha;
  public:
    Neuron(int, int);
    ~Neuron();
    void initializeWeights(int previousLayerSize, int currentLayerSize);
    void setError(double signalError);
    void setBias(double value);
    void printWeights();
    void setWeight(double, int);
    double calculatePreActivation(std::vector<double>);
    double activate();
    double calculateOutputDerivative();
    double sigmoid();
    double relu();
    double leakyRelu();
    double inverseSqrtLU();
    double getOutput();
    double getOutputDerivative();
    double getError();
    double getBias();
    std::vector<double> getWeights();
};

#endif
