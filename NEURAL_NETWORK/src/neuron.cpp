#include "neuron.hpp"
#include <random>

double generateRandomNumder(double min, double max)
{
    double random = (double) rand() / RAND_MAX;
    return min + random * (max - min);
}

Neuron::Neuron(int previousLayerSize, int currentLayerSize)
{
    initializeWeights(previousLayerSize, currentLayerSize);
}

void Neuron::initializeWeights(int previousLayerSize, int currentLayerSize)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    for(int i = 0; i < previousLayerSize; i++)
    {
        weights.push_back(generateRandomNumder(-0.5, 0.5));
    }
}

double Neuron::activate(std::vector<double> input)
{
  this->activation = this->bias;
  for(int i = 0; i < input.size(); i++)
  {
    this->activation += weights.at(i) * input.at(i);
  }
  return this->activation;
}

double Neuron::transfer(double activation)
{
  return (1.0 / (1 + exp(activation)));
}

double Neuron::transferDerivative(double output)
{
  return output * (1 - output);
}
