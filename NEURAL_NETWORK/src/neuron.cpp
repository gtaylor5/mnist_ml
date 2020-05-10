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
    for(int i = 0; i < previousLayerSize + 1; i++) // +1 to include bias
    {
      weights.push_back(generateRandomNumder(-1.0, 1.0));
    }
}
