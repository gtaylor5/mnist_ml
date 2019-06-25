#include "layer.hpp"


Layer::Layer(int previousLayerSize, int currentLayerSize)
{
  for(int i = 0; i < currentLayerSize; i++)
  {
    neurons.push_back(new Neuron(previousLayerSize, currentLayerSize));
    this->layerOutputs.push_back(0.0);
  }
  this->currentLayerSize = currentLayerSize;
}

Layer::~Layer()
{
}

std::vector<double> Layer::getLayerOutputs()
{
  return this->layerOutputs;
}

int Layer::getSize()
{
  return this->currentLayerSize;
}

