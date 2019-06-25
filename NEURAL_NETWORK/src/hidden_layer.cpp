#include "hidden_layer.hpp"


void HiddenLayer::feedForward(Layer prev)
{
  for(int i = 0; i < this->neurons.size(); i++)
  {
    this->neurons.at(i)->calculatePreActivation(prev.getLayerOutputs());
    layerOutputs[i] = this->neurons.at(i)->activate();
    this->neurons.at(i)->calculateOutputDerivative();
  }
}

void HiddenLayer::backProp(Layer next)
{
  for(int i = 0; i < this->neurons.size(); i++)
  {
    double sum = 0;
    for(int j = 0; j < next.neurons.size(); j++)
    {
      sum += next.neurons.at(j)->getWeights().at(i) * next.neurons.at(j)->getError();
    }
    this->neurons.at(i)->setError(sum * this->neurons.at(i)->getOutputDerivative());
  }
}

void HiddenLayer::updateWeights(double eta, Layer *prev)
{
  for(int i = 0; i < this->neurons.size(); i++)
  {
    for(int j = 0; j < prev->neurons.size(); j++)
    {
      double delta = -eta * prev->neurons.at(j)->getOutput() * neurons.at(i)->getError();
      neurons.at(i)->setWeight(delta, j);
    }
    neurons.at(i)->setBias(neurons.at(i)->getBias());
  }
}
