#include "output_layer.hpp"
#include "data.h"

void OutputLayer::feedForward(Layer prev)
{
  for(int i = 0; i < this->neurons.size(); i++)
  {
    this->neurons.at(i)->calculatePreActivation(prev.getLayerOutputs());
    layerOutputs[i] = this->neurons.at(i)->activate();
    this->neurons.at(i)->calculateOutputDerivative();
  }
}

void OutputLayer::backProp(data* data)
{
 // printf("Error: [");
  for(int i = 0; i < neurons.size(); i++)
  {
    Neuron *n = neurons.at(i);

    double error = (n->getOutput() - (double) data->getClassVector().at(i)) * n->getOutputDerivative();
    n->setError(error);
   // printf("%f (%f, %f), ", n->getError(), n->getOutput(), (double)data->getClassVector().at(i));
  }
  //printf("]\n");
}

void OutputLayer::updateWeights(double eta, Layer *prev)
{
  for(int i = 0; i < this->neurons.size(); i++)
  {
 //   this->neurons.at(i)->printWeights();
    for(int j = 0; j < prev->neurons.size(); j++)
    {
      double delta = -eta * prev->neurons.at(j)->getOutput() * neurons.at(i)->getError();
      neurons.at(i)->setWeight(delta, j);
    }
   // this->neurons.at(i)->printWeights();
    neurons.at(i)->setBias(neurons.at(i)->getBias());
  }
}

void OutputLayer::print()
{
  int i;
  printf("[ ");
  for(i = 0; i < neurons.size() - 1; i++)
    printf("%f, ", neurons.at(i)->getOutput());
  printf("%f]\n", neurons.at(i)->getOutput());
}
