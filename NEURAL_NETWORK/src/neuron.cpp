#include "neuron.hpp"
#include <random>


double generateRandomNumber(double min, double max)
{
  double random = (double) rand() / RAND_MAX;
  return min + random * (max - min);
}

Neuron::Neuron(int previousLayerSize, int currentLayerSize)
{
  initializeWeights(previousLayerSize, currentLayerSize);
  this->alpha = 0.5;
}

void Neuron::initializeWeights(int previousLayerSize, int currentLayerSize)
{
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0,1.0);
  for(int i = 0; i < previousLayerSize; i++)
  {
    weights.push_back(generateRandomNumber(-0.5, 0.5) * sqrt(2.0/previousLayerSize));
  }
#ifdef DEBUG
  fprintf(stderr, "Weights have been initialized\n");
#endif
}

void Neuron::setError(double signalError)
{
  this->error = signalError;
}

void Neuron::setBias(double bias)
{
  this->bias = bias;
}

void Neuron::setWeight(double delta, int index)
{
  weights[index] = weights[index] + delta;
}

double Neuron::calculatePreActivation(std::vector<double> previousLayerOutputs)
{
  if(previousLayerOutputs.size() != this->weights.size())
  {
    fprintf(stderr, "Error layer outputs and neuron weights size mismatch\n");
    exit(EXIT_FAILURE);
  }
  double linearSum = 0.0;
  for(int i = 0; i < previousLayerOutputs.size(); i++)
  {
    linearSum += (previousLayerOutputs.at(i) * weights.at(i));
  }
  this->preActivation = linearSum;
#ifdef DEBUG
  fprintf(stderr, "Pre Activation Calculated\n");
#endif
  return linearSum;
}

double Neuron::activate()
{
#ifdef SIGMOID
  this->activatedOutput = sigmoid();
#elif defined RELU
  this->activatedOutput = relu();
#elif defined LEAKY_RELU
  this->activatedOutput = leakyRelu();
#else
  this->activatedOutput = inverseSqrtLU();
#endif
#ifdef DEBUG
  fprintf(stderr, "Activation Calculated\n");
#endif
  return this->activatedOutput;
}

double Neuron::calculateOutputDerivative()
{
#ifdef SIGMOID
  this->outputDerivative = this->activatedOutput * (1 - this->activatedOutput);
#elif defined RELU
  this->outputDerivative = this->activatedOutput <= 0 ? 0 : 1;
#elif defined LEAKY_RELU
  this->outputDerivative = this->activatedOutput < 0 ? 0.25 : 1;
#else
  this->outputDerivative = this->activatedOutput >= 0 ? 1 : 
    pow(1 / sqrt(1 + this->alpha * pow(this->activatedOutput,2)),3);
#endif
#ifdef DEBUG
  fprintf(stderr, "Output Derivative Calculated\n");
#endif
  return this->outputDerivative;
}

double Neuron::sigmoid()
{
  return 1 / (1 + exp(-this->preActivation));
}

double Neuron::relu()
{
  return (preActivation <= 0) ? 0 : preActivation;
}

double Neuron::leakyRelu()
{
  return (preActivation < 0) ? 0.25 * preActivation : preActivation;
}

double Neuron::inverseSqrtLU()
{

  return (preActivation <= 0) ? preActivation / sqrt(1 + this->alpha * pow(this->preActivation, 2)) : preActivation;
}

double Neuron::getOutput()
{
  return this->activatedOutput;
}

double Neuron::getOutputDerivative()
{
  return this->outputDerivative;
}

double Neuron::getError()
{
  return this->error;
}

double Neuron::getBias()
{
  return this->bias;
}

std::vector<double> Neuron::getWeights()
{
  return this->weights;
}

void Neuron::printWeights()
{
  printf("Weights: [");
  for(auto val : weights)
    printf("%f, ", val);
  printf("]\n");
} 

