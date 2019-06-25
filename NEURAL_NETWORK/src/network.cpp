#include "network.hpp"
#include "layer.hpp"
#include "data_handler.h"
#include <time.h>
Network::Network(std::vector<int> hiddenLayerSpec, int inputSize, int numClasses)
{
  inputLayer = new InputLayer(0, inputSize);
  for(int i = 0; i < hiddenLayerSpec.size(); ++i)
  {
    if(i == 0)
      hiddenLayers.push_back(new HiddenLayer(inputLayer->getSize(), hiddenLayerSpec.at(i)));
    else
      hiddenLayers.push_back(new HiddenLayer(hiddenLayers.at(i-1)->getSize(), hiddenLayerSpec.at(i)));
  }
  outputLayer = new OutputLayer(hiddenLayers.at(hiddenLayers.size() - 1)->getSize(), numClasses);
  this->eta = 0.01;
}

Network::~Network()
{
  delete inputLayer;
  delete outputLayer;
  for(HiddenLayer * h : hiddenLayers)
    delete h;
}

void Network::fprop(data *data)
{
  inputLayer->setLayerOutputs(data);
  for(int i = 0; i < hiddenLayers.size(); i++)
  {
    if(i == 0)
      hiddenLayers.at(i)->feedForward(*inputLayer);
    else
      hiddenLayers.at(i)->feedForward(*hiddenLayers.at(i-1));
  }
  outputLayer->feedForward(*hiddenLayers.at(hiddenLayers.size()-1));
  //outputLayer->print();
}

void Network::bprop(data *data)
{
  outputLayer->backProp(data);
  for(int i = hiddenLayers.size()-1; i >= 0; i--)
  {
    if(i == hiddenLayers.size()-1)
      hiddenLayers.at(i)->backProp(*outputLayer);
    else
      hiddenLayers.at(i)->backProp(*hiddenLayers.at(i+1));
  }
}

void Network::updateWeights()
{
  for(int i = 0; i < hiddenLayers.size(); i++)
  {
    if(i == 0)
      hiddenLayers.at(i)->updateWeights(eta, inputLayer);
    else
      hiddenLayers.at(i)->updateWeights(eta, hiddenLayers.at(i-1));
  }
  outputLayer->updateWeights(eta, hiddenLayers.at(hiddenLayers.size()-1));
}

void Network::train()
{
  double numCorrect = 0.0;
  for(data *data : *this->training_data)
  {
    fprop(data);
    bprop(data);
    updateWeights();
    std::vector<double> outputs = outputLayer->getLayerOutputs();
    double maxValue = outputs.at(0);
    int maxIndex = 0;
    for(int i = 1; i < outputs.size(); i++)
    {
      if(maxValue < outputs.at(i))
      {
        maxIndex = i;
        maxValue = outputs.at(i);
      }
    }
    for(int i = 0; i < data->getClassVector().size(); i++)
    {
      if(data->getClassVector().at(i) == 1)
      {
        if(i == maxIndex) numCorrect++;
        break;
      }
    }
  }
  fprintf(stderr, "Current Performance: %.4f\n", numCorrect / (double)this->training_data->size());
}

void Network::test()
{
  double numCorrect = 0.0;
  for(data *data : *this->test_data)
  {
    fprop(data);
    std::vector<double> outputs = outputLayer->getLayerOutputs();
    double maxValue = outputs.at(0);
    int maxIndex = 0;
    for(int i = 1; i < outputs.size(); i++)
    {
      if(maxValue < outputs.at(i))
      {
        maxIndex = i;
        maxValue = outputs.at(i);
      }
    }
    for(int i = 0; i < data->getClassVector().size(); i++)
    {
      if(data->getClassVector().at(i) == 1)
      {
        if(i == maxIndex) numCorrect++;
        break;
      }
    }
  }
  fprintf(stderr, "Current Performance: %.4f\n", numCorrect / (double)this->training_data->size());
}

void Network::validate()
{
  
}


int main()
{
  data_handler *dh = new data_handler();
#ifdef MNIST
  dh->read_input_data("../train-images-idx3-ubyte");
  dh->read_label_data("../train-labels-idx1-ubyte");
  dh->normalize();
#else
  dh->read_csv("/home/gerardta/iris.data", ",");
#endif
  dh->count_classes();
  dh->split_data();
  std::vector<int> hiddenLayers = {300};
  Network *net = new Network(hiddenLayers, dh->get_test_data()->at(0)->get_normalized_feature_vector()->size(), dh->get_class_counts());
  net->set_training_data(dh->get_training_data());
  net->set_test_data(dh->get_test_data());
  net->set_validation_data(dh->get_validation_data());
  for(int i = 0; i < 10000; i++)
    net->train();
  net->test();
}
