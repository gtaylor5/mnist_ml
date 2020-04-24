#include "network.hpp"
#include "layer.hpp"
#include "DataHandler.h"
#include <numeric>
#include <thread>


Network::Network(std::vector<int> spec, int inputSize, int numClasses)
{
    inputLayer = new InputLayer(0, inputSize);
    for(int i = 0; i < spec.size(); i++)
    {
        if(i == 0)
            layers.push_back(new Layer(inputSize, spec.at(i)));
        else
            layers.push_back(new Layer(layers.at(i-1)->neurons.size(), spec.at(i)));
        
    }
    layers.push_back(new Layer(layers.at(layers.size()-1)->neurons.size(), numClasses));
    this->eta = 0.1;
}

Network::~Network()
{

}

void Network::fprop(Data *data)
{
    std::vector<double> inputs = *data->getNormalizedFeatureVector();
    for(Layer *layer : layers)
    {
        std::vector<double> newInputs;
        for(Neuron *n : layer->neurons)
        {
            double activation = n->activate(inputs);
            n->output = n->transfer(activation);
            newInputs.push_back(n->output);
        }
        inputs = newInputs;
    }
}

void Network::bprop(Data *data)
{
    for(int i = layers.size() - 1; i >= 0; i--)
    {
        Layer *layer = layers.at(i);
        std::vector<double> errors;
        if(i == layers.size() - 1)
        {
            for(int j = 0; j < layer->neurons.size(); j++)
            {
                Neuron *n = layer->neurons.at(j);
                errors.push_back((double)(data->getClassVector().at(j)) - n->output);
            }
        } else {
            double error = 0.0;
            for(int j = 0; j < layer->neurons.size(); j++)
            {
                for(Neuron *n : layers.at(i+1)->neurons)
                {
                    error += (n->weights.at(j) * n->delta);
                }
                errors.push_back(error);
            }
        }
        for(int j = 0; j < layer->neurons.size(); j++)
        {
            Neuron *n = layer->neurons.at(j);
            n->delta = errors.at(j) * n->transferDerivative(n->output);
        }
    }
}

void Network::updateWeights(Data *data)
{
    for(int i = 0; i < layers.size(); i++)
    {
        std::vector<double> inputs;
        if(i != 0) 
        {
            for(Neuron *n : layers.at(i-1)->neurons)
            {
                inputs.push_back(n->output);
            }
        } else {
            inputs = *data->getNormalizedFeatureVector();
        }
        for(Neuron *n : layers.at(i)->neurons)
        {
            for(int j = 0; j < inputs.size(); j++)
            {
                n->weights[j] += (eta * n->delta * inputs.at(j));
            }
            n->bias += n->delta * eta;
        }
    }
}

void Network::train()
{
    for(Data *data : *this->trainingData)
    {
        if (data->getLabel() != target) continue;
        fprop(data);
        bprop(data);
        updateWeights(data);
    }
}

void Network::test()
{
    double numCorrect = 0.0;
    double count = 0.0;
    for(Data *data : *this->testData)
    {
        if(data->getLabel() != target) continue;
        count++;
        fprop(data);
        std::vector<double> outputs;
        for(Neuron *n : layers.back()->neurons){
            outputs.push_back(n->output);
        };
        int maxIndex = 0;
        double maxValue = outputs.at(0);
        for(int i = 1; i < outputs.size(); i++)
        {
            if(outputs.at(i) < maxValue)
            {
                maxValue = outputs.at(i);
                maxIndex = i;
            }
        }
        for(int i = 0; i < data ->getClassVector().size(); i++)
        {
            if(data->getClassVector().at(i) == 1)
            {
                if(i == maxIndex) numCorrect++;
                break;
            }
        }
    }

    testPerformance = (numCorrect / count) * 100;
    //fprintf(stderr, "Test Performance: %.4f\n", numCorrect / count);
}

void Network::validate()
{
   double numCorrect = 0.0;
   double count = 0.0;
    for(Data *data : *this->validationData)
    {
        if(data->getLabel() != target) continue;
        count++;
        fprop(data);
        std::vector<double> outputs;
        //printf("Output Layer: ");
        for(Neuron *n : layers.back()->neurons){
            //printf("%.4f ", n->output);
            outputs.push_back(n->output);
        }
        //printf("\n");
        int minIndex = 0;
        double minValue = outputs.at(0);
        for(int i = 1; i < outputs.size(); i++)
        {
            if(outputs.at(i) < minValue)
            {
                minValue = outputs.at(i);
                minIndex = i;
            }
        }
        
        for(int i = 0; i < data ->getClassVector().size(); i++)
        {
            if(data->getClassVector().at(i) == 1)
            {
                if(i == minIndex) numCorrect++;
                break;
            }
        }
    }
    //fprintf(stderr, "Validation Performance: %.4f\n", numCorrect / count);
}

int 
main(int argc, char const *argv[])
{
    DataHandler *dh = new DataHandler();
#ifdef MNIST
    dh->readInputData("../train-images-idx3-ubyte");
    dh->readLabelData("../train-labels-idx1-ubyte");
    dh->countClasses();
#else
    dh->readCsv("../iris.data", ",");
#endif
    dh->splitData();
    std::vector<int> hiddenLayers = {2};
    auto lamba = [&](int target) {
        Network *net = new Network(hiddenLayers, dh->getTrainingData()->at(0)->getNormalizedFeatureVector()->size(), dh->getClassCounts());
        net->target = target;
        net->setTrainingData(dh->getTrainingData());
        net->setTestData(dh->getTestData());
        net->setValidationData(dh->getValidationData());
        printf("Size of net %d: %zu\n", target, sizeof(*net));
        for(int i = 0; i < 100; i++)
        {
            net->train();
            if(i % 10 == 0)
                net->validate();
        }
        net->test();
        fprintf(stderr, "Test Performance for %d: %f -> Network Size: %zu", target, net->testPerformance, sizeof(*net));
    };
    
    std::vector<std::thread> threads;

    for(int i = 0; i < 10; i++) 
    {
        threads.emplace_back(std::thread(lamba, i));
    }
    
    for(auto &th : threads)
    {
        th.join();
    }

}
