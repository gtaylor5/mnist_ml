#include "network.hpp"
#include "layer.hpp"
#include "DataHandler.h"
#include <numeric>
#include <thread>


Network::Network(std::vector<int> spec, int inputSize, int numClasses, double learningRate)
{
    for(int i = 0; i < spec.size(); i++)
    {
        if(i == 0)
            layers.push_back(new Layer(inputSize, spec.at(i)));
        else
            layers.push_back(new Layer(layers.at(i-1)->neurons.size(), spec.at(i)));
        
    }
    layers.push_back(new Layer(layers.at(layers.size()-1)->neurons.size(), numClasses));
    this->eta = learningRate;
}

Network::~Network()
{

}

double Network::activate(std::vector<double> weights, std::vector<double> input)
{
    double activation = weights.back();
    for(int i = 0; i < weights.size() - 1; i++)
    {
        activation += weights[i] * input[i];
    }
    return activation;
}

double Network::transfer(double activation)
{
    return 1.0 / (1.0 + exp(-activation)); //sigmoid
}

double Network::transferDerivative(double output)
{
    return output * (1 - output); // signmoid derivative.
}

std::vector<double> Network::fprop(Data *data)
{
    std::vector<double> inputs = *data->getNormalizedFeatureVector();
    for(int i = 0; i < layers.size(); i++)
    {
        Layer *layer = layers.at(i);
        std::vector<double> newInputs;
        for(Neuron *n : layer->neurons)
        {
            double activation = this->activate(n->weights, inputs);
            n->output = this->transfer(activation);
            newInputs.push_back(n->output);
        }
        inputs = newInputs;
    }
    return inputs;
}

void Network::bprop(Data *data)
{
    for(int i = layers.size() - 1; i >= 0; i--)
    {
        Layer *layer = layers.at(i);
        std::vector<double> errors;
        if(i != layers.size() - 1)
        {
            for(int j = 0; j < layer->neurons.size(); j++)
            {
                double error = 0.0;
                for(Neuron *n : layers.at(i + 1)->neurons)
                {
                    error += (n->weights.at(j) * n->delta);
                }
                errors.push_back(error);
            }
        } else {
            for(int j = 0; j < layer->neurons.size(); j++)
            {
                Neuron *n = layer->neurons.at(j);
                errors.push_back((double) data->getClassVector().at(j) - n->output);
            }
        }
        for(int j = 0; j < layer->neurons.size(); j++)
        {
            Neuron *n = layer->neurons.at(j);
            n->delta = errors.at(j) * this->transferDerivative(n->output);
        }
    }
}

void printWeights(Layer *layer, std::string prefix) {
    int counter = 1;
    std::for_each(layer->neurons.begin(), layer->neurons.end(), [&](Neuron *n) {
        printf("%s Neuron %d: ", prefix.c_str(), counter++);
        std::for_each(n->weights.begin(), n->weights.end(), [](double value){
            printf("%.3f ", value);
        });
        printf("\n");
    });
}

void Network::updateWeights(Data *data)
{
    std::vector<double> inputs = *data->getNormalizedFeatureVector();
    for(int i = 0; i < layers.size(); i++)
    {
        if(i != 0)
        {
            for(Neuron *n : layers.at(i - 1)->neurons)
            {
                inputs.push_back(n->output);
            }
        }
        for(Neuron *n : layers.at(i)->neurons)
        {
            for(int j = 0; j < inputs.size(); j++)
            {
                n->weights.at(j) += this->eta * n->delta * inputs.at(j);
            }
            n->weights.back() += this->eta * n->delta; // udpate bias.
        }
        inputs.clear();
    }
}

int Network::predict(Data *data)
{
    std::vector<double> outputs = fprop(data);
    return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}

void Network::train(int numEpochs)
{
    for(int i = 0; i < numEpochs; i++)
    {
        double sumError = 0.0;
        for(Data *data : *this->trainingData)
        {
            std::vector<double> outputs = fprop(data);
            std::vector<int> expected = data->getClassVector();
            double tempErrorSum = 0.0;
            for(int j = 0; j < outputs.size(); j++)
            {
                tempErrorSum += pow((double) expected.at(j) - outputs.at(j), 2);
            }
            sumError += tempErrorSum;
            bprop(data);
            updateWeights(data);
        }
        printf("EPOCH=%d\terror=%.3f\n", i, sumError);
    }
}

double Network::test()
{
    double numCorrect = 0.0;
    double count = 0.0;
    for(Data *data : *this->testData)
    {
        count++;
        int index = predict(data);
        if(data->getClassVector().at(index) == 1) numCorrect++;
    }
    testPerformance = (numCorrect / count);
    return testPerformance;
}

void Network::validate()
{
   double numCorrect = 0.0;
   double count = 0.0;
    for(Data *data : *this->validationData)
    {
        count++;
        int predictedIndex = this->predict(data);
        if(data->getClassVector().at(predictedIndex) == 1) numCorrect++;
    }
    fprintf(stderr, "Validation Performance: %.4f\n", numCorrect / count);
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
    std::vector<int> hiddenLayers = {10, 15, 20};
    auto lambda = [&]() {
        Network *net = new Network(hiddenLayers, dh->getTrainingData()->at(0)->getNormalizedFeatureVector()->size(), dh->getClassCounts(), 0.25);
        net->setTrainingData(dh->getTrainingData());
        net->setTestData(dh->getTestData());
        net->setValidationData(dh->getValidationData());
        net->train(100);
        net->validate();
        printf("Test Performance: %.3f\n", net->test());
    };

    lambda();

}
