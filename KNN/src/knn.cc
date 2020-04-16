#include "knn.h"
#include <cmath>
#include <limits>
#include <map>
#include "stdint.h"
#include "DataHandler.h"


KNN::KNN(int val)
{
  k = val;
}

KNN::KNN()
{

}

KNN::~KNN()
{
  // NOTHING TO DO 
}

void KNN::findKnearest(Data *queryPoint)
{
  neighbors = new std::vector<Data *>;
  double min = std::numeric_limits<double>::max();
  double previousMin = min;
  int index;
  for(int i = 0; i < k; i++)
  {
    if(i == 0)
    {
      for(int j = 0; j < trainingData->size(); j++)
      {
        double dist = calculateDistance(queryPoint, trainingData->at(j));
        trainingData->at(j)->setDistance(dist);
        if(dist < min)
        {
          min = dist;
          index = j;
        }
      }
      neighbors->push_back(trainingData->at(index));
      previousMin = min;
      min = std::numeric_limits<double>::max();
    } else 
    {
      for(int j = 0; j < trainingData->size(); j++)
      {
        double dist = trainingData->at(j)->getDistance();
        if(dist > previousMin && dist < min)
        {
          min = dist;
          index = j;
        }
      }
      neighbors->push_back(trainingData->at(index));
      previousMin = min;
      min = std::numeric_limits<double>::max();
    }
  }
}
void KNN::setK(int val)
{
  k = val;
}

int KNN::findMostFrequentClass()
{
  std::map<uint8_t, int> frequencyMap;
  for(int i = 0; i < neighbors->size(); i++)
  {
    if(frequencyMap.find(neighbors->at(i)->getLabel()) == frequencyMap.end())
    {
      frequencyMap[neighbors->at(i)->getLabel()] = 1;
    } else 
    {
      frequencyMap[neighbors->at(i)->getLabel()]++;
    }
  }

  int best = 0;
  int max = 0;

  for(auto kv : frequencyMap)
  {
    if(kv.second > max)
    {
      max = kv.second;
      best = kv.first;
    }
  }
  delete neighbors;
  return best;

}

double KNN::calculateDistance(Data* queryPoint, Data* input)
{
  double value = 0;
  if(queryPoint->getNormalizedFeatureVector()->size() != input->getNormalizedFeatureVector()->size())
  {
    printf("Vector size mismatch.\n");
    exit(1);
  }
#ifdef EUCLID
  for(unsigned i = 0; i < queryPoint->getNormalizedFeatureVector()->size(); i++)
  {
    value += pow(queryPoint->getNormalizedFeatureVector()->at(i) - input->getNormalizedFeatureVector()->at(i),2);
  }
  return sqrt(value);
#elif defined MANHATTAN
  //do some stuff
#endif
}

double KNN::validatePerformance()
{
  double current_performance = 0;
  int count = 0;
  int data_index = 0;
  for(Data *queryPoint : *validationData)
  {
    findKnearest(queryPoint);
    int prediction = findMostFrequentClass();
    data_index++;
    if(prediction == queryPoint->getLabel())
    {
      count++;
    }
    printf("Current Performance: %.3f %%\n", ((double) count)*100.0 / ((double) data_index));
  }
  current_performance = ((double) count)*100.0/((double) validationData->size());
  printf("Validation Performance for K = %d: %.3f\n", k, current_performance);
  return current_performance;
}
double KNN::testPerformance()
{
  double current_performance = 0;
  int count = 0;
  for(Data *queryPoint : *testData)
  {
    findKnearest(queryPoint);
    int prediction = findMostFrequentClass();
    if(prediction == queryPoint->getLabel())
    {
      count++;
    }
  }
  current_performance = ((double) count)*100.0/((double) testData->size());
  printf("Validation Performance for K = %d: %.3f\n", k, current_performance);
  return current_performance;
}

int
main()
{
  DataHandler *dh = new DataHandler();
  //dh->read_csv("/home/gerardta/iris.data",",");
  dh->readInputData("../train-images-idx3-ubyte");
  dh->readLabelData("../train-labels-idx1-ubyte");
  dh->countClasses();
  dh->splitData();
  KNN *nearest = new KNN();
  nearest->setK(3);
  nearest->setTrainingData(dh->getTrainingData());
  nearest->setTestData(dh->getTestData());
  nearest->setValidationData(dh->getValidationData());
  double performance = 0;
  double best_performance = 0;
  int best_k = 1;
  for(int k = 1; k <= 3; k++)
  {
    if(k == 1)
    {
      performance = nearest->validatePerformance();
      best_performance = performance;
    } else 
    {
      nearest->setK(k);
      performance = nearest->validatePerformance();
      if(performance > best_performance)
      {
        best_performance = performance;
        best_k = k;
      }
    }
  }
  nearest->setK(best_k);
  nearest->testPerformance();
}
