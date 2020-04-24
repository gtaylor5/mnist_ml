#include "../include/data.h"

void Data::setDistance(double dist)
{
  distance = dist;
}
void Data::setFeatureVector(std::vector<uint8_t>* vect)
{
  featureVector = vect;
}


void Data::setNormalizedFeatureVector(std::vector<double>* vect)
{
  normalizedFeatureVector = vect;
}
void Data::appendToFeatureVector(uint8_t val)
{
  featureVector->push_back(val);
}
void Data::appendToFeatureVector(double val)
{
  normalizedFeatureVector->push_back(val);
}
void Data::setLabel(uint8_t val)
{
  label = val;
}
void Data::setEnumeratedLabel(uint8_t val)
{
  enumeratedLabel = val;
}

void Data::setClassVector(int classCounts)
{
  classVector = new std::vector<int>();
  for(int i = 0; i < classCounts; i++)
  {
    if(i == label)
      classVector->push_back(1);
    else
      classVector->push_back(0);
  }
}

void Data::printVector()
{
  printf("[ ");
  for(uint8_t val : *featureVector)
  {
    printf("%u ", val);
  }
  printf("]\n");
}

void Data::printNormalizedVector()
{
  printf("[ ");
  for(auto val : *normalizedFeatureVector)
  {
    printf("%.2f ", val);
  }
  printf("]\n");
  
}

double Data::getDistance()
{
  return distance;
}

int Data::getFeatureVectorSize()
{
  return featureVector->size();
}
uint8_t Data::getLabel()
{
  return label;
}
uint8_t Data::getEnumeratedLabel()
{
  return enumeratedLabel;
}

std::vector<uint8_t> * Data::getFeatureVector()
{
  return featureVector;
}
std::vector<double> * Data::getNormalizedFeatureVector()
{
  return normalizedFeatureVector;
}

std::vector<int>  Data::getClassVector()
{
  return *classVector;
}
