#include "../include/common.hpp"

void CommonData::setTrainingData(std::vector<Data *> * vect)
{
 trainingData = vect;
}
void CommonData::setTestData(std::vector<Data *> * vect)
{
 testData = vect;
}
void CommonData::setValidationData(std::vector<Data *> * vect)
{
  validationData = vect;
}
