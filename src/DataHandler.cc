#include "../include/DataHandler.h"
#include <algorithm>
#include <random>


DataHandler::DataHandler()
{
  dataArray = new std::vector<Data *>;
  trainingData = new std::vector<Data *>;
  testData = new std::vector<Data *>;
  validationData = new std::vector<Data *>;
}

DataHandler::~DataHandler()
{
  // FIX ME
}

void DataHandler::readCsv(std::string path, std::string delimiter)
{
  class_counts = 0;
  std::ifstream data_file;
  data_file.open(path.c_str());
  std::string line;

  while(std::getline(data_file, line))
  {
    if(line.length() == 0) continue;
    Data *d = new Data();
    d->setNormalizedFeatureVector(new std::vector<double>());
    size_t position = 0;
    std::string token;
    while((position = line.find(delimiter)) != std::string::npos)
    {
      token = line.substr(0, position);
      d->appendToFeatureVector(std::stod(token));
      line.erase(0, position + delimiter.length());
    }

    if(classFromString.find(line) != classFromString.end())
    {
      d->setLabel(classFromString[line]);
    } else 
    {
      classFromString[line] = class_counts;
      d->setLabel(classFromString[token]);
      class_counts++;
    }
    dataArray->push_back(d);
  }
  for(Data *data : *dataArray)
    data->setClassVector(class_counts);;
  //normalize();
  featureVectorSize = dataArray->at(0)->getNormalizedFeatureVector()->size();
}

void DataHandler::readInputData(std::string path)
{
  uint32_t magic = 0;
  uint32_t num_images = 0;
  uint32_t num_rows = 0;
  uint32_t num_cols = 0;

  unsigned char bytes[4];
  FILE *f = fopen(path.c_str(), "r");
  if(f)
  {
    int i = 0;
    while(i < 4)
    {
      if(fread(bytes, sizeof(bytes), 1, f))
      {
        switch(i)
        {
          case 0:
            magic = format(bytes);
            i++;
            break;
          case 1:
            num_images = format(bytes);
            i++;
            break;
          case 2:
            num_rows = format(bytes);
            i++;
            break;
          case 3:
            num_cols = format(bytes);
            i++;
            break;
        }
      }
    }
    printf("Done getting file header.\n");
    uint32_t image_size = num_rows * num_cols;
    for(i = 0; i < num_images; i++)
    {
      Data *d = new Data();
      d->setFeatureVector(new std::vector<uint8_t>());
      uint8_t element[1];
      for(int j = 0; j < image_size; j++)
      {
        if(fread(element, sizeof(element), 1, f))
        {
          d->appendToFeatureVector(element[0]);
        }
      }
      dataArray->push_back(d);
      dataArray->back()->setClassVector(class_counts);
    }
    normalize();
    featureVectorSize = dataArray->at(0)->getFeatureVector()->size();
    printf("Successfully read %lu data entries.\n", dataArray->size());
    printf("The Feature Vector Size is: %d\n", featureVectorSize);
  } else
  {
    printf("Invalid Input File Path\n");
    exit(1);
  }
}
void DataHandler::readLabelData(std::string path)
{
  uint32_t magic = 0;
  uint32_t num_images = 0;
  unsigned char bytes[4];
  FILE *f = fopen(path.c_str(), "r");
  if(f)
  {
    int i = 0;
    while(i < 2)
    {
      if(fread(bytes, sizeof(bytes), 1, f))
      {
        switch(i)
        {
          case 0:
            magic = format(bytes);
            i++;
            break;
          case 1:
            num_images = format(bytes);
            i++;
            break;
        }
      }
    }

    for(unsigned j = 0; j < num_images; j++)
    {
      uint8_t element[1];
      if(fread(element, sizeof(element), 1, f))
      {
        dataArray->at(j)->setLabel(element[0]);
      }
    }

    printf("Done getting Label header.\n");
  } 
  else
  {
    printf("Invalid Label File Path\n");
    exit(1);
  }
}
void DataHandler::splitData()
{
  std::unordered_set<int> used_indexes;
  int train_size = dataArray->size() * TRAIN_SET_PERCENT;
  int test_size = dataArray->size() * TEST_SET_PERCENT;
  int valid_size = dataArray->size() * VALID_SET_PERCENT;
  
  std::random_shuffle(dataArray->begin(), dataArray->end());

  // Training Data

  int count = 0;
  int index = 0;
  while(count < train_size)
  {
    trainingData->push_back(dataArray->at(index++));
    count++;
  }

  // Test Data
  count = 0;
  while(count < test_size)
  {
    testData->push_back(dataArray->at(index++));
    count++;
  }

  // Test Data

  count = 0;
  while(count < valid_size)
  {
    validationData->push_back(dataArray->at(index++));
    count++;
  }

  printf("Training Data Size: %lu.\n", trainingData->size());
  printf("Test Data Size: %lu.\n", testData->size());
  printf("Validation Data Size: %lu.\n", validationData->size());
}

void DataHandler::countClasses()
{
  int count = 0;
  for(unsigned i = 0; i < dataArray->size(); i++)
  {
    if(classFromInt.find(dataArray->at(i)->getLabel()) == classFromInt.end())
    {
      classFromInt[dataArray->at(i)->getLabel()] = count;
      dataArray->at(i)->setEnumeratedLabel(count);
      count++;
    }
    else 
    {
      dataArray->at(i)->setEnumeratedLabel(classFromInt[dataArray->at(i)->getLabel()]);
    }
  }
  
  class_counts = count;
  for(Data *data : *dataArray)
    data->setClassVector(class_counts);
  printf("Successfully Extraced %d Unique Classes.\n", class_counts);
}

void DataHandler::normalize()
{
  std::vector<double> mins, maxs;
  // fill min and max lists
  
  Data *d = dataArray->at(0);
  for(auto val : *d->getFeatureVector())
  {
    mins.push_back(val);
    maxs.push_back(val);
  }

  for(int i = 1; i < dataArray->size(); i++)
  {
    d = dataArray->at(i);
    for(int j = 0; j < d->getFeatureVectorSize(); j++)
    {
      double value = (double) d->getFeatureVector()->at(j);
      if(value < mins.at(j)) mins[j] = value;
      if(value > maxs.at(j)) maxs[j] = value;
    }
  }
  // normalize data array
  
  for(int i = 0; i < dataArray->size(); i++)
  {
    dataArray->at(i)->setNormalizedFeatureVector(new std::vector<double>());
    dataArray->at(i)->setClassVector(class_counts);
    for(int j = 0; j < dataArray->at(i)->getFeatureVectorSize(); j++)
    {
      if(maxs[j] - mins[j] == 0) dataArray->at(i)->appendToFeatureVector(0.0);
      else
        dataArray->at(i)->appendToFeatureVector(
          (double)(dataArray->at(i)->getFeatureVector()->at(j) - mins[j])/(maxs[j]-mins[j]));
    }
  }
}

int DataHandler::getClassCounts()
{
  return class_counts;
}

int DataHandler::getDataArraySize()
{
  return dataArray->size();
}
int DataHandler::getTrainingDataSize()
{
  return trainingData->size();
}
int DataHandler::getTestDataSize()
{
  return testData->size();
}
int DataHandler::getValidationSize()
{
  return validationData->size();
}

uint32_t DataHandler::format(const unsigned char* bytes)
{
  return (uint32_t)((bytes[0] << 24) |
                    (bytes[1] << 16)  |
                    (bytes[2] << 8)   |
                    (bytes[3]));
}

std::vector<Data *> * DataHandler::getTrainingData()
{
  return trainingData;
}
std::vector<Data *> * DataHandler::getTestData()
{
  return testData;
}
std::vector<Data *> * DataHandler::getValidationData()
{
  return validationData;
}

std::map<uint8_t, int> DataHandler::getClassMap()
{
  return classFromInt;
}

void DataHandler::print()
{
  printf("Training Data:\n");
  for(auto data : *trainingData)
  {
    for(auto value : *data->getNormalizedFeatureVector())
    {
      printf("%.3f,", value);
    }
    printf(" ->   %d\n", data->getLabel());
  }
  return;

  printf("Test Data:\n");
  for(auto data : *testData)
  {
    for(auto value : *data->getNormalizedFeatureVector())
    {
      printf("%.3f,", value);
    }
    printf(" ->   %d\n", data->getLabel());
  }

  printf("Validation Data:\n");
  for(auto data : *validationData)
  {
    for(auto value : *data->getNormalizedFeatureVector())
    {
      printf("%.3f,", value);
    }
    printf(" ->   %d\n", data->getLabel());
  }

}

