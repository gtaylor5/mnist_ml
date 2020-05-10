#ifndef __DATA_HANDLER_H
#define __DATA_HANDLER_H

#include "fstream"
#include "stdint.h"
#include "data.h"
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <math.h>

class DataHandler
{
  std::vector<Data *> *dataArray; // all of the data
  std::vector<Data *> *trainingData;
  std::vector<Data *> *testData;
  std::vector<Data *> *validationData;
  int class_counts;
  int featureVectorSize;
  std::map<uint8_t, int> classFromInt;
  std::map<std::string, int> classFromString; //string key

  public:
  const double TRAIN_SET_PERCENT = .1;
  const double TEST_SET_PERCENT = .075;
  const double VALID_SET_PERCENT = 0.005;

  DataHandler();
  ~DataHandler();
  
  void readCsv(std::string, std::string);
  void readInputData(std::string path);
  void readLabelData(std::string path);
  void splitData();
  void countClasses();
  void normalize();
  void print();
  
  int getClassCounts();
  int getDataArraySize();
  int getTrainingDataSize();
  int getTestDataSize();
  int getValidationSize();

  uint32_t format(const unsigned char* bytes);

  std::vector<Data *> * getTrainingData();
  std::vector<Data *> * getTestData();
  std::vector<Data *> * getValidationData();
  std::map<uint8_t, int> getClassMap();

};

#endif
