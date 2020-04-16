#ifndef __DATA_H
#define __DATA_H

#include <vector>
#include "stdint.h" // uint8_t 
#include "stdio.h"
class Data
{
  std::vector<uint8_t> *featureVector;
  std::vector<double> *normalizedFeatureVector;
  std::vector<int> *classVector;
  uint8_t label; 
  uint8_t enumeratedLabel; // A -> 1
  double distance;

  public:
  void setDistance(double);
  void setFeatureVector(std::vector<uint8_t>*);
  void setNormalizedFeatureVector(std::vector<double>*);
  void setClassVector(int counts);
  void appendToFeatureVector(uint8_t);
  void appendToFeatureVector(double);
  void setLabel(uint8_t);
  void setEnumeratedLabel(uint8_t);
  void printVector();
  void printNormalizedVector();

  double getDistance();
  int getFeatureVectorSize();
  uint8_t getLabel();
  uint8_t getEnumeratedLabel();

  std::vector<uint8_t> * getFeatureVector();
  std::vector<double> * getNormalizedFeatureVector();
  std::vector<int> getClassVector();

};

#endif
