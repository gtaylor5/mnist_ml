#ifndef __KNN_H
#define __KNN_H

#include "Common.hpp"

// O(k*n) where k is the number of neighbors and N is the size of training data
// O(n) + O(k*n) + k

class KNN : public CommonData
{
  int k;
  std::vector<Data *> * neighbors;

  public:
  KNN(int);
  KNN();
  ~KNN();

  void findKnearest(Data *queryPoint);
  void setK(int val);
  int findMostFrequentClass();
  double calculateDistance(Data* queryPoint, Data* input);
  double validatePerformance();
  double testPerformance();
};

#endif
