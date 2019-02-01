#ifndef __KNN_H
#define __KNN_H

#include "common.hpp"

// O(k*n) where k is the number of neighbors and N is the size of training data
// O(n) + O(k*n) + k

class knn : public common_data
{
  int k;
  std::vector<data *> * neighbors;

  public:
  knn(int);
  knn();
  ~knn();

  void find_knearest(data *query_point);
  void set_k(int val);
  int find_most_frequent_class();
  double calculate_distance(data* query_point, data* input);
  double validate_perforamnce();
  double test_performance();


};

#endif
