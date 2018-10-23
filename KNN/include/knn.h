#ifndef __KNN_H
#define __KNN_H

#include <vector>
#include "data.h"

// O(k*n) where k is the number of neighbors and N is the size of training data
// O(n) + O(k*n) + k

class knn 
{
  int k;
  std::vector<data *> * neighbors;
  std::vector<data *> * training_data;
  std::vector<data *> * test_data;
  std::vector<data *> * validation_data;

  public:
  knn(int);
  knn();
  ~knn();

  void find_knearest(data *query_point);
  void set_training_data(std::vector<data*>* vect);
  void set_test_data(std::vector<data*>* vect);
  void set_validation_data(std::vector<data*>* vect);
  void set_k(int val);
  int find_most_frequent_class();
  double calculate_distance(data* query_point, data* input);
  double validate_perforamnce();
  double test_performance();


};

#endif