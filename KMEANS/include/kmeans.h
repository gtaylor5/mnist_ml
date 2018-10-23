#ifndef __KMEANS_H
#define __KMEANS_H

#include <vector>
#include "data.h"
#include "cluster.h"
#include <unordered_set>


class kmeans
{
  int k;
  std::vector<cluster *> *clusters;
  std::vector<data *> *training_data;
  std::vector<data *> *test_data;
  std::vector<data *> *validation_data;
  std::unordered_set<int> used_indexes;
  
  public:
  kmeans(int);
  ~kmeans();
  void reset();
  void set_k(int);
  void initialize_clusters(); // set up clusters based on k value
  void set_test_data(std::vector<data *>*);
  void set_training_data(std::vector<data *>*);
  void set_validation_data(std::vector<data *>*);
  void assign_to_cluster(data *);
  void train();
  int find_best_cluster(data *d);
  int get_prediction(int cluster);
  double validate();
  double test();

};



#endif