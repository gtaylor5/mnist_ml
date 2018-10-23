#ifndef __CLUSTER_H
#define __CLUSTER_H

#include <vector>
#include "data.h"

class cluster
{
  std::vector<double> *centroid;
  double distance;
  std::vector<data *> * _cluster;
  public:
  cluster();
  ~cluster();
  void update_centroid();
  void calculate_distance(data *);
  void append_to_cluster(data *);
  double get_distance();
  std::vector<data *> * get_cluster();
  std::vector<double> *get_centroid();
  
};

#endif