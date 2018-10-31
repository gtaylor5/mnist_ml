#ifndef __NEURON_H
#define __NEURON_H

#include <vector>
#include <map>
#include "edge.hpp"

class neuron
{
  double output;
  std::vector<edge *> * edges;
  std::map<int, edge *> connection_map;

  public:
  static int counter;
  int id;
  neuron();
  ~neuron();
  void calculate_output(); // dot product
  void activate();

};

#endif