#ifndef __EDGE_H
#define __EDGE_H

#include <map>
#include <vector>

class neuron;

class edge
{
  double weight;
  double best_weight;
  double prev_weight_delta;
  double weight_delta;

  neuron *left_neuron;

  public:
  static int counter;
  int id;

  edge(neuron *);
  ~edge();

  void set_weight(double);
  void set_best_weight(double);
  void set_delta_weight(double);
  void set_weight_as_best();

  double get_previous_weight_delta();
  double get_weight();
  neuron * get_left_neuron();
  
};

#endif