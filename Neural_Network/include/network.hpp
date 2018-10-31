#ifndef __NETWORK_H
#define __NETWORK_H

#include <vector>
#include "data.h"
#include "neuron.hpp"

class network 
{

  std::vector<data *> * training_data;
  std::vector<data *> * test_data;
  std::vector<data *> * validation_data;

  std::vector<std::vector<neuron *> *> * hidden_layers;
  std::vector<neuron *> * output_layer;

  std::vector<double> network_output;



  public:

  void feed_forward(data *);
  void back_prop();
  void calculate_network_output(data *);
  void set_training_data(std::vector<data *> * data);
  void set_test_data(std::vector<data *> * data);
  void set_validation(std::vector<data *> * data);

  std::vector<double> get_network_output();
  

};

#endif