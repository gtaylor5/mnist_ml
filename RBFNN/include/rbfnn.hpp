#ifndef __RBFNN_HPP
#define __RBFNN_HPP

#include "neuron.hpp"
#include "common.hpp"
#include "../KMEANS/include/kmeans.hpp"
#include <map>
class rbfnn : public common_data
{

  std::vector<rbfneuron_t*> *hidden_layer;
  std::vector<output_neuron_t*> *output_layer;
  std::vector<double> basis_outputs;

  public:

  rbfnn(std::map<uint8_t, int> class_map, std::vector<cluster_t *> * clusters, int size);
  ~rbfnn();

  void calculate_basis_outputs(data* query_point);
  void collect_basis_outputs();
  void calculate_network_outputs();
  void update_weights(data *);
  void train();
  double validate();
  void test();

};

#endif
