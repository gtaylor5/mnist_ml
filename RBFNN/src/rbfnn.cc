#include "../include/rbfnn.hpp"

rbfnn::rbfnn(std::map<uint8_t, int> class_map, std::vector<cluster_t *> * clusters)
{
  for(int i = 0; i < clusters->size(); i++)
  {
    rbfneuron_t *neuron = new rbfneuron_t;
    neuron->center = clusters->at(i)->centroid;
    hidden_layer->push_back(neuron);
  }

  for(auto kv : class_map)
  {
    output_neuron_t *neuron = new output_neuron_t;
    neuron->target = kv.second;
    neuron->initializeWeights(training_data->at(0)->get_feature_vector_size());
  }
}
rbfnn::~rbfnn()
{

}

void rbfnn::calculate_basis_outputs(data* query_point)
{
  for(rbfneuron_t *neuron : *hidden_layer)
  {
    neuron->activate(query_point);
  }
}

void rbfnn::collect_basis_outputs()
{
  basis_outputs.clear();
  for(rbfneuron_t *neuron : *hidden_layer)
  {
    basis_outputs.push_back(neuron->output);
  }
}

void rbfnn::calculate_network_outputs()
{
  for(output_neuron_t *neuron : *output_layer)
  {
    neuron->activate(basis_outputs);
  }
}

void rbfnn::update_weights(data* query_point)
{

}
void rbfnn::train()
{
  for(data* query_point : *trainin_data)
  {
    calculate_basis_outputs(query_point);
    collect_basis_outputs();
    calculate_network_outputs();
    update_weights(query_point);
  }
}
void rbfnn::test()
{

}

int
main()
{
  data_handler *dh = new data_handler();
  dh->read_input_data("../train-images-idx3-ubyte");
  dh->read_label_data("../train-labels-idx1-ubyte");
  dh->count_classes();
  dh->split_data();
  kmeans *km = new kmeans(450);
  km->set_training_data(dh->get_training_data());
  km->set_test_data(dh->get_test_data());
  km->set_validation_data(dh->get_validation_data());
  km->init_clusters();
  km->train();

  rbfnn *nn = new rbfnn(dh->get_class_map(), km->get_clusters());
  nn->set_training_data(dh->get_training_data());
  nn->set_test_data(dh->get_test_data());
  nn->set_validation_data(dh->get_validation_data());
  return 0;
}
