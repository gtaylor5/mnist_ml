#include "../include/rbfnn.hpp"
#include <cstdlib>

rbfnn::rbfnn(std::map<uint8_t, int> class_map, std::vector<cluster_t *> * clusters, int size)
{
  hidden_layer = new std::vector<rbfneuron_t*>;
  output_layer = new std::vector<output_neuron_t*>;

  // initalize hidden layers

  for(int i = 0; i < clusters->size(); i++)
  {
    rbfneuron_t *neuron = new rbfneuron_t;
    neuron->center = clusters->at(i)->centroid;
    hidden_layer->push_back(neuron);
  }

  // initialize output layer

  for(auto kv : class_map)
  {
    output_neuron_t *neuron = new output_neuron_t;
    neuron->target = kv.second;
    neuron->initializeWeights(size);
    output_layer->push_back(neuron);
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
  int target_class = query_point->get_label();
  for(auto output_neuron : *output_layer)
  {
    // If output target is the target then we want the output to be 1.
    // otherwise we want the target to be zero.
    int desired_output = 1;
    if(output_neuron->target != target_class)
    {
      desired_output = 0;
    }
    double current_output = output_neuron->output;
    for(int i = 0; i < basis_outputs.size(); i++)
    {
      double gradient = (desired_output - current_output)*basis_outputs.at(i)*(current_output * (1-current_output));
      output_neuron->weights[i] += gradient;
    }
  }
}
void rbfnn::train()
{
  double threshold = 90.0;
  while(validate() < threshold)
  {
    for(data* query_point : *training_data)
    {
      calculate_basis_outputs(query_point);
      collect_basis_outputs();
      calculate_network_outputs();
      update_weights(query_point);
    }
  }
}


double rbfnn::validate()
{
  double performance = 0.0;
  for(data *query_point : *validation_data)
  {
    calculate_basis_outputs(query_point);
    collect_basis_outputs();
    calculate_network_outputs();
    int target_class = query_point->get_label();
    int best_index = 0;
    double max_output = std::numeric_limits<double>::min();
    for(int i = 0; i < output_layer->size(); i++)
    {
      if(output_layer->at(i)->output > max_output)
      {
        max_output = output_layer->at(i)->output;
        best_index = i;
      }
    }
    if(target_class == output_layer->at(best_index)->target)
      performance++;
  }
  printf("Current Performance = %.2f\n", 100.0 * (performance / (double) validation_data->size()));
  return 100.0 * (performance / (double) validation_data->size());
}

void rbfnn::test()
{

}

int
main(int argc, char *argv[])
{
  data_handler *dh = new data_handler();
  dh->read_input_data("../train-images-idx3-ubyte");
  dh->read_label_data("../train-labels-idx1-ubyte");
  dh->count_classes();
  dh->split_data();
  kmeans *km = new kmeans(atoi(argv[1]));
  km->set_training_data(dh->get_training_data());
  km->set_test_data(dh->get_test_data());
  km->set_validation_data(dh->get_validation_data());
  fprintf(stderr, "Initializing Clusters\n");
  km->init_clusters();
  fprintf(stderr,"Training KMEANS\n");
  km->train();
  fprintf(stderr,"Initializing RBFNN\n");
  rbfnn *nn = new rbfnn(dh->get_class_map(), km->get_clusters(), dh->get_training_data()->at(0)->get_feature_vector_size());
  nn->set_training_data(dh->get_training_data());
  nn->set_test_data(dh->get_test_data());
  nn->set_validation_data(dh->get_validation_data());
  nn->train();
  nn->test();
  return 0;
}
