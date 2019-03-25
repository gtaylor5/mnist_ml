#ifndef __NUERON_HPP
#define __NEURON_HPP

#include <vector>
#include <map>
#include <cmath>
#include "data.h"

typedef struct rbfneuron
{
  std::vector<double> *center = NULL;
  double sigma = 1.5;
  double output;

  void activate(data* query_point)
  {
    //query_point->print_normalized_vector();
    double dot = get_norm(query_point->get_normalized_feature_vector());
    double gaussian = (-1.0) / (2.0 * pow(sigma, 2));
    output = exp(gaussian*dot);
 //   fprintf(stderr, "Output: %.5f\n", output);
  }

  double get_norm(std::vector<double> *feature_vector)
  {
    double dist = 0.0;
    for(int i = 0; i < feature_vector->size(); i++)
    {
      dist += (double)pow(feature_vector->at(i) - center->at(i), 2);
    }
    return sqrt(dist);
  }

} rbfneuron_t;

typedef struct output_neuron
{
  std::vector<double> weights;
  double output;
  int target;

  void initializeWeights(int size)
  {
    for(int i = 0; i < size; i++)
    {
      double val = (double) rand() / RAND_MAX;
      weights.push_back((-1.0 + val*2)); // between -1.0 and 1.0
    }
  }

  void activate(std::vector<double> input)
  {
    double raw_output = get_norm(input);
    output = 1.0 / (1.0 + exp(-1.0 * raw_output));
  }

  double get_norm(std::vector<double> feature_vector)
  {
    double dist = 0.0;
    for(int i = 0; i < feature_vector.size(); i++)
    {
      dist += pow((double)feature_vector.at(i) - weights.at(i), 2);
    }
    return sqrt(dist);
  }

} output_neuron_t;

#endif
