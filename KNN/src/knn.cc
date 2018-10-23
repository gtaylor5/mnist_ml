#include "knn.h"
#include <cmath>
#include <limits>
#include <map>
#include "stdint.h"
#include "data_handler.h"


knn::knn(int val)
{
  k = val;
}

knn::knn()
{

}

knn::~knn()
{
  // NOTHING TO DO 
}

void knn::find_knearest(data *query_point)
{
  neighbors = new std::vector<data *>;
  double min = std::numeric_limits<double>::max();
  double previous_min = min;
  int index;
  for(int i = 0; i < k; i++)
  {
    if(i == 0)
    {
      for(int j = 0; j < training_data->size(); j++)
      {
        double dist = calculate_distance(query_point, training_data->at(j));
        training_data->at(j)->set_distance(dist);
        if(dist < min)
        {
          min = dist;
          index = j;
        }
      }
      neighbors->push_back(training_data->at(index));
      previous_min = min;
      min = std::numeric_limits<double>::max();
    } else 
    {
      for(int j = 0; j < training_data->size(); j++)
      {
        double dist = training_data->at(j)->get_distance();
        if(dist > previous_min && dist < min)
        {
          min = dist;
          index = j;
        }
      }
      neighbors->push_back(training_data->at(index));
      previous_min = min;
      min = std::numeric_limits<double>::max();
    }
  }
}
void knn::set_training_data(std::vector<data*>* vect)
{
  training_data = vect;
}
void knn::set_test_data(std::vector<data*>* vect)
{
  test_data = vect;
}
void knn::set_validation_data(std::vector<data*>* vect)
{
  validation_data = vect;
}
void knn::set_k(int val)
{
  k = val;
}

int knn::find_most_frequent_class()
{
  std::map<uint8_t, int> freq_map;
  for(int i = 0; i < neighbors->size(); i++)
  {
    if(freq_map.find(neighbors->at(i)->get_label()) == freq_map.end())
    {
      freq_map[neighbors->at(i)->get_label()] = 1;
    } else 
    {
      freq_map[neighbors->at(i)->get_label()]++;
    }
  }

  int best = 0;
  int max = 0;

  for(auto kv : freq_map)
  {
    if(kv.second > max)
    {
      max = kv.second;
      best = kv.first;
    }
  }
  delete neighbors;
  return best;

}

double knn::calculate_distance(data* query_point, data* input)
{
  double value = 0;
  if(query_point->get_feature_vector_size() != input->get_feature_vector_size())
  {
    printf("Vector size mismatch.\n");
    exit(1);
  }
#ifdef EUCLID
  for(unsigned i = 0; i < query_point->get_feature_vector_size(); i++)
  {
    value += pow(query_point->get_feature_vector()->at(i) - input->get_feature_vector()->at(i),2);
  }
  return sqrt(value);
#elif defined MANHATTAN
  //do some stuff
#endif
}

double knn::validate_perforamnce()
{
  double current_performance = 0;
  int count = 0;
  for(data *query_point : *validation_data)
  {
    find_knearest(query_point);
    int prediction = find_most_frequent_class();
    if(prediction == query_point->get_label())
    {
      count++;
    }
  }
  current_performance = ((double)count)*100.0/((double)validation_data->size());
  printf("Validation Performance for K = %d: %.3f\n", k, current_performance);
  return current_performance;
}
double knn::test_performance()
{
  double current_performance = 0;
  int count = 0;
  for(data *query_point : *test_data)
  {
    find_knearest(query_point);
    int prediction = find_most_frequent_class();
    if(prediction == query_point->get_label())
    {
      count++;
    }
  }
  current_performance = ((double)count)*100.0/((double)test_data->size());
  printf("Validation Performance for K = %d: %.3f\n", k, current_performance);
  return current_performance;
}

int
main()
{
  data_handler *dh = new data_handler();
  dh->read_input_data("../train-images-idx3-ubyte");
  dh->read_label_data("../train-labels-idx1-ubyte");
  dh->count_classes();
  dh->split_data();
  knn *nearest = new knn();
  nearest->set_k(1);
  nearest->set_training_data(dh->get_training_data());
  nearest->set_test_data(dh->get_test_data());
  nearest->set_validation_data(dh->get_validation_data());
  double performance = 0;
  double best_performance = 0;
  int best_k = 1;
  for(int k = 1; k <= 100; k++)
  {
    if(k == 1)
    {
      performance = nearest->validate_perforamnce();
      best_performance = performance;
    } else 
    {
      nearest->set_k(k);
      performance = nearest->validate_perforamnce();
      if(performance > best_performance)
      {
        best_performance = performance;
        best_k = k;
      }
    }
  }
  nearest->set_k(best_k);
  nearest->test_performance();
}