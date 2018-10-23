#include "data_handler.h"
#include "kmeans.h"
#include "cluster.h"
#include <map>

kmeans::kmeans(int _k)
{
  k = _k;
  clusters = new std::vector<cluster *>();
}

kmeans::~kmeans()
{
  // FREE STUFF
}

void kmeans::reset()
{
  delete clusters;
}

void kmeans::set_k(int _k)
{
  k = _k;
}

void kmeans::initialize_clusters()
{
  for(int i = 0; i < k; i++)
  {
    cluster *c = new cluster();
    int random_index = rand() % training_data->size();
    c->append_to_cluster(training_data->at(random_index));
    training_data->erase(training_data->begin() + random_index);
    clusters->push_back(c);
  }
}

void kmeans::set_test_data(std::vector<data *> *vect)
{
  test_data = vect;
}
void kmeans::set_training_data(std::vector<data *> *vect)
{
  training_data = vect;
}
void kmeans::set_validation_data(std::vector<data *> *vect)
{
  validation_data = vect;
}
void kmeans::assign_to_cluster(data *d)
{
  int closest_cluster = 0;
  double distance;
  for(int i = 0; i < clusters->size(); i++)
  {
    if(i == 0)
    {
      clusters->at(i)->calculate_distance(d);
      distance = clusters->at(i)->get_distance();
    } else 
    {
      double new_dist;
      clusters->at(i)->calculate_distance(d);
      new_dist = clusters->at(i)->get_distance();
      if(new_dist < distance)
      {
        distance = new_dist;
        closest_cluster = i;
      }
    }
  }
  clusters->at(closest_cluster)->append_to_cluster(d);
}

int kmeans::find_best_cluster(data *d)
{
  int closest_cluster = 0;
  double distance;
  for(int i = 0; i < clusters->size(); i++)
  {
    if(i == 0)
    {
      clusters->at(i)->calculate_distance(d);
      distance = clusters->at(i)->get_distance();
    } else 
    {
      double new_dist;
      clusters->at(i)->calculate_distance(d);
      new_dist = clusters->at(i)->get_distance();
      if(new_dist < distance)
      {
        distance = new_dist;
        closest_cluster = i;
      }
    }
  }
  return closest_cluster;
}

int kmeans::get_prediction(int cl)
{
  cluster *c;
  c = clusters->at(cl);
  std::map<uint8_t, int> freqs;
  for(int i = 0; i < c->get_cluster()->size(); i++)
  {
    if(freqs.find(c->get_cluster()->at(i)->get_label()) == freqs.end())
    {
      freqs[c->get_cluster()->at(i)->get_label()] = 1;
    } else 
    {
      freqs[c->get_cluster()->at(i)->get_label()]++;
    }
  }
  int best_class;
  int max = 0;
  for(auto kv : freqs)
  {
    if(kv.second > max)
    {
      max = kv.second;
      best_class = kv.first;
    }
  }
  return best_class;
}

void kmeans::train()
{
  for(int i = 0; i < training_data->size(); i++)
  {
    assign_to_cluster(training_data->at(i));
  }
  printf("Done Clustering Training Data.\n");
}
double kmeans::validate()
{
  double performance = 0;
  int count = 0;
  for(int i = 0; i < validation_data->size(); i++)
  {
    int cluster = find_best_cluster(validation_data->at(i));
    int prediction = get_prediction(cluster);
    if(prediction == validation_data->at(i)->get_label())
    {
      count++;
    }
  }
  performance = ((double) count) *100.0 / ((double) validation_data->size());
  printf("Performance = %.3f %%\n", performance);
  return performance;
}

double kmeans::test()
{
  //
  return 0.0;
}

int
main()
{
  data_handler *dh = new data_handler();
  dh->read_input_data("../train-images-idx3-ubyte");
  dh->read_label_data("../train-labels-idx1-ubyte");
  dh->count_classes();
  dh->split_data();
  kmeans *k;
  int best_k = 1;
  double best_peformance = 0;
  for(int i = 1; i < 150; i++)
  {
    k = new kmeans(i);
    k->set_training_data(dh->get_training_data());
    k->set_test_data(dh->get_test_data());
    k->set_validation_data(dh->get_validation_data());
    k->initialize_clusters();
    k->train();
    double current_perf = k->validate();
    if(current_perf > best_peformance)
    {
      best_peformance = current_perf;
      best_k = i;
      printf("New Best Performance of: %.3f %% with K = %d\n", best_peformance, best_k);
    }
    delete k;
  }

  printf("Overall Best Performance: %.3f\n",best_peformance);
  printf("Overall Best K: %d\n", best_k);
  
}