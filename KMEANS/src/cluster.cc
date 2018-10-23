#include "data.h"
#include "cluster.h"
#include <cmath>

cluster::cluster()
{
  _cluster = new std::vector<data *>;
}
cluster::~cluster()
{
  delete _cluster;
}
void cluster::update_centroid()
{
  // after a new point is added we need to recalculate centroid.
  // assume each new point is added at the back of the vector

  // average = sum(vals) / vals.size()
  // new average algo is:
  // sum(vals) = average * vals.size()
  // sum(vals) += new_val
  // average = sum(vals) * vals+1;

  // O(N) instead of O(N^2)

  for(int j = 0; j < centroid->size(); j++)
  {
    double prev = centroid->at(j) * (_cluster->size() - 1);
    prev += _cluster->back()->get_feature_vector()->at(j);
    centroid->at(j) = (prev / (double) _cluster->size());
  }
}

void cluster::calculate_distance(data* d)
{
  // using euclidean distance
  int size = centroid->size();

  if(size != d->get_feature_vector()->size())
  {
    printf("Error the centroid and data vector sizes do not match. exiting\n");
    exit(1);
  }
  double value = 0;
  for(int i = 0; i < size; i++)
  {
    value += pow(d->get_feature_vector()->at(i)-centroid->at(i), 2);
  }
  distance = sqrt(value);
}
void cluster::append_to_cluster(data *d)
{
  if(_cluster->size() == 0)
  {
    centroid = new std::vector<double>();
    _cluster->push_back(d);
    for(int i = 0; i < d->get_feature_vector()->size(); i++)
    {
      centroid->push_back(d->get_feature_vector()->at(i));
    }
  } else
  {
    _cluster->push_back(d);
    update_centroid();
  }
  
}
double cluster::get_distance()
{
  return distance;
}
std::vector<data *>* cluster::get_cluster()
{
  return _cluster;
}
std::vector<double> * cluster::get_centroid()
{
  return centroid;
}