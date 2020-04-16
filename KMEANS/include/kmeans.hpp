#ifndef __KMEANS_HPP
#define __KMEANS_HPP
#include "Common.hpp"
#include <unordered_set>
#include <limits>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <math.h>
#include <map>
#include "DataHandler.h"
typedef struct cluster 
{
 std::vector<double> *centroid;
 std::vector<Data *> *clusterPoints;
 std::map<int, int> classCounts;
 int mostFrequentClass;
 cluster(Data *initialPoint)
 {
   centroid = new std::vector<double>;
   clusterPoints = new std::vector<Data *>;
   for(auto val : *(initialPoint->getNormalizedFeatureVector()))
   {
     if(isnan(val))
       centroid->push_back(0);
     else
      centroid->push_back(val);
   }
   clusterPoints->push_back(initialPoint);
   classCounts[initialPoint->getLabel()] = 1;
   mostFrequentClass = initialPoint->getLabel();
 }
 
void add_to_cluster(Data* point)
 {
   int previous_size = clusterPoints->size();
   clusterPoints->push_back(point);
   for(int i = 0; i < centroid->size(); i++)
   {
   	double val = centroid->at(i);
     val *= previous_size;
     val += point->getNormalizedFeatureVector()->at(i);
     val /= (double)clusterPoints->size();
     centroid->at(i) = val;
   }
   if(classCounts.find(point->getLabel()) == classCounts.end())
   {
     classCounts[point->getLabel()] = 1;
   } else
   {
     classCounts[point->getLabel()]++;
   }
   set_mostFrequentClass();
 }
 void set_mostFrequentClass()
 {
   int best_class;
   int freq = 0;
   for(auto kv : classCounts)
   {
    if(kv.second > freq)
    {
      freq = kv.second;
      best_class = kv.first;
    }
   }
   mostFrequentClass = best_class;
 }
} cluster_t;

class kmeans : public CommonData
{
  int numClusters;
  std::vector<cluster_t *> *clusters;
  std::unordered_set<int> *usedIndexes;
  public:
  kmeans(int k);
  void initClusters();
  void initClustersForEachClass();
  void train();
  double euclideanDistance(std::vector<double> *, Data *);
  double validate();
  double test();
  std::vector<cluster_t *> * getClusters();
};
#endif
