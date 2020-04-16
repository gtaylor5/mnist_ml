#include "../include/kmeans.hpp"

kmeans::kmeans(int k)
{
 numClusters = k;
 clusters = new std::vector<cluster_t *>;
 usedIndexes = new std::unordered_set<int>;
}

void kmeans::initClusters()
{
 for(int i = 0; i < numClusters; i++)
 {
   int index = (rand() % trainingData->size());
   while(usedIndexes->find(index) != usedIndexes->end())
   {
     index = (rand() % trainingData->size());
   }
   clusters->push_back(new cluster_t(trainingData->at(index)));
   usedIndexes->insert(index);
 }
}

void kmeans::initClustersForEachClass()
{
 std::unordered_set<int> classes_used;
 for(int i = 0; i < trainingData->size(); i++)
 {
   if(classes_used.find(trainingData->at(i)->getLabel()) == classes_used.end())
   {
     clusters->push_back(new cluster_t(trainingData->at(i)));
     classes_used.insert(trainingData->at(i)->getLabel());
     usedIndexes->insert(i);
   }
 }
}
void kmeans::train()
{
 while(usedIndexes->size() < trainingData->size())
 {
   int index = (rand() % trainingData->size());
   while(usedIndexes->find(index) != usedIndexes->end())
   {
     index = (rand() % trainingData->size());
   }
   double min_dist = std::numeric_limits<double>::max();
   int best_cluster = 0;
   for(int j = 0; j < clusters->size(); j++)
   {
     double dist = euclideanDistance(clusters->at(j)->centroid, trainingData->at(index));
     if(dist < min_dist)
     {
       min_dist = dist;
       best_cluster = j;
     }
   }
   clusters->at(best_cluster)->add_to_cluster(trainingData->at(index));
   usedIndexes->insert(index);
 }
}

double kmeans::euclideanDistance(std::vector<double> * centroid, Data *query_point)
{
 double dist = 0.0;
 for(int i = 0; i < centroid->size(); i++)
 {
   dist += pow(centroid->at(i) - query_point->getNormalizedFeatureVector()->at(i), 2);
 }
 return sqrt(dist);
}
double kmeans::validate()
{
 double num_correct = 0.0;
 for(auto query_point : *validationData)
 {
   double min_dist = std::numeric_limits<double>::max();
   int best = 0;
   for(int i = 0; i < clusters->size(); i++)
   {
     double current_dist = euclideanDistance(clusters->at(i)->centroid, query_point);
     if(current_dist < min_dist)
     {
       min_dist = current_dist;
       best = i;
     }
   }
   if(clusters->at(best)->mostFrequentClass == query_point->getLabel()) num_correct++;
 }
 return 100.0 * (num_correct / (double) validationData->size());
}
double kmeans::test()
{
 double num_correct = 0.0;
 for(auto query_point : *testData)
 {
   double min_dist = std::numeric_limits<double>::max();
   int best = 0;
   for(int i = 0; i < clusters->size(); i++)
   {
     double current_dist = euclideanDistance(clusters->at(i)->centroid, query_point);
     if(current_dist < min_dist)
     {
       min_dist = current_dist;
       best = i;
     }
   }
   if(clusters->at(best)->mostFrequentClass == query_point->getLabel()) num_correct++;
 }
 return 100.0 * (num_correct / (double) testData->size());
}

std::vector<cluster_t *> * kmeans::getClusters()
{
  return this->clusters;
}


int
main()
{
  DataHandler *dh = new DataHandler();
  dh->readInputData("../train-images-idx3-ubyte");
  dh->readLabelData("../train-labels-idx1-ubyte");
  dh->countClasses();
  dh->splitData();
  double performance = 0;
  double best_performance = 0;
  int best_k = 1;
  for(int k = dh->getClassCounts(); k < dh->getTrainingData()->size()*0.1; k++)
  {
    kmeans *km = new kmeans(k);
    km->setTrainingData(dh->getTrainingData());
    km->setTestData(dh->getTestData());
    km->setValidationData(dh->getValidationData());
    km->initClusters();
    km->train();
    performance = km->validate();
    printf("Current Perforamnce @ K = %d: %.2f\n", k, performance);
    if(performance > best_performance)
    {
      best_performance = performance;
      best_k = k;
    }
  }
  kmeans *km = new kmeans(best_k);
  km->setTrainingData(dh->getTrainingData());
  km->setTestData(dh->getTestData());
  km->setValidationData(dh->getValidationData());
  km->initClusters();
  km->train();
  printf("Overall Performance: %.2f\n",km->test());

}

