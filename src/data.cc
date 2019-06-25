#include "../include/data.h"

void data::set_distance(double dist)
{
  m_distance = dist;
}
void data::set_feature_vector(std::vector<uint8_t>* vect)
{
  m_feature_vector = vect;
}


void data::set_normalized_feature_vector(std::vector<double>* vect)
{
  m_normalized_feature_vector = vect;
}
void data::append_to_feature_vector(uint8_t val)
{
  m_feature_vector->push_back(val);
}
void data::append_to_feature_vector(double val)
{
  m_normalized_feature_vector->push_back(val);
}
void data::set_label(uint8_t val)
{
  m_label = val;
}
void data::set_enumerated_label(uint8_t val)
{
  m_enumerated_label = val;
}

void data::setClassVector(int classCounts)
{
  class_vector = new std::vector<int>();
  for(int i = 0; i < classCounts; i++)
  {
    if(i == m_label)
      class_vector->push_back(1);
    else
      class_vector->push_back(0);
  }
}

void data::print_vector()
{
  printf("[ ");
  for(uint8_t val : *m_feature_vector)
  {
    printf("%u ", val);
  }
  printf("]\n");
}

void data::print_normalized_vector()
{
  printf("[ ");
  for(auto val : *m_normalized_feature_vector)
  {
    printf("%.2f ", val);
  }
  printf("]\n");
  
}

double data::get_distance()
{
  return m_distance;
}

int data::get_feature_vector_size()
{
  return m_feature_vector->size();
}
uint8_t data::get_label()
{
  return m_label;
}
uint8_t data::get_enumerated_label()
{
  return m_enumerated_label;
}

std::vector<uint8_t> * data::get_feature_vector()
{
  return m_feature_vector;
}
std::vector<double> * data::get_normalized_feature_vector()
{
  return m_normalized_feature_vector;
}

std::vector<int>  data::getClassVector()
{
  return *class_vector;
}
