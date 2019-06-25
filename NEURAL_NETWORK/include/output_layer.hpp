#ifndef OUTPUT_LAYER_HPP
#define OUTPUT_LAYER_HPP
#include "layer.hpp"
#include "data.h"

class OutputLayer : public Layer 
{
  public:
    OutputLayer(int prev, int curr) : Layer(prev, curr) {}
    void feedForward(Layer);
    void backProp(data *data);
    void updateWeights(double, Layer*);
    void print();
};
#endif
