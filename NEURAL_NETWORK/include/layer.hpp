#ifndef __LAYER_HPP
#define __LAYER_HPP
#include "neuron.hpp"
#include <stdint.h>
#include <vector>

static int layerId = 0;

class Layer {
  
  public:

    int id;
    int currentLayerSize;
    std::vector<Neuron *> neurons;
    std::vector<double> layerOutputs;
    Layer(int, int);
    Layer();
    ~Layer();
    
};
#endif
