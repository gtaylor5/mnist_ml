#include "input_layer.hpp"

void InputLayer::setLayerOutputs(data *d)
{
  this->layerOutputs = *d->get_normalized_feature_vector();
}
