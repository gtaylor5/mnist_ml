#include "input_layer.hpp"

void InputLayer::setLayerOutputs(Data *d)
{
    this->layerOutputs = *d->getNormalizedFeatureVector();
}