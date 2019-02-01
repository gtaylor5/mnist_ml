#! /bin/bash

if [[ -z $MNIST_ML_ROOT ]]; then
  echo "Please define MNIST_ML_ROOT"
  exit 1
fi

dir=$(echo "$@" | tr a-z A-Z)
model_lower=$(echo "$@" | tr A-Z a-z)
echo $model_lower
mkdir -p $MNIST_ML_ROOT/$dir/include $MNIST_ML_ROOT/$dir/src
touch $MNIST_ML_ROOT/$dir/Makefile
touch $MNIST_ML_ROOT/$dir/include/"$model_lower.hpp"
touch $MNIST_ML_ROOT/$dir/src/"$model_lower.cc"
