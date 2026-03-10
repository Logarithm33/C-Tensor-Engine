#ifndef DATA_H
#define DATA_H

#include "tensor.h"

Tensor* load_mnist_images(const char* filename);

Tensor* load_mnist_labels(const char* filename);

#endif