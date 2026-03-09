#ifndef OPERATION_H
#define OPERATION_H

#include "tensor.h"

Tensor* tensor_add(const Tensor* a, const Tensor* b);
Tensor* tensor_matmul(const Tensor* a, const Tensor* b);

#endif