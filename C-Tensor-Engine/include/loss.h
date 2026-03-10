#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"
#include <math.h>

Tensor* tensor_mse_loss(Tensor* pred, Tensor*target);

Tensor* tensor_cross_entropy_loss(Tensor* pred, Tensor* target);

#endif