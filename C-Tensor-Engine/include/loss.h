#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

Tensor* tensor_mse_loss(Tensor* pred, Tensor*target);

#endif