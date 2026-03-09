#ifndef OPTIM_H
#define OPTIM_H

#include "tensor.h"

typedef struct {
    Tensor **params;
    int param_count;
    float learning_rate;
} SGD;

SGD *create_sgd(Tensor **params, int param_count, float learning_rate);

void sgd_step(SGD *optimizer);

void sgd_zero_grad(SGD *optimizer);

void free_sgd(SGD *optimizer);

#endif