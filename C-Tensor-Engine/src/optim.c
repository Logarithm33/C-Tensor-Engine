#include <stdlib.h>
#include <string.h>
#include "optim.h"

SGD *create_sgd(Tensor **params, int param_count, float learning_rate) {
    if (!params || param_count <= 0 || learning_rate <= 0.0f) return NULL;

    SGD *optimizer = (SGD *)malloc(sizeof(SGD));
    if (!optimizer) return NULL;

    optimizer->params = (Tensor **)malloc(sizeof(Tensor *) * param_count);
    if (!optimizer->params) {
        free(optimizer);
        return NULL;
    }

    memcpy(optimizer->params, params, sizeof(Tensor *) * param_count);
    optimizer->param_count = param_count;
    optimizer->learning_rate = learning_rate;

    return optimizer;
}

void sgd_step(SGD *optimizer) {
    if (!optimizer || !optimizer->params) return;

    for (int i = 0; i < optimizer->param_count; i++) {
        Tensor *param = optimizer->params[i];
        if (param && param->grad) {
            for (size_t j = 0; j < param->size; j++) {
                param->data[j] -= optimizer->learning_rate * param->grad[j];
            }
        }
    }
}

void sgd_zero_grad(SGD *optimizer) {
    if (!optimizer || !optimizer->params) return;

    for (int i = 0; i < optimizer->param_count; i++) {
        Tensor *param = optimizer->params[i];
        param->_visited = false; 
        if (param && param->grad) {
            memset(param->grad, 0, sizeof(float) * param->size);
        }
    }
}

void free_sgd(SGD *optimizer) {
    if (!optimizer) return;

    if (optimizer->params) {
        free(optimizer->params);
    }
    free(optimizer);
}