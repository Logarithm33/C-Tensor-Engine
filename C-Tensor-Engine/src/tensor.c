#include "tensor.h"

Tensor* create_tensor(int ndim, const int* shape) {
    size_t total_count = 1;
    for(int i = 0; i < ndim; i++) {
        total_count *= shape[i];
    }
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    if(!tensor) return NULL;

    tensor->data = (float *)malloc(sizeof(float) * total_count);
    if(!tensor->data) {
        free(tensor);
        return NULL;
    }

    tensor->shape = (int *)malloc(sizeof(int) * ndim);
    if(!tensor->shape) {
        free(tensor->data);
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, ndim * sizeof(int));

    tensor->ndim = ndim;
    tensor->size = total_count;

    tensor->grad = NULL;
    tensor->requires_grad = false;

    tensor->_prev = NULL;
    tensor->_prev_count = 0;
    tensor->_op[0] = '\0';

    tensor->_backward = NULL;

    tensor->_visited = false;

    return tensor;
}

void free_tensor(Tensor *t) {
    if(t) {
        if(t->grad) {
            free(t->grad);
        }

        if(t->_prev) {
            free(t->_prev);
        }

        free(t->data);
        free(t->shape);
        free(t);
    }
}