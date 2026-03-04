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

    tensor->shape = (int *)malloc(sizeof(int)*ndim);
    if(!tensor->shape) {
        free(tensor->data);
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, ndim * sizeof(int));

    tensor->ndim = ndim;
    tensor->size = total_count;
    return tensor;
}

Tensor* tensor_add(const Tensor* a, const Tensor* b) {
    if (!a || !b) return NULL;
    if(a->ndim != b->ndim) return NULL;
    for(int i = 0; i < a->ndim; i++) {
        if(a->shape[i] != b->shape[i]) return NULL;
    }
    Tensor *result = create_tensor(a->ndim, a->shape);
    if(!result) return NULL;

    for(size_t i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    return result;
}

Tensor* tensor_matmul(const Tensor* a, const Tensor* b) {
    if (!a || !b) return NULL;
    if(a->ndim != 2 || b->ndim != 2) return NULL;
    if(a->shape[1] != b->shape[0]) return NULL;

    int m = a->shape[0];
    int n = a->shape[1];
    int p = b->shape[1];

    int result_shape[] = {m, p};
    Tensor *result = create_tensor(2, result_shape);
    if(!result) return NULL;

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < p; j++) {
            float sum = 0.0f;
            for(int k = 0; k < n; k++) {
                sum += a->data[i * n + k] * b->data[k * p + j];
            }
            result->data[i * p + j] = sum;
        }
    }
    return result;
}

void free_tensor(Tensor *t) {
    if(t) {
        free(t->data);
        free(t->shape);
        free(t);
    }
}