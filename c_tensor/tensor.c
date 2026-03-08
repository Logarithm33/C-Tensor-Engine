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

    int row_a = a->shape[0];
    int column_a = a->shape[1];
    int column_b = b->shape[1];

    int result_shape[] = {row_a, column_b};
    Tensor *result = create_tensor(2, result_shape);
    if(!result) return NULL;

    for(int row_result = 0; row_result < row_a; row_result++) {
        for(int column_result = 0; column_result < column_b; column_result++) {
            float sum = 0.0f;
            for(int k = 0; k < column_a; k++) {
                sum += a->data[row_result * column_a + k] * b->data[k * column_b + column_result];
            }
            result->data[row_result * column_b + column_result] = sum;
        }
    }
    return result;
}

void free_tensor(Tensor *t) {
    if(t) {
        free(t->data);
        free(t->shape);
        free(t);

        if(t->grad) {
            free(t->grad);
        }

        if(t->_prev) {
            free(t->_prev);
        }
    }
}