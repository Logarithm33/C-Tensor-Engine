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

void add_backward (Tensor *self) {
    if (!self || !self->grad) return;

    Tensor *a = self->_prev[0];
    Tensor *b = self->_prev[1];

    if (a->requires_grad) {
        if (!a->grad) {
            a->grad = (float *)calloc(a->size, sizeof(float));
        }
        for (size_t i = 0; i < a->size; i++) {
            a->grad[i] += self->grad[i];
        }
    }
    if (b->requires_grad) {
        if (!b->grad) {
            b->grad = (float *)calloc(b->size, sizeof(float));
        }
        for (size_t i = 0; i < b->size; i++) {
            b->grad[i] += self->grad[i];
        }
    }
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

    result->requires_grad = a->requires_grad || b->requires_grad;
    if(result->requires_grad) {
        result->_prev_count = 2;
        result->_prev = (Tensor **)malloc(sizeof(Tensor *) * 2);
        result->_prev[0] = (Tensor *)a;
        result->_prev[1] = (Tensor *)b;

        strncpy(result->_op, "add", 16);
        result->_backward = add_backward;
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

    result->requires_grad = a->requires_grad || b->requires_grad;
    if(result->requires_grad) {
        result->_prev_count = 2;
        result->_prev = (Tensor **)malloc(sizeof(Tensor *) * 2);
        result->_prev[0] = (Tensor *)a;
        result->_prev[1] = (Tensor *)b;
        strncpy(result->_op, "matmul", 16);
        // result->_backward = matmul_backward; 
    }
    return result;
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