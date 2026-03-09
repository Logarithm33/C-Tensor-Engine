#include "loss.h"

static void mse_loss_backward (Tensor *self) {
    if (!self || !self->grad) return;

    Tensor *pred = self->_prev[0];
    Tensor *target = self->_prev[1];

    if (pred->requires_grad) {
        if (!pred->grad) {
            pred->grad = (float *)calloc(pred->size, sizeof(float));
        }

        float factor = 2.0f / pred->size;

        float upstream_grad = self->grad[0];

        for (size_t i = 0; i < pred->size; i++) {
            float diff = pred->data[i] - target->data[i];
            pred->grad[i] += factor * diff * upstream_grad;
        }
    }
}

Tensor *tensor_mse_loss(Tensor* pred, Tensor* target) {
    if (!pred || !target) return NULL;
    if(pred->ndim != target->ndim) return NULL;

    for(int i = 0; i < pred->ndim; i++) {
        if(pred->shape[i] != target->shape[i]) return NULL;
    }
    int result_shape[] = {1};
    Tensor *result = create_tensor(1, result_shape);
    if(!result) return NULL;

    float sum = 0.0f;
    for(size_t i = 0; i < pred->size; i++) {
        float diff = pred->data[i] - target->data[i];
        sum += diff * diff;
    }
    result->data[0] = sum / pred->size;

    result->requires_grad = pred->requires_grad;
    if(result->requires_grad) {
        result->_prev_count = 2;
        result->_prev = (Tensor **)malloc(sizeof(Tensor *) * 2);
        result->_prev[0] = pred;
        result->_prev[1] = target;

        strncpy(result->_op, "MSE", 16);
        result->_backward = mse_loss_backward;
    }
    return result;
}