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

static void cross_entropy_loss_backward (Tensor *self) {
    if (!self || !self->grad) return;

    Tensor *pred = self->_prev[0];
    Tensor *target = self->_prev[1];

    if (pred->requires_grad) {
        if (!pred->grad) {
            pred->grad = (float *)calloc(pred->size, sizeof(float));
        }

        int batch_size = pred->shape[0];
        int num_classes = pred->shape[1];

        float factor = 1.0 / batch_size;
        float upstream_grad = self->grad[0];

        for (int i = 0; i < batch_size; i++) {
            float max_val = pred->data[i * num_classes];
            for (int j = 1; j < num_classes; j++) {
                if (pred->data[i * num_classes + j] > max_val) {
                    max_val = pred->data[i * num_classes + j];
                }
            }

            float sum_exp = 0.0f;
            for (int j = 0; j < num_classes; j++) {
                sum_exp += expf(pred->data[i * num_classes + j] - max_val);
            }

            for (int j = 0; j < num_classes; j++) {
                float y = target->data[i * num_classes + j];
                float prob = expf(pred->data[i * num_classes + j] - max_val) / sum_exp;
                pred->grad[i * num_classes + j] += (prob - y) * factor * upstream_grad;
            }
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

Tensor* tensor_cross_entropy_loss(Tensor* pred, Tensor* target) {
    if (!pred || !target) return NULL;
    if (pred->ndim != 2 || target->ndim != 2) return NULL;
    if (pred->shape[0] != target->shape[0] || pred->shape[1] != target->shape[1]) return NULL;

    int batch_size = pred->shape[0];
    int num_classes = pred->shape[1];

    int result_shape[] = {1};
    Tensor *result = create_tensor(1, result_shape);
    if (!result) return NULL;

    float total_loss = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        float max_val = pred->data[i * num_classes];
        for (int j = 1; j < num_classes; j++) {
            if (pred->data[i * num_classes + j] > max_val) {
                max_val = pred->data[i * num_classes + j];
            }
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++) {
            sum_exp += expf(pred->data[i * num_classes + j] - max_val);
        }

        for (int j = 0; j < num_classes; j++) {
            float y = target->data[i * num_classes + j];
            if (y > 0.0) {
                float prob = expf(pred->data[i * num_classes + j] - max_val) / sum_exp;
                total_loss -= y * logf(prob + 1e-8f);
            }
        }
    }

    result->data[0] = total_loss / batch_size;

    result->requires_grad = pred->requires_grad;
    if (result->requires_grad) {
        result->_prev_count = 2;
        result->_prev = (Tensor **)malloc(sizeof(Tensor *) * 2);
        result->_prev[0] = pred;
        result->_prev[1] = target;

        strncpy(result->_op, "CrossEntropy", 16);
        result->_backward = cross_entropy_loss_backward;
    }
    return result;
}