#include "operation.h"

static void add_backward (Tensor *self) {
    if (!self || !self->grad) return;

    Tensor *a = self->_prev[0];
    Tensor *b = self->_prev[1];

    int a_rows = (a->ndim == 2) ? a->shape[0] : 1;
    int a_cols = (a->ndim == 2) ? a->shape[1] : a->shape[0];
    int b_rows = (b->ndim == 2) ? b->shape[0] : 1;
    int b_cols = (b->ndim == 2) ? b->shape[1] : b->shape[0];

    int result_rows = self->shape[0];
    int result_cols = self->shape[1];

    if (a->requires_grad) {
        if (!a->grad) {
            a->grad = (float *)calloc(a->size, sizeof(float));
        }
        for (int i = 0; i < result_rows; i++) {
            for (int j = 0; j < result_cols; j++) {
                int a_row_index = (a_rows == 1) ? 0 : i;
                int a_col_index = (a_cols == 1) ? 0 : j;
                a->grad[a_row_index * a_cols + a_col_index] += self->grad[i * result_cols + j];
            }
        }
    }
    if (b->requires_grad) {
        if (!b->grad) {
            b->grad = (float *)calloc(b->size, sizeof(float));
        }
        for (int i = 0; i < result_rows; i++) {
            for (int j = 0; j < result_cols; j++) {
                int b_row_index = (b_rows == 1) ? 0 : i;
                int b_col_index = (b_cols == 1) ? 0 : j;
                b->grad[b_row_index * b_cols + b_col_index] += self->grad[i * result_cols + j];
            }
        }
    }
}

static void matmul_backward (Tensor *self) {
    if (!self || !self->grad) return;

    Tensor *a = self->_prev[0];
    Tensor *b = self->_prev[1];

    int row_a = a->shape[0];
    int column_a = a->shape[1];
    int column_b = b->shape[1];

    if (a->requires_grad) {
        if (!a->grad) {
            a->grad = (float *)calloc(a->size, sizeof(float));
        }
        for (int i = 0; i < row_a; i++) {
            for (int j = 0; j < column_a; j++) {
                float sum = 0.0f;
                for (int k = 0; k < column_b; k++) {
                    sum +=self->grad[i * column_b + k] * b->data[j * column_b + k];
                }
                a->grad[i * column_a + j] += sum;
            }
        }
    }
    if (b->requires_grad) {
        if (!b->grad) {
            b->grad = (float *)calloc(b->size, sizeof(float));
        }
        for (int i = 0; i < column_a; i++) {
            for (int j = 0; j < column_b; j++) {
                float sum = 0.0f;
                for (int k = 0; k < row_a; k++) {
                    sum += self->grad[k * column_b + j] * a->data[k * column_a + i];
                }
                b->grad[i * column_b + j] += sum;
            }
        }
    }
}

static void relu_backward (Tensor *self) {
    if (!self || !self->grad) return;

    Tensor *a = self->_prev[0];

    if (a->requires_grad) {
        if (!a->grad) {
            a->grad = (float *)calloc(a->size, sizeof(float));
        }
        for (size_t i = 0; i < a->size; i++) {
            a->grad[i] += (a->data[i] > 0 ? 1.0f : 0.0f) * self->grad[i];
        }
    }
}

Tensor* tensor_add(const Tensor* a, const Tensor* b) {
    if (!a || !b) return NULL;
    if(a->ndim > 2 || b->ndim > 2) return NULL;

    int a_rows = (a->ndim == 2) ? a->shape[0] : 1;
    int a_cols = (a->ndim == 2) ? a->shape[1] : a->shape[0];
    int b_rows = (b->ndim == 2) ? b->shape[0] : 1;
    int b_cols = (b->ndim == 2) ? b->shape[1] : b->shape[0];

    int result_rows = (a_rows > b_rows) ? a_rows : b_rows;
    int result_cols = (a_cols > b_cols) ? a_cols : b_cols;

    if ((a_rows != b_rows && a_rows != 1 && b_rows != 1) || (a_cols != b_cols && a_cols != 1 && b_cols != 1)) {
        return NULL;
    }
    int result_shape[] = {result_rows, result_cols};
    Tensor *result = create_tensor(2, result_shape);
    if(!result) return NULL;

    for (int i = 0; i < result_rows; i++) {
        for (int j = 0; j < result_cols; j++) {
            int a_row_index = (a_rows == 1) ? 0 : i;
            int a_col_index = (a_cols == 1) ? 0 : j;
            int b_row_index = (b_rows == 1) ? 0 : i;
            int b_col_index = (b_cols == 1) ? 0 : j;

            float a_val = a->data[a_row_index * a_cols + a_col_index];
            float b_val = b->data[b_row_index * b_cols + b_col_index];
            result->data[i * result_cols + j] = a_val + b_val;
        }
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
        result->_backward = matmul_backward; 
    }
    return result;
}

Tensor* tensor_relu(const Tensor* a) {
    if (!a) return NULL;

    Tensor *result = create_tensor(a->ndim, a->shape);
    if(!result) return NULL;

    for(size_t i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] > 0 ? a->data[i] : 0;
    }

    result->requires_grad = a->requires_grad;
    if(result->requires_grad) {
        result->_prev_count = 1;
        result->_prev = (Tensor **)malloc(sizeof(Tensor *));
        result->_prev[0] = (Tensor *)a;

        strncpy(result->_op, "relu", 16);
        result->_backward = relu_backward;
    }
    return result;
}