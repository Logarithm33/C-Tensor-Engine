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

void save_tensor(Tensor *target, const char *filename) {
    if(!target || !filename) return;

    FILE *file = fopen(filename, "wb");
    if(!file) {
        perror("Failed to open file for writing");
        return;
    }

    fwrite(&(target->ndim), sizeof(int), 1, file);
    fwrite(target->shape, sizeof(int), target->ndim, file);
    fwrite(target->data, sizeof(float), target->size, file);

    fclose(file);
    printf("Saved tensor to %s successfully.\n", filename);
}

Tensor* load_tensor(const char *filename) {
    if(!filename) return NULL;

    FILE *file = fopen(filename, "rb");
    if(!file) {
        perror("Failed to open file for reading");
        return NULL;
    }

    int ndim;
    if (fread(&ndim, sizeof(int), 1, file) != 1) {
        fclose(file);
        return NULL;
    }

    int *shape = (int *)malloc(sizeof(int) * ndim);
    if(!shape) {
        fclose(file);
        return NULL;
    }
    if (fread(shape, sizeof(int), ndim, file) != ndim) {
        free(shape);
        fclose(file);
        return NULL;
    }

    Tensor *tensor = create_tensor(ndim, shape);
    free(shape);

    if(!tensor) {
        fclose(file);
        return NULL;
    }
    
    if (fread(tensor->data, sizeof(float), tensor->size, file) != tensor->size) {
        free_tensor(tensor);
        fclose(file);
        return NULL;
    }

    fclose(file);
    printf("Loaded tensor from %s successfully.\n", filename);
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