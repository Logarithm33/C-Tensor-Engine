#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

typedef struct Tensor Tensor;

struct Tensor{
    float *data;   
    int *shape;    
    int ndim;      
    size_t size;

    float *grad;
    bool requires_grad;

    Tensor **_prev;
    int _prev_count;
    char _op[16];

    void (*_backward)(Tensor *self);

    bool _visited;
};

Tensor* create_tensor(int ndim, const int* shape);

void save_tensor(Tensor *target, const char *filename);

Tensor* load_tensor(const char *filename);

void free_tensor(Tensor* t);

#endif