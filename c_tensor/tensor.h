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

Tensor* tensor_add(const Tensor* a, const Tensor* b);

Tensor* tensor_matmul(const Tensor* a, const Tensor* b);

void add_backward (Tensor *self);

void matmul_backward (Tensor *self);

void build_topo (Tensor *t,Tensor **topo_list, int *topo_size);

void tensor_backward (Tensor *loss);

void free_tensor(Tensor* t);

#endif