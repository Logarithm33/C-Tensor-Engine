#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float *data;   
    int *shape;    
    int ndim;      
    size_t size;
} Tensor;

Tensor* create_tensor(int ndim, const int* shape);

Tensor* tensor_add(const Tensor* a, const Tensor* b);

Tensor* tensor_matmul(const Tensor* a, const Tensor* b);

void free_tensor(Tensor* t);

#endif