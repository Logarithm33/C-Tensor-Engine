#include <stdio.h>
#include "tensor.h"

int main() {
    printf("=== C-Tensor Engine Operator Test ===\n\n");

    // ==========================================
    // 1. 测试张量加法 (tensor_add)
    // ==========================================
    printf("[*] Testing tensor_add...\n");
    int shape_add[] = {2, 3};
    Tensor* a = create_tensor(2, shape_add);
    Tensor* b = create_tensor(2, shape_add);

    // 赋值: a = [1, 2, 3, 4, 5, 6]
    // 赋值: b = [2, 4, 6, 8, 10, 12]
    for (size_t i = 0; i < a->size; i++) {
        a->data[i] = (float)(i + 1);
        b->data[i] = (float)(i + 1) * 2.0f;
    }

    Tensor* c = tensor_add(a, b);
    if (c) {
        printf("    [+] tensor_add success.\n");
        printf("        c->data[0] = %.1f (Expected: 3.0)\n", c->data[0]);
        printf("        c->data[5] = %.1f (Expected: 18.0)\n", c->data[5]);
    } else {
        printf("    [-] tensor_add failed.\n");
    }
    printf("\n");


    // ==========================================
    // 2. 测试矩阵乘法 (tensor_matmul)
    // ==========================================
    printf("[*] Testing tensor_matmul...\n");
    int shape_A[] = {2, 3};
    int shape_B[] = {3, 2};
    Tensor* A = create_tensor(2, shape_A);
    Tensor* B = create_tensor(2, shape_B);

    // 赋值: A (2x3) 
    // [[1, 2, 3], 
    //  [4, 5, 6]]
    for (size_t i = 0; i < A->size; i++) {
        A->data[i] = (float)(i + 1);
    }

    // 赋值: B (3x2)
    // [[1, 2], 
    //  [3, 4], 
    //  [5, 6]]
    for (size_t i = 0; i < B->size; i++) {
        B->data[i] = (float)(i + 1);
    }

    Tensor* C = tensor_matmul(A, B);
    if (C) {
        printf("    [+] tensor_matmul success.\n");
        printf("        Result shape: [%d, %d]\n", C->shape[0], C->shape[1]);
        
        // 验证: 1*1 + 2*3 + 3*5 = 22
        printf("        C[0][0] = %.1f (Expected: 22.0)\n", C->data[0]);
        // 验证: 1*2 + 2*4 + 3*6 = 28
        printf("        C[0][1] = %.1f (Expected: 28.0)\n", C->data[1]);
        // 验证: 4*1 + 5*3 + 6*5 = 49
        printf("        C[1][0] = %.1f (Expected: 49.0)\n", C->data[2]);
        // 验证: 4*2 + 5*4 + 6*6 = 64
        printf("        C[1][1] = %.1f (Expected: 64.0)\n", C->data[3]);
    } else {
        printf("    [-] tensor_matmul failed.\n");
    }
    printf("\n");


    // ==========================================
    // 3. 释放内存
    // ==========================================
    printf("[*] Freeing all memory...\n");
    free_tensor(a);
    free_tensor(b);
    free_tensor(c);
    free_tensor(A);
    free_tensor(B);
    free_tensor(C);
    
    printf("[+] All memory freed successfully.\n");

    return 0;
}