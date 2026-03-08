#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"

int main() {
    printf("=== C-Tensor MatMul Autograd Test ===\n\n");

    // 1. 准备数据: A (2x3) 和 B (3x2)
    int shape_A[] = {2, 3};
    int shape_B[] = {3, 2};
    Tensor* A = create_tensor(2, shape_A);
    Tensor* B = create_tensor(2, shape_B);

    A->requires_grad = true;
    B->requires_grad = true;

    // A = [[1, 2, 3], 
    //      [4, 5, 6]]
    for (size_t i = 0; i < A->size; i++) A->data[i] = (float)(i + 1);

    // B = [[1, 2], 
    //      [3, 4], 
    //      [5, 6]]
    for (size_t i = 0; i < B->size; i++) B->data[i] = (float)(i + 1);

    // 2. 前向传播: C = A * B
    Tensor* C = tensor_matmul(A, B);
    printf("[*] Forward Pass: C = A * B\n");
    printf("    -> C shape: [%d, %d]\n", C->shape[0], C->shape[1]);

    // 3. 模拟误差并反向传播
    printf("\n[*] Simulating Upstream Gradient (dC)...\n");
    C->grad = (float*)malloc(C->size * sizeof(float));
    // 假设上游传下来的 dC 全是 1.0
    for (size_t i = 0; i < C->size; i++) C->grad[i] = 1.0f;

    printf("[*] Executing C->_backward(C)...\n");
    C->_backward(C);

    // 4. 验证梯度的微积分结果
    // 数学推导 dA = dC * B^T
    // dC(2x2)全是1, B^T(2x3)是 [[1, 3, 5], [2, 4, 6]]
    // dA的预期结果应该是 [[3, 7, 11], [3, 7, 11]]
    printf("\n[*] Validating Gradients:\n");
    if (A->grad) {
        printf("    [+] dA[0][0] = %.1f (Expected: 3.0)\n", A->grad[0]);
        printf("    [+] dA[0][1] = %.1f (Expected: 7.0)\n", A->grad[1]);
        printf("    [+] dA[0][2] = %.1f (Expected: 11.0)\n", A->grad[2]);
    }

    // 5. 清理战场
    free_tensor(A);
    free_tensor(B);
    free_tensor(C);
    printf("\n[+] Test Finished & Memory Freed.\n");

    return 0;
}