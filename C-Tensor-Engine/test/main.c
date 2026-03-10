#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "operation.h"
#include "autograd.h"

int main() {
    printf("==================================================\n");
    printf("  C-Tensor Engine v0.3: 2D Broadcasting Test\n");
    printf("==================================================\n\n");

    // ==========================================================
    // 测试 1: 行广播 (Row Broadcasting) [2, 3] + [1, 3]
    // 模拟场景: 2 个样本 (每个 3 维特征) 加上同一个偏置向量
    // ==========================================================
    printf("[*] Test 1: Row Broadcasting [2, 3] + [1, 3]\n");
    
    int shape_A1[] = {2, 3};
    Tensor *A1 = create_tensor(2, shape_A1);
    A1->requires_grad = true;
    // A1 = [[1, 2, 3],
    //       [4, 5, 6]]
    for (int i = 0; i < 6; i++) A1->data[i] = (float)(i + 1);

    int shape_B1[] = {1, 3};
    Tensor *B1 = create_tensor(2, shape_B1);
    B1->requires_grad = true;
    // B1 = [[10, 20, 30]]
    B1->data[0] = 10.0f; B1->data[1] = 20.0f; B1->data[2] = 30.0f;

    Tensor *C1 = tensor_add(A1, B1);
    
    printf("    -> Forward C1 (A1 + B1) Expected:\n");
    printf("       [[11.0, 22.0, 33.0],\n");
    printf("        [14.0, 25.0, 36.0]]\n");
    printf("    -> Actual Output:\n");
    for(int i = 0; i < 2; i++) {
        printf("       [[%.1f, %.1f, %.1f]]\n", C1->data[i*3+0], C1->data[i*3+1], C1->data[i*3+2]);
    }

    // 引擎点火：默认会将 C1 的梯度全部设为 1.0f，然后反向传播
    tensor_backward(C1);

    printf("\n    -> Backward B1->grad (Row Reduce Sum) Expected:\n");
    printf("       [[2.0, 2.0, 2.0]]\n");
    printf("    -> Actual Output:\n");
    printf("       [[%.1f, %.1f, %.1f]]\n\n", B1->grad[0], B1->grad[1], B1->grad[2]);


    // ==========================================================
    // 测试 2: 列广播 (Column Broadcasting) [2, 3] + [2, 1]
    // 模拟场景: 给每一个样本加上其专属的一个独立标量权重
    // ==========================================================
    printf("[*] Test 2: Column Broadcasting [2, 3] + [2, 1]\n");
    
    int shape_A2[] = {2, 3};
    Tensor *A2 = create_tensor(2, shape_A2);
    A2->requires_grad = true;
    for (int i = 0; i < 6; i++) A2->data[i] = (float)(i + 1);

    int shape_B2[] = {2, 1};
    Tensor *B2 = create_tensor(2, shape_B2);
    B2->requires_grad = true;
    // B2 = [[100],
    //       [200]]
    B2->data[0] = 100.0f; B2->data[1] = 200.0f;

    Tensor *C2 = tensor_add(A2, B2);
    
    printf("    -> Forward C2 (A2 + B2) Expected:\n");
    printf("       [[101.0, 102.0, 103.0],\n");
    printf("        [204.0, 205.0, 206.0]]\n");
    printf("    -> Actual Output:\n");
    for(int i = 0; i < 2; i++) {
        printf("       [[%.1f, %.1f, %.1f]]\n", C2->data[i*3+0], C2->data[i*3+1], C2->data[i*3+2]);
    }

    tensor_backward(C2);

    printf("\n    -> Backward B2->grad (Column Reduce Sum) Expected:\n");
    printf("       [[3.0],\n        [3.0]]\n");
    printf("    -> Actual Output:\n");
    printf("       [[%.1f],\n        [%.1f]]\n\n", B2->grad[0], B2->grad[1]);


    // ==========================================================
    // 清理战场
    // ==========================================================
    free_tensor(A1); free_tensor(B1); free_tensor(C1);
    free_tensor(A2); free_tensor(B2); free_tensor(C2);

    printf("[+] All tests completed successfully.\n");
    return 0;
}