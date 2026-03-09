#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "operation.h"
#include "autograd.h"

int main() {
    printf("=== C-Tensor Engine v0.1: Full DAG Autograd Test ===\n\n");

    // ==========================================
    // 1. 初始化变量与数据
    // ==========================================
    int shape[] = {2, 2};
    Tensor* A = create_tensor(2, shape);
    Tensor* B = create_tensor(2, shape);

    // 开启求导追踪
    A->requires_grad = true;
    B->requires_grad = true;

    // A = [[1, 2], 
    //      [3, 4]]
    A->data[0] = 1.0f; A->data[1] = 2.0f;
    A->data[2] = 3.0f; A->data[3] = 4.0f;

    // B = [[5, 6], 
    //      [7, 8]]
    B->data[0] = 5.0f; B->data[1] = 6.0f;
    B->data[2] = 7.0f; B->data[3] = 8.0f;

    // ==========================================
    // 2. 构建非线性计算图 (DAG)
    // Loss = (A + B) * A
    // ==========================================
    printf("[*] Building DAG: Loss = (A + B) * A\n");
    
    Tensor* C = tensor_add(A, B);       // 第一步：C = A + B
    Tensor* Loss = tensor_matmul(C, A); // 第二步：Loss = C * A  <-- A 被重用了！

    printf("    -> Loss shape: [%d, %d]\n", Loss->shape[0], Loss->shape[1]);
    printf("    -> Loss data: [%.1f, %.1f, %.1f, %.1f]\n", 
            Loss->data[0], Loss->data[1], Loss->data[2], Loss->data[3]);

    // ==========================================
    // 3. 引擎点火：一键自动化反向传播！
    // ==========================================
    printf("\n[*] IGNITION: Calling tensor_backward(Loss)...\n");
    // 这里会自动推导拓扑序列、给 Loss 赋 1.0 的梯度，并按顺序触发闭包！
    tensor_backward(Loss);

    // ==========================================
    // 4. 验收微积分结果
    // ==========================================
    // 数学理论值推导:
    // C = [[6, 8], [10, 12]]
    // dLoss = [[1, 1], [1, 1]]
    // dC = dLoss * A^T = [[3, 7], [3, 7]]
    // dB = dC = [[3, 7], [3, 7]]
    //
    // A 的梯度来自两条路径：
    // 1. dA_来自乘法 = C^T * dLoss = [[16, 16], [20, 20]]
    // 2. dA_来自加法 = dC = [[3, 7], [3, 7]]
    // 最终 dA = [[16+3, 16+7], [20+3, 20+7]] = [[19, 23], [23, 27]]

    printf("\n[*] Validating Gradients:\n");
    if (A->grad && B->grad) {
        printf("    [+] dA[0][0] = %.1f (Expected: 19.0)\n", A->grad[0]);
        printf("    [+] dA[0][1] = %.1f (Expected: 23.0)\n", A->grad[1]);
        printf("    [+] dA[1][0] = %.1f (Expected: 23.0)\n", A->grad[2]);
        printf("    [+] dA[1][1] = %.1f (Expected: 27.0)\n", A->grad[3]);

        printf("\n    [+] dB[0][0] = %.1f (Expected:  3.0)\n", B->grad[0]);
        printf("    [+] dB[1][1] = %.1f (Expected:  7.0)\n", B->grad[3]);
    } else {
        printf("    [-] Error: Gradients not computed!\n");
    }

    // ==========================================
    // 5. 清理战场 (内存压力测试)
    // ==========================================
    printf("\n[*] Freeing Memory...\n");
    free_tensor(A);
    free_tensor(B);
    free_tensor(C);
    free_tensor(Loss); // 连同动态排队的数组和懒加载的梯度一起灰飞烟灭
    
    printf("[+] All memory freed successfully.\n");

    return 0;
}