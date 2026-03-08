#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"

int main() {
    printf("=== C-Tensor Engine Autograd (Backward) Test ===\n\n");

    // ==========================================
    // 1. 准备节点与数据
    // ==========================================
    int shape[] = {2, 2}; // 简单的 2x2 矩阵
    Tensor* a = create_tensor(2, shape);
    Tensor* b = create_tensor(2, shape);

    // [核心开关]：告诉引擎，这两个基础变量需要跟踪梯度！
    a->requires_grad = true;
    b->requires_grad = true;

    for (size_t i = 0; i < a->size; i++) {
        a->data[i] = (float)(i + 1);       // A = [1, 2, 3, 4]
        b->data[i] = (float)((i + 1) * 2); // B = [2, 4, 6, 8]
    }

    // ==========================================
    // 2. 前向传播 (Forward Pass)
    // ==========================================
    printf("[*] Forward Pass: C = A + B\n");
    Tensor* c = tensor_add(a, b);

    if (c) {
        printf("    [+] Forward Success.\n");
        printf("        C->data[0] = %.1f (Expected: 3.0)\n", c->data[0]);
        // 验证图拓扑是否成功建立
        printf("        C->requires_grad = %s (Expected: true)\n", c->requires_grad ? "true" : "false");
        printf("        C->_op = %s\n", c->_op);
    }

    // ==========================================
    // 3. 模拟误差回传 & 触发反向传播 (Backward Pass)
    // ==========================================
    printf("\n[*] Simulating Upstream Gradient arriving at C...\n");
    // 懒加载：我们手动为 C 分配梯度内存，并填入测试用的误差值 2.0
    c->grad = (float*)malloc(c->size * sizeof(float));
    for (size_t i = 0; i < c->size; i++) {
        c->grad[i] = 2.0f; 
    }
    printf("    -> Set C->grad[*] to 2.0\n");

    printf("\n[*] IGNITION: Calling C->_backward(C)...\n");
    // 唤醒 C 脑子里的闭包！
    if (c->_backward) {
        c->_backward(c);
        printf("    [+] Backward closure executed.\n");
    } else {
        printf("    [-] Error: C has no backward closure!\n");
    }

    // ==========================================
    // 4. 验收微积分结果
    // ==========================================
    printf("\n[*] Validating Gradients for A and B:\n");
    if (a->grad && b->grad) {
        printf("    [+] A->grad[0] = %.1f (Expected: 2.0)\n", a->grad[0]);
        printf("    [+] A->grad[3] = %.1f (Expected: 2.0)\n", a->grad[3]);
        printf("    [+] B->grad[0] = %.1f (Expected: 2.0)\n", b->grad[0]);
    } else {
        printf("    [-] Error: Gradients were NOT allocated for A or B. Lazy allocation failed!\n");
    }

    // ==========================================
    // 5. 内存压力测试
    // ==========================================
    printf("\n[*] Freeing DAG Memory...\n");
    free_tensor(a);
    free_tensor(b);
    free_tensor(c);
    printf("[+] All memory freed successfully.\n");

    return 0;
}