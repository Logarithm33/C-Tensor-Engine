#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "operation.h"
#include "autograd.h"
#include "loss.h"
#include "optim.h"

int main() {
    printf("==================================================\n");
    printf("  C-Tensor Engine v0.2: Epic MLP Training Test\n");
    printf("==================================================\n\n");

    // ==========================================
    // 1. 准备数据 (Batch Size = 1)
    // ==========================================
    int shape_X[] = {1, 2};
    Tensor *X = create_tensor(2, shape_X);
    X->data[0] = 2.0f; X->data[1] = -1.0f; // 输入特征

    int shape_Y[] = {1, 1};
    Tensor *Target = create_tensor(2, shape_Y);
    Target->data[0] = 10.0f; // 模型需要努力学习逼近的真实目标值

    // ==========================================
    // 2. 初始化模型参数 (网络架构: 2 -> 3 -> 1)
    // ==========================================
    int shape_W1[] = {2, 3};
    Tensor *W1 = create_tensor(2, shape_W1);
    W1->requires_grad = true; // 这是模型的脑细胞，需要追踪梯度！
    // 随意初始化一些较小的浮点数
    W1->data[0] = 0.1f; W1->data[1] = 0.2f; W1->data[2] = -0.1f;
    W1->data[3] = -0.2f; W1->data[4] = 0.3f; W1->data[5] = 0.1f;

    int shape_B1[] = {1, 3};
    Tensor *B1 = create_tensor(2, shape_B1);
    B1->requires_grad = true;
    B1->data[0] = 0.1f; B1->data[1] = 0.1f; B1->data[2] = 0.1f;

    int shape_W2[] = {3, 1};
    Tensor *W2 = create_tensor(2, shape_W2);
    W2->requires_grad = true;
    W2->data[0] = 0.2f; W2->data[1] = -0.1f; W2->data[2] = 0.3f;

    // ==========================================
    // 3. 实例化上帝之手 (SGD 优化器)
    // ==========================================
    Tensor *params[] = {W1, B1, W2};
    // 学习率设为 0.01，你可以试着调大调小看看会发生什么
    SGD *optimizer = create_sgd(params, 3, 0.01f); 

    printf("[*] Model initialized. Starting training loop...\n\n");

    // ==========================================
    // 4. 终极训练大循环 (Epoch Loop)
    // ==========================================
    for (int epoch = 0; epoch <= 100; epoch++) {
        
        // --- A. 前向传播 (构建当前 Epoch 的计算图 DAG) ---
        // 层 1: H1 = X * W1
        Tensor *H1 = tensor_matmul(X, W1);
        // 加偏置: H2 = H1 + B1
        Tensor *H2 = tensor_add(H1, B1);
        // 激活: A1 = ReLU(H2)
        Tensor *A1 = tensor_relu(H2);
        // 层 2 (输出层): Pred = A1 * W2
        Tensor *Pred = tensor_matmul(A1, W2);

        // --- B. 计算误差 ---
        Tensor *Loss = tensor_mse_loss(Pred, Target);

        // 每 10 步打印一次学习进度
        if (epoch % 10 == 0) {
            printf("  Epoch %3d | Loss: %8.4f | Prediction: %8.4f\n", epoch, Loss->data[0], Pred->data[0]);
        }

        // --- C. 核心三部曲：清零 -> 反向传播 -> 走一步 ---
        sgd_zero_grad(optimizer);
        tensor_backward(Loss);
        sgd_step(optimizer);

        // --- D. 极度关键的内存回收：销毁本轮计算图 ---
        // 注意：我们绝不能 free 掉 X, Target, W1, B1, W2，因为下一轮还要用它们。
        // 我们只销毁这一轮产生的中间“废料节点”，从而释放计算图！
        free_tensor(H1);
        free_tensor(H2);
        free_tensor(A1);
        free_tensor(Pred);
        free_tensor(Loss);
    }

    printf("\n[*] Training completed!\n");

    // ==========================================
    // 5. 清理全局战场
    // ==========================================
    free_tensor(X);
    free_tensor(Target);
    free_tensor(W1);
    free_tensor(B1);
    free_tensor(W2);
    free_sgd(optimizer);

    printf("[+] All global memory freed successfully.\n");
    return 0;
}