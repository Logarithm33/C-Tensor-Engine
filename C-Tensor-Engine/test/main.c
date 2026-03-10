#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tensor.h"
#include "operation.h"
#include "loss.h"
#include "optim.h"
#include "data.h"
#include "autograd.h"

int main() {
    // ==========================================
    // 1. 数据加载：MNIST 手写数字数据集
    // ==========================================

    Tensor *X_train = load_mnist_images("data/train-images-idx3-ubyte");
    Tensor *Y_train = load_mnist_labels("data/train-labels-idx1-ubyte");

    if (!X_train || !Y_train) {
        perror("Error: Failed to load MNIST dataset");
        return -1;
    }
    printf("Images: %zu, Labels: %zu\n\n", X_train->shape[0], Y_train->shape[0]);

    // ==========================================
    // 2. 模型构建：一个简单的两层全连接网络
    // ==========================================

    int batch_size = 256;         // 每次看 256 张图片
    int epochs = 5;               // 总共复习 5 轮
    float learning_rate = 0.1f;   // 学习率
    int input_size = 784;         // 28x28 像素展平
    int hidden_size = 128;        // 隐藏层神经元数量
    int num_classes = 10;         // 0~9 十个数字
    int num_batches = X_train->shape[0] / batch_size;  // 每轮需要多少个批次
    
    int shape_W1[] = {input_size, hidden_size};
    Tensor *W1 = create_tensor(2, shape_W1);
    W1->requires_grad = true;

    int shape_B1[] = {1, hidden_size};
    Tensor *B1 = create_tensor(2, shape_B1);
    B1->requires_grad = true;

    int shape_W2[] = {hidden_size, num_classes};
    Tensor *W2 = create_tensor(2, shape_W2);
    W2->requires_grad = true;

    int shape_B2[] = {1, num_classes};
    Tensor *B2 = create_tensor(2, shape_B2);
    B2->requires_grad = true;

    // 参数初始化：小随机数，打破对称性
    srand(42);
    for (size_t i = 0; i < W1->size; i++) W1->data[i] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
    for (size_t i = 0; i < W2->size; i++) W2->data[i] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
    for (size_t i = 0; i < B1->size; i++) B1->data[i] = 0.0f;
    for (size_t i = 0; i < B2->size; i++) B2->data[i] = 0.0f;

    Tensor *params[] = {W1, B1, W2, B2};
    SGD *optimizer = create_sgd(params, 4, learning_rate);

    // ==========================================
    // 3. 训练准备：批次容器和训练循环的前置条件
    // ==========================================

    int shape_batch_X[] = {batch_size, input_size};
    Tensor *batch_X = create_tensor(2, shape_batch_X);
    
    int shape_batch_Y[] = {batch_size, num_classes};
    Tensor *batch_Y = create_tensor(2, shape_batch_Y);

    printf("Starting Training Loop (Epochs: %d, Batch Size: %d)\n\n", epochs, batch_size);

    // ==========================================
    // 4. 训练循环：前向传播、损失计算、反向传播、参数更新
    // ==========================================

    for (int epoch = 1; epoch <= epochs; epoch++) {
        float total_loss = 0.0f;
        int total_correct = 0;

        for (int b = 0; b < num_batches; b++) {
            // A. 数据切片：把这一批次的数据装填进容器
            memcpy(batch_X->data, X_train->data + b * batch_size * input_size, batch_size * input_size * sizeof(float));
            memcpy(batch_Y->data, Y_train->data + b * batch_size * num_classes, batch_size * num_classes * sizeof(float));

            // B. 前向传播 
            Tensor *H1_raw = tensor_matmul(batch_X, W1);
            Tensor *H1 = tensor_add(H1_raw, B1); 
            Tensor *A1 = tensor_relu(H1);
            
            Tensor *Pred_raw = tensor_matmul(A1, W2);
            Tensor *Pred = tensor_add(Pred_raw, B2); 
            
            Tensor *Loss = tensor_cross_entropy_loss(Pred, batch_Y);

            total_loss += Loss->data[0];

            // C. 统计正确率
            for (int i = 0; i < batch_size; i++) {
                int max_pred_idx = 0; float max_pred_val = Pred->data[i * num_classes];
                int max_true_idx = 0; float max_true_val = batch_Y->data[i * num_classes];
                for (int j = 1; j < num_classes; j++) {
                    if (Pred->data[i * num_classes + j] > max_pred_val) {
                        max_pred_val = Pred->data[i * num_classes + j];
                        max_pred_idx = j;
                    }
                    if (batch_Y->data[i * num_classes + j] > max_true_val) {
                        max_true_val = batch_Y->data[i * num_classes + j];
                        max_true_idx = j;
                    }
                }
                if (max_pred_idx == max_true_idx) total_correct++;
            }

            // D. 反向传播
            sgd_zero_grad(optimizer);
            tensor_backward(Loss);
            sgd_step(optimizer);

            // E. 清理计算图的中间废料
            free_tensor(H1_raw);
            free_tensor(H1);
            free_tensor(A1);
            free_tensor(Pred_raw);
            free_tensor(Pred);
            free_tensor(Loss);

            // 每 50 个 Batch 汇报一次
            if ((b + 1) % 50 == 0 || b == num_batches - 1) {
                printf("    Epoch %d/%d | Batch %3d/%d | Loss: %.4f | Acc: %.2f%%\n", 
                        epoch, epochs, b + 1, num_batches, 
                        total_loss / (b + 1), 
                        (float)total_correct / ((b + 1) * batch_size) * 100.0f);
            }
        }
        printf(">>> Epoch %d Completed | Avg Loss: %.4f | Train Acc: %.2f%%\n\n", 
                epoch, total_loss / num_batches, (float)total_correct / (num_batches * batch_size) * 100.0f);
    }

    // ==========================================
    // 5. 储存参数：模型落盘
    // ==========================================
    printf("Training Finished! Saving model parameters...\n");
    save_tensor(W1, "W1.ctensor");
    save_tensor(B1, "B1.ctensor");
    save_tensor(W2, "W2.ctensor");
    save_tensor(B2, "B2.ctensor");

    // ==========================================
    // 6. 全局释放
    // ==========================================
    free_tensor(X_train); free_tensor(Y_train);
    free_tensor(batch_X); free_tensor(batch_Y);
    free_tensor(W1); free_tensor(B1); free_tensor(W2); free_tensor(B2);
    free_sgd(optimizer);

    printf("All global memory freed successfully.\n");
    return 0;
}