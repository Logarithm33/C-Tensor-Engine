#include <stdio.h>
#include "tensor.h"

int main() {
    printf("=== C-Tensor Engine v0.1 Memory Test ===\n");

    // 1. 定义一个 3D 张量，形状为 [3, 4, 5]
    int shape[] = {3, 4, 5};
    int ndim = 3;

    // 2. 调用你写的引擎分配内存
    printf("[*] Creating tensor with shape [%d, %d, %d]...\n", shape[0], shape[1], shape[2]);
    Tensor* t = create_tensor(ndim, shape);

    if (!t) {
        printf("[!] Error: Failed to allocate tensor!\n");
        return -1;
    }

    // 3. 验证逻辑属性
    printf("[+] Tensor created successfully!\n");
    printf("    -> ndim: %d\n", t->ndim);
    printf("    -> Expected size: 60\n");
    printf("    -> Actual size: %zu\n", t->size);

    // 4. 验证物理内存写入 (向第一个和最后一个元素写值，测试是否越界)
    if (t->size > 0) {
        t->data[0] = 3.14f;
        t->data[t->size - 1] = 2.71f;
        printf("[+] Memory write test passed.\n");
        printf("    -> t->data[0] = %.2f\n", t->data[0]);
        printf("    -> t->data[%zu] = %.2f\n", t->size - 1, t->data[t->size - 1]);
    }

    // 5. 释放内存
    printf("[*] Freeing tensor memory...\n");
    free_tensor(t);
    printf("[+] Memory freed.\n");

    return 0;
}