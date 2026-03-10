#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "data.h"

static uint32_t swap_endian(uint32_t val) {
    return ((val >> 24) & 0x000000FF) |
           ((val >> 8) & 0x0000FF00) |
           ((val << 8) & 0x00FF0000) |
           ((val << 24) & 0xFF000000);
}

Tensor* load_mnist_images(const char* filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Could not open file: %s\n", filename);
        return NULL;
    }

    uint32_t magic, num_images, num_rows, num_cols;
    if (fread(&magic, sizeof(uint32_t), 1, file) != 1 ||
        fread(&num_images, sizeof(uint32_t), 1, file) != 1 ||
        fread(&num_rows, sizeof(uint32_t), 1, file) != 1 ||
        fread(&num_cols, sizeof(uint32_t), 1, file) != 1) {
        printf("Error: Failed to read header from %s\n", filename);
        fclose(file);
        return NULL;
    }

    magic = swap_endian(magic);
    num_images = swap_endian(num_images);
    num_rows = swap_endian(num_rows);
    num_cols = swap_endian(num_cols);

    if (magic != 2051) {
        printf("Invalid magic number in MNIST image file: %u\n", magic);
        fclose(file);
        return NULL;
    }

    int shape[] = {num_images, num_rows * num_cols};
    Tensor *tensor = create_tensor(2, shape);
    if (!tensor) {
        fclose(file);
        return NULL;
    }

    uint8_t *buffer = (uint8_t *)malloc(num_images * num_rows * num_cols);
    if (!buffer) {
        printf("Error: Failed to allocate temporary buffer!\n");
        free_tensor(tensor);
        fclose(file);
        return NULL;
    }

    if (fread(buffer, sizeof(uint8_t), num_images * num_rows * num_cols, file) != num_images * num_rows * num_cols) {
        printf("Error: Incomplete image data in file!\n");
        free(buffer);
        free_tensor(tensor);
        fclose(file);
        return NULL;
    }

    for (size_t i = 0; i < tensor->size; i++) {
        tensor->data[i] = (float) buffer[i] / 255.0f;
    }

    free(buffer);
    fclose(file);
    return tensor;
}

Tensor* load_mnist_labels(const char* filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Could not open file: %s\n", filename);
        return NULL;
    }

    uint32_t magic, num_labels;
    if (fread(&magic, sizeof(uint32_t), 1, file) != 1 ||
        fread(&num_labels, sizeof(uint32_t), 1, file) != 1) {
        printf("Error: Failed to read header from labels!\n");
        fclose(file);
        return NULL;
    }

    magic = swap_endian(magic);
    num_labels = swap_endian(num_labels);

    if (magic != 2049) {
        printf("Invalid magic number in MNIST label file: %u\n", magic);
        fclose(file);
        return NULL;
    }

    int shape[] = {num_labels, 10};
    Tensor *tensor = create_tensor(2, shape);
    if (!tensor) {
        fclose(file);
        return NULL;
    }

    for (size_t i = 0; i < tensor->size; i++) {
        tensor->data[i] = 0.0f;
    }

    uint8_t *buffer = (uint8_t *)malloc(num_labels);
    if (!buffer) {
        printf("Error: Failed to allocate label buffer!\n");
        free_tensor(tensor);
        fclose(file);
        return NULL;
    }

    if (fread(buffer, sizeof(uint8_t), num_labels, file) != num_labels) {
        printf("Error: Incomplete label data in file!\n");
        free(buffer);
        free_tensor(tensor);
        fclose(file);
        return NULL;
    }

    for (size_t i = 0; i < num_labels; i++) {
        tensor->data[i*10 + buffer[i]] = 1.0f;
    }

    free(buffer);
    fclose(file);
    return tensor;
}