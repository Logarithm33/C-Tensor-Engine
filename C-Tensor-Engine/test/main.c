#include "data.h"

int main() {
    Tensor *X_train = load_mnist_images("data/train-images-idx3-ubyte");
    Tensor *Y_train = load_mnist_labels("data/train-labels-idx1-ubyte");

    if (X_train && Y_train) {
        printf("Loaded Training Images Shape: [%d, %d]\n", X_train->shape[0], X_train->shape[1]);
        printf("Loaded Training Labels Shape: [%d, %d]\n", Y_train->shape[0], Y_train->shape[1]);
    }
}