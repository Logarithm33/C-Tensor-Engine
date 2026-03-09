#include "autograd.h"

static void build_topo (Tensor *t,Tensor **topo_list, int *topo_size) {
    if (!t || t->_visited) return;

    t->_visited = true;
    for (int i = 0; i < t->_prev_count; i++) {
        build_topo(t->_prev[i], topo_list, topo_size);
    }
    topo_list[(*topo_size)++] = t;
}

void tensor_backward (Tensor *loss) {
    if (!loss) return;

    Tensor **topo_list = (Tensor **)malloc(sizeof(Tensor *) * 1024);
    int topo_size = 0;

    build_topo(loss, topo_list, &topo_size);

    if (!loss->grad) {
        loss->grad = (float *)malloc(sizeof(float) * loss->size);
    }
    for (size_t i = 0; i < loss->size; i++) {
        loss->grad[i] = 1.0f;
    }

    for (int i = topo_size - 1; i >= 0; i--) {
        if (topo_list[i]->_backward) {
            topo_list[i]->_backward(topo_list[i]);
        }
    }

    free(topo_list);
}