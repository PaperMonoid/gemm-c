#include <stdlib.h>

#define BLOCK_SIZE 16


float* gemm_block(float* a, int n, int m, float* b, int p) {
    if (n < 1 || m < 1 || p < 1)
        return NULL;

    float* c = malloc(sizeof(float) * n * p);
    if (c == NULL) {
        return NULL;
    }

    for (int i = 0; i < n * p; i++) {
      c[i] = 0.0f;
    }

    for (int i0 = 0; i0 < n; i0 += BLOCK_SIZE) {
        for (int j0 = 0; j0 < p; j0 += BLOCK_SIZE) {
            for (int k0 = 0; k0 < m; k0 += BLOCK_SIZE) {
                for (int i = i0; i < i0 + BLOCK_SIZE && i < n; i++) {
                    for (int j = j0; j < j0 + BLOCK_SIZE && j < p; j++) {
                        float sum = c[i * p + j];
                        for (int k = k0; k < k0 + BLOCK_SIZE && k < m; k++) {
                            sum += a[i * m + k] * b[k * p + j];
                        }
                        c[i * p + j] = sum;
                    }
                }
            }
        }
    }

    return c;
}
