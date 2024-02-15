#include <stdlib.h>


float* gemm_transposed(float* a, int n, int m, float* b, int p) {
  if (n < 1 || m < 1 || p < 1)
    return NULL;

  float* c = malloc(sizeof(float) * n * p);
  if (c == NULL) {
    return NULL;
  }

  float* b_t = malloc(sizeof(float) * m * p);
  if (b_t == NULL) {
    return NULL;
  }

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < p; j++) {
      b_t[j * m + i] = b[i * p + j];
    }
  }

  for (int i = 0; i < n * p; i++) {
    c[i] = 0.0f;
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      for (int k = 0; k < m; k++) {
      	c[i * p + j] += a[i * m + k] * b_t[j * m + k];
      }
    }
  }

  free(b_t);

  return c;
}
