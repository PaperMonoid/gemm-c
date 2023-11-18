#include <stdlib.h>


float* gemm_basic(float* a, int n, int m, float* b, int p) {
  if (n < 1 || m < 1 || p < 1)
    return NULL;

  float* c = malloc(sizeof(float) * n * p);
  if (c == NULL) {
    return NULL;
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      c[i * p + j] = 0.0f;
    }
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      for (int k = 0; k < m; k++) {
	c[i * p + j] += a[i * m + k] * b[k * p + j];
      }
    }
  }

  return c;
}
