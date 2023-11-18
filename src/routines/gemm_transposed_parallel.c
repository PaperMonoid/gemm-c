#include <stdlib.h>


float* gemm_transposed_parallel(float* a, int n, int m, float* b, int p) {
  if (n < 1 || m < 1 || p < 1)
    return NULL;

  float* c = malloc(sizeof(float) * n * p);
  if (c == NULL) {
    return NULL;
  }

  float* b_t = malloc(sizeof(float) * m * p);
  #pragma omp parallel for
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < p; j++) {
      b_t[i * p + j] = b[j * m + i];
    }
  }

  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      c[i * p + j] = 0.0f;
    }
  }

  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      for (int k = 0; k < m; k++) {
	c[i * p + j] += a[i * m + k] * b[k * p + j];
      }
    }
  }

  free(b_t);

  return c;
}
