#include <stdlib.h>

#include "../headers/matrix.h"


Matrix *gemm_basic_parallel(Matrix *first, Matrix *second) {
  float *a = first->data;
  float *b = second->data;
  int n = first->n;
  int m = first->m;
  int p = second->m;

  if (n < 1 || m < 1 || p < 1)
    return NULL;

  float* c = malloc(sizeof(float) * n * p);
  if (c == NULL) {
    return NULL;
  }

#pragma omp parallel for
  for (int i = 0; i < n * p; i++) {
    c[i] = 0.0f;
  }

#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < m; k++) {
      for (int j = 0; j < p; j++) {
	c[i * p + j] += a[i * m + k] * b[k * p + j];
      }
    }
  }

  return new_matrix(c, n, p);
}
