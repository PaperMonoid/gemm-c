#include <stdlib.h>
#include "../matrix.h"


Matrix *gemm_basic(Matrix *A, Matrix *B) {
  float *a = A->data;
  float *b = B->data;
  int n = A->n;
  int m = A->m;
  int p = B->m;

  if (n < 1 || m < 1 || p < 1)
    return NULL;

  float* c = malloc(sizeof(float) * n * p);
  if (c == NULL) {
    return NULL;
  }

  for (int i = 0; i < n * p; i++) {
    c[i] = 0.0f;
  }

  for (int i = 0; i < n; i++) {
    for (int k = 0; k < m; k++) {
      for (int j = 0; j < p; j++) {
	c[i * p + j] += a[i * m + k] * b[k * p + j];
      }
    }
  }

  return new_matrix(c, n, p);
}
