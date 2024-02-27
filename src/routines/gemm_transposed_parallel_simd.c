#include <immintrin.h>
#include <stdlib.h>
#include "../matrix.h"


Matrix *gemm_transposed_parallel_simd(Matrix *A,
				      Matrix *B) {
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

  float* b_t = malloc(sizeof(float) * m * p);
  if (b_t == NULL) {
    return NULL;
  }

#pragma omp parallel for
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < p; j++) {
      b_t[j * m + i] = b[i * p + j];
    }
  }

#pragma omp parallel for
  for (int i = 0; i < n * p; i++) {
    c[i] = 0.0f;
  }

#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      __m256 sum = _mm256_setzero_ps();

      // process 8 elements at a time
      int k = 0;
      for (; k <= m - 8; k += 8) {
	__m256 a_vec = _mm256_loadu_ps(a + i * m + k);
	float b_elements[8] = {
	  b_t[(j + 0) * m + k],
	  b_t[(j + 1) * m + k],
	  b_t[(j + 2) * m + k],
	  b_t[(j + 3) * m + k],
	  b_t[(j + 4) * m + k],
	  b_t[(j + 5) * m + k],
	  b_t[(j + 6) * m + k],
	  b_t[(j + 7) * m + k]
	};
	__m256 b_vec = _mm256_loadu_ps(b_elements);
	sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
      }

      // handle remaining elements
      float final_sum = 0.0f;
      for (; k < m; ++k) {
	final_sum += a[i * m + k] * b_t[j * m + k];
      }

      // sum up the values
      float sums[8];
      _mm256_storeu_ps(sums, sum);
      final_sum += sums[0] + sums[1] + sums[2] + sums[3]
	+ sums[4] + sums[5] + sums[6] + sums[7];

      c[i * p + j] = final_sum;
    }
  }

  free(b_t);

  return new_matrix(c, n, p);
}
