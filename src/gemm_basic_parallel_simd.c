#include <immintrin.h>
#include <stdlib.h>


float* gemm_basic_parallel_simd(float* a, int n, int m, float* b, int p) {
  if (n < 1 || m < 1 || p < 1)
    return NULL;

  float* c = malloc(sizeof(float) * n * p);
  if (c == NULL) {
    return NULL;
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
      int k = 0;
      __m256 sum = _mm256_setzero_ps();

      // process 8 elements at a time
      for (k = 0; k <= m - 8; k += 8) {
	__m256 a_vec = _mm256_loadu_ps(a + i * m + k);
	__m256 b_vec = _mm256_loadu_ps(b + k * p + j);
	sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
      }

      // handle remaining elements
      float tail_elements[8] = {0};
      int tail_count = m - k;

      if (tail_count > 0) {
	for (int t = 0; t < tail_count; ++t) {
	  tail_elements[t] = a[i * m + k + t];
	}
	__m256 a_vec_tail = _mm256_loadu_ps(tail_elements);
	__m256 b_vec_tail = _mm256_loadu_ps(b + k * p + j);
	sum = _mm256_fmadd_ps(a_vec_tail, b_vec_tail, sum);
      }

      // add remaining elements
      float temp_sum[8];
      _mm256_storeu_ps(temp_sum, sum);
      float final_sum = 0;
      for (int x = 0; x < 8; ++x) {
	final_sum += temp_sum[x];
      }

      c[i * p + j] = final_sum;
    }
  }

  return c;
}
