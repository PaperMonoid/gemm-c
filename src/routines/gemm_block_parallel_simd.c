#include <immintrin.h>
#include <stdlib.h>

 // Ensure that BLOCK_SIZE is a multiple of 8
#define BLOCK_SIZE 64


float* gemm_block_parallel_simd(float* a, int n, int m, float* b, int p) {
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

    int i_max, j_max, k_max;

    #pragma omp parallel for
    for (int i0 = 0; i0 < n; i0 += BLOCK_SIZE) {
      i_max = (i0 + BLOCK_SIZE < n) ? i0 + BLOCK_SIZE : n;
      for (int j0 = 0; j0 < p; j0 += BLOCK_SIZE) {
	j_max = (j0 + BLOCK_SIZE < p) ? j0 + BLOCK_SIZE : p;
        for (int k0 = 0; k0 < m; k0 += BLOCK_SIZE) {
	  k_max = (k0 + BLOCK_SIZE < m) ? k0 + BLOCK_SIZE : m;
	  for (int i = i0; i < i_max; i++) {
	    for (int j = j0; j < j_max; j++) {
	      __m256 sum_vec = _mm256_setzero_ps();
	      for (int k = k0; k < k_max; k += 8) {
		__m256 a_vec = _mm256_loadu_ps(a + i * m + k);
		float b_elements[8] = {0};
		int t;
		for (t = 0; t < 8 && (k + t) < m; ++t) {
		  b_elements[t] = b[(k + t) * p + j];
		}
		__m256 b_vec = _mm256_loadu_ps(b_elements);
		sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
	      }

	      float temp_sum[8];
	      _mm256_storeu_ps(temp_sum, sum_vec);
	      float sum = 0.0f;
	      for (int x = 0; x < 8; ++x) {
		sum += temp_sum[x];
	      }
	      c[i * p + j] += sum;
	    }
	  }
        }
      }
    }

    return c;
}
