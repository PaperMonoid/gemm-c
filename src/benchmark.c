#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"
#include "gemm_basic.h"
#include "gemm_basic_parallel.h"
#include "gemm_basic_parallel_simd.h"
#include "gemm_transposed.h"
#include "gemm_transposed_parallel.h"
#include "gemm_transposed_parallel_simd.h"


#define MODE_BASIC 1
#define MODE_BASIC_PARALLEL 2
#define MODE_BASIC_PARALLEL_SIMD 3
#define MODE_TRANSPOSED 4
#define MODE_TRANSPOSED_PARALLEL 5
#define MODE_TRANSPOSED_PARALLEL_SIMD 6


void generate_random_floats(float *data, int size, float min, float max) {
    srand((unsigned int)time(NULL));
    for (int i = 0; i < size; i++) {
        float scale = rand() / (float) RAND_MAX;
        data[i] = min + scale * (max - min);
    }
}


Matrix *new_random_matrix(int n, int m) {
  int size = n * m;
  float* data = malloc(sizeof(float) * size);
  generate_random_floats(data, size, 0.0, 1.0);
  return new_matrix(data, n, m);
}


void benchmark(FILE *file, int mode) {
  int sizes[2] = {
    //4, 16, 64, 256, 1024, 2058, 3000
    2058, 3000
  };
  char* modes[9] = {
    "basic", "basic_parallel", "basic_parallel_simd",
    "transposed", "transposed_parallel", "transposed_parallel_simd"
  };
  int n, m, p;
  Matrix *A, *B, *C;
  int executions = 0;
  time_t start_time, current_time;
  double elapsed = 0.0;

  printf("Benchmarking Mode: %s", modes[mode - 1]);
  fflush(stdout);
  for (int i = 0; i < 2; i++) {
    n = sizes[i];
    m = sizes[i];
    p = sizes[i];
    executions = 0;
    elapsed = 0.0;
    time(&start_time);
    while (elapsed <= 10.0) {
      A = new_random_matrix(n, m);
      B = new_random_matrix(m, p);
      switch(mode) {
      case MODE_BASIC:
	C = gemm_basic(A, B);
	break;
      case MODE_BASIC_PARALLEL:
	C = gemm_basic_parallel(A, B);
	break;
      case MODE_BASIC_PARALLEL_SIMD:
	C = gemm_basic_parallel_simd(A, B);
	break;
      case MODE_TRANSPOSED:
	C = gemm_transposed(A, B);
	break;
      case MODE_TRANSPOSED_PARALLEL:
	C = gemm_transposed_parallel(A, B);
	break;
      case MODE_TRANSPOSED_PARALLEL_SIMD:
	C = gemm_transposed_parallel_simd(A, B);
	break;
      }
      free_matrix(A);
      free_matrix(B);
      free_matrix(C);
      executions++;
      time(&current_time);
      elapsed = difftime(current_time, start_time);
    }
    double executions_per_second = (double) executions / elapsed;
    fprintf(file, "%s,%d,%d,%f\n", modes[mode - 1], n, m, executions_per_second);

    printf(" %dx%d", n, m);
    fflush(stdout);
  }
  printf("\n");
}
