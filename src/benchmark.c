#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "gemm_basic.h"
#include "gemm_basic_parallel.h"
#include "gemm_basic_parallel_simd.h"
#include "gemm_transposed.h"
#include "gemm_transposed_parallel.h"
#include "gemm_transposed_parallel_simd.h"
#include "gemm_block.h"
#include "gemm_block_parallel.h"
#include "gemm_block_parallel_simd.h"


#define MODE_BASIC 1
#define MODE_BASIC_PARALLEL 2
#define MODE_BASIC_PARALLEL_SIMD 3
#define MODE_TRANSPOSED 4
#define MODE_TRANSPOSED_PARALLEL 5
#define MODE_TRANSPOSED_PARALLEL_SIMD 6
#define MODE_BLOCK 7
#define MODE_BLOCK_PARALLEL 8
#define MODE_BLOCK_PARALLEL_SIMD 9


void generate_random_floats(float *array, int size, float min, float max) {
    srand((unsigned int)time(NULL));
    for (int i = 0; i < size; i++) {
        float scale = rand() / (float) RAND_MAX;
        array[i] = min + scale * (max - min);
    }
}


float* new_random_matrix(int n, int m) {
  int size = n * m;
  float* array = malloc(sizeof(float) * size);
  generate_random_floats(array, size, 0.0, 1.0);
  return array;
}


void benchmark(FILE *file, int mode) {
  int sizes[7] = {
    4, 16, 64, 256, 1024, 2058, 3000
  };
  char* modes[9] = {
    "basic", "basic_parallel", "basic_parallel_simd",
    "transposed", "transposed_parallel", "transposed_parallel_simd",
    "block", "block_parallel", "block_parallel_simd"
  };
  int n, m, p;
  float *a, *b, *c;
  int executions = 0;
  time_t start_time, current_time;
  double elapsed = 0.0;

  printf("Benchmarking Mode: %s", modes[mode - 1]);
  fflush(stdout);
  for (int i = 0; i < 7; i++) {
    n = sizes[i];
    m = sizes[i];
    p = sizes[i];
    executions = 0;
    elapsed = 0.0;
    time(&start_time);
    while (elapsed <= 10.0) {
      a = new_random_matrix(n, m);
      b = new_random_matrix(m, p);
      switch(mode) {
      case MODE_BASIC:
	c = gemm_basic(a, n, m, b, p);
	break;
      case MODE_BASIC_PARALLEL:
	c = gemm_basic_parallel(a, n, m, b, p);
	break;
      case MODE_BASIC_PARALLEL_SIMD:
	c = gemm_basic_parallel_simd(a, n, m, b, p);
	break;
      case MODE_TRANSPOSED:
	c = gemm_transposed(a, n, m, b, p);
	break;
      case MODE_TRANSPOSED_PARALLEL:
	c = gemm_transposed_parallel(a, n, m, b, p);
	break;
      case MODE_TRANSPOSED_PARALLEL_SIMD:
	c = gemm_transposed_parallel_simd(a, n, m, b, p);
	break;
      case MODE_BLOCK:
	c = gemm_block(a, n, m, b, p);
	break;
      case MODE_BLOCK_PARALLEL:
	c = gemm_block(a, n, m, b, p);
	break;
      case MODE_BLOCK_PARALLEL_SIMD:
	c = gemm_block(a, n, m, b, p);
	break;
      }
      free(a);
      free(b);
      free(c);
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
