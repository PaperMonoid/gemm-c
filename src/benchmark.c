#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "gemm_basic.h"
#include "gemm_basic_parallel.h"
#include "gemm_basic_parallel_simd.h"
#include "gemm_transposed.h"
#include "gemm_transposed_parallel.h"
#include "gemm_transposed_parallel_simd.h"
#include "gemm_block_parallel.h"
#include "gemm_block_parallel_simd.h"

#define MODE_BASIC 1
#define MODE_BASIC_PARALLEL 2
#define MODE_BASIC_PARALLEL_SIMD 3


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


void benchmark(int mode) {
  int sizes[7] = {
    4, 16, 64, 256, 1024, 2058, 3000 //, 4096
  };
  int n, m, p;
  float *a, *b, *c;
  int executions = 0;
  time_t start_time, current_time;
  double elapsed = 0.0;

  printf("Starting benchmark...\n");
  switch(mode) {
  case MODE_BASIC:
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
	//c = gemm_basic(a, n, m, b, p);
	//c = gemm_transposed(a, n, m, b, p);
	//c = gemm_basic_parallel_simd(a, n, m, b, p)
	//c = gemm_transposed_parallel_simd(a, n, m, b, p);
 	c = gemm_block_parallel_simd(a, n, m, b, p);
	free(a);
	free(b);
	free(c);
	executions++;
	time(&current_time);
	elapsed = difftime(current_time, start_time);
      }
      double executions_per_second = (double) executions / elapsed;
      printf("Matrix size %dx%d | Executions per second: %f\n", n, m, executions_per_second);
    }
    break;

  case MODE_BASIC_PARALLEL:
    break;

  case MODE_BASIC_PARALLEL_SIMD:
    break;

  }

  printf("Finished benchmark!\n");
}
