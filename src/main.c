#include <stdlib.h>
#include <stdio.h>
#include "gemm_basic.h"
#include "gemm_basic_parallel.h"
#include "gemm_basic_parallel_simd.h"
#include "gemm_transposed.h"
#include "gemm_transposed_parallel.h"
#include "gemm_transposed_parallel_simd.h"
#include "gemm_block.h"
#include "gemm_block_parallel.h"
#include "gemm_block_parallel_simd.h"
#include "benchmark.c"


void test() {
  int n = 2;
  int m = 3;
  float a[2][3] = {
    {1.0, 2.0, 3.0},
      {5.0, 6.0, 7.0},
  };

  int p = 4;
  float b[3][4] = {
    {1.0, 2.0, 3.0, 4.0},
    {5.0, 6.0, 7.0, 8.0},
    {9.0, 10.0, 11.0, 12.0}
  };

  float* c;

  printf("Gemm Basic: \n\n");

  c = gemm_basic(&a[0][0], n, m, &b[0][0], p);
  if (c) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
	printf("%f, ", c[i * p + j]);
      }
      printf("\n");
    }
    free(c);
  }

  printf("\n--------------------------------\n\n");
  printf("Gemm Basic Parallel: \n\n");

  c = gemm_basic_parallel(&a[0][0], n, m, &b[0][0], p);
  if (c) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
	printf("%f, ", c[i * p + j]);
      }
      printf("\n");
    }
    free(c);
  }

  printf("\n--------------------------------\n\n");
  printf("Gemm Basic Parallel Simd: \n\n");

  c = gemm_basic_parallel_simd(&a[0][0], n, m, &b[0][0], p);
  if (c) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
	printf("%f, ", c[i * p + j]);
      }
      printf("\n");
    }
    free(c);
  }

  printf("\n--------------------------------\n\n");
  printf("Gemm Transposed: \n\n");

  c = gemm_transposed(&a[0][0], n, m, &b[0][0], p);
  if (c) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
	printf("%f, ", c[i * p + j]);
      }
      printf("\n");
    }
    free(c);
  }

  printf("\n--------------------------------\n\n");
  printf("Gemm Transposed Parallel: \n\n");

  c = gemm_transposed_parallel(&a[0][0], n, m, &b[0][0], p);
  if (c) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
	printf("%f, ", c[i * p + j]);
      }
      printf("\n");
    }
    free(c);
  }

  printf("\n--------------------------------\n\n");
  printf("Gemm Transposed Parallel Simd: \n\n");

  c = gemm_transposed_parallel_simd(&a[0][0], n, m, &b[0][0], p);
  if (c) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
	printf("%f, ", c[i * p + j]);
      }
      printf("\n");
    }
    free(c);
  }

  printf("\n--------------------------------\n\n");
  printf("Gemm Block: \n\n");

  c = gemm_block(&a[0][0], n, m, &b[0][0], p);
  if (c) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
	printf("%f, ", c[i * p + j]);
      }
      printf("\n");
    }
    free(c);
  }

  printf("\n--------------------------------\n\n");
  printf("Gemm Block Parallel: \n\n");

  c = gemm_block_parallel(&a[0][0], n, m, &b[0][0], p);
  if (c) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
	printf("%f, ", c[i * p + j]);
      }
      printf("\n");
    }
    free(c);
  }

  printf("\n--------------------------------\n\n");
  printf("Gemm Block Parallel Simd: \n\n");

  c = gemm_block_parallel_simd(&a[0][0], n, m, &b[0][0], p);
  if (c) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
	printf("%f, ", c[i * p + j]);
      }
      printf("\n");
    }
    free(c);
  }

  printf("\n--------------------------------\n\n");
}


int main() {
  test();

  FILE *file;
  file = fopen("results.txt", "w");
  fprintf(file, "");
  fclose(file);
  file = fopen("results.txt", "a");

  benchmark(file, 1);
  benchmark(file, 2);
  benchmark(file, 3);
  benchmark(file, 4);
  benchmark(file, 5);
  benchmark(file, 6);
  benchmark(file, 7);
  benchmark(file, 8);
  benchmark(file, 9);

  fclose(file);
  return 0;
}
