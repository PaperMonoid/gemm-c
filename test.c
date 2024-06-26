#include <stdio.h>
#include <stdio.h>


#include "headers/matrix.h"
#include "headers/gemm_basic.h"
#include "headers/gemm_basic_parallel.h"
#include "headers/gemm_basic_parallel_simd.h"
#include "headers/gemm_transposed.h"
#include "headers/gemm_transposed_parallel.h"
#include "headers/gemm_transposed_parallel_simd.h"



void test() {
  int n = 2;
  int m = 3;
  float a[2][3] = {
    {1.0, 2.0, 3.0},
    {5.0, 6.0, 7.0},
  };
  Matrix A = {
    .data = &a[0][0],
    .n = n,
    .m = m
  };

  int p = 4;
  float b[3][4] = {
    {1.0, 2.0, 3.0, 4.0},
    {5.0, 6.0, 7.0, 8.0},
    {9.0, 10.0, 11.0, 12.0}
  };
  Matrix B = {
    .data = &b[0][0],
    .n = m,
    .m = p
  };

  Matrix *C;
  float* c;

  printf("Gemm Basic: \n\n");

  C = gemm_basic(&A, &B);
  if (C) {
    c = C->data;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
	printf("%f, ", c[i * p + j]);
      }
      printf("\n");
    }
    free_matrix(C);
  }

  printf("\n--------------------------------\n\n");
  printf("Gemm Basic Parallel: \n\n");

  C = gemm_basic_parallel(&A, &B);
  if (C) {
    c = C->data;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
	printf("%f, ", c[i * p + j]);
      }
      printf("\n");
    }
    free_matrix(C);
  }

  printf("\n--------------------------------\n\n");
  printf("Gemm Basic Parallel Simd: \n\n");

  C = gemm_basic_parallel_simd(&A, &B);
  if (C) {
    c = C->data;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
	printf("%f, ", c[i * p + j]);
      }
      printf("\n");
    }
    free_matrix(C);
  }

  printf("\n--------------------------------\n\n");
  printf("Gemm Transposed: \n\n");

  C = gemm_transposed(&A, &B);
  if (C) {
    c = C->data;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
	printf("%f, ", c[i * p + j]);
      }
      printf("\n");
    }
    free_matrix(C);
  }

  printf("\n--------------------------------\n\n");
  printf("Gemm Transposed Parallel: \n\n");

  C = gemm_transposed_parallel(&A, &B);
  if (C) {
    c = C->data;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
	printf("%f, ", c[i * p + j]);
      }
      printf("\n");
    }
    free_matrix(C);
  }

  printf("\n--------------------------------\n\n");
  printf("Gemm Transposed Parallel Simd: \n\n");

  C = gemm_transposed_parallel_simd(&A, &B);
  if (C) {
    c = C->data;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < p; j++) {
	printf("%f, ", c[i * p + j]);
      }
      printf("\n");
    }
    free_matrix(C);
  }

  printf("\n--------------------------------\n\n");

}
