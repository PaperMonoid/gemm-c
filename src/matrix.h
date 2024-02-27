#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
  float* data;
  int n;
  int m;
} Matrix;

Matrix *new_matrix(float* data, int n, int m);

void free_matrix(Matrix *A);

#endif
