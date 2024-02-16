#ifndef MATRIX_H
#define MATRIX_H

struct Matrix {
  float* data;
  int n;
  int m;
};

struct Matrix *new_matrix(float* data, int n, int m);

void free_matrix(struct Matrix *A);

#endif
