#include <stdlib.h>

#include "../headers/matrix.h"


Matrix *new_matrix(float* data, int n, int m) {
  Matrix *A = malloc(sizeof(Matrix));
  A->data = data;
  A->n = n;
  A->m = m;
  return A;
}

void free_matrix(Matrix *A) {
  free(A->data);
  free(A);
}
