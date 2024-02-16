#include <stdlib.h>
#include "matrix.h"


struct Matrix *new_matrix(float* data, int n, int m) {
  struct Matrix *A = malloc(sizeof(struct Matrix));
  A->data = data;
  A->n = n;
  A->m = m;
  return A;
}

void free_matrix(struct Matrix *A) {
  free(A->data);
  free(A);
}
