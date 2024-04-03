#include <stdio.h>
#include <time.h>

#include "headers/matrix.h"
#include "benchmark.c"


int main() {
  srand((unsigned int)time(NULL));

  FILE *file;
  file = fopen("results.txt", "w");
  benchmark(file, 2);
  benchmark(file, 3);
  benchmark(file, 4);
  benchmark(file, 5);
  benchmark(file, 6);

  fclose(file);
  return 0;
}
