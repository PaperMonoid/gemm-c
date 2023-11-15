#include <stdlib.h>
#include <time.h>


void generate_random_floats(float *array, int size, float min, float max) {
    srand((unsigned int)time(NULL));
    for (int i = 0; i < size; i++) {
        float scale = rand() / (float) RAND_MAX;
        array[i] = min + scale * (max - min);
    }
}


void benchmark() {

}
