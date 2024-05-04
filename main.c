#include "stdio.h"
#include <stdlib.h>
#include <stdint.h>

void save_matrix(uint32_t *matrix, int32_t n_rows, int32_t n_cols, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    // write the dimensions of the matrix to the file (to be able to reconstruct it later)
    fwrite(&n_rows, sizeof(int32_t), 1, file);
    fwrite(&n_cols, sizeof(int32_t), 1, file);

    fwrite(matrix, sizeof(uint32_t), n_rows * n_cols, file);

    fclose(file);
}

void naive_mandelbrot(uint32_t *n_iterations,
                      int lower_rational,
                      int upper_rational,
                      int lower_irrational,
                      int upper_irrational) {

}

void parallelized_mandelbrot(uint32_t *n_iterations,
                             int lower_rational,
                             int upper_rational,
                             int lower_irrational,
                             int upper_irrational) {

}

// Arguments:
// [0]: (char[]) name of the program
// [1]: (int) lower boundary for rational part
// [2]: (int) upper boundary for rational part
// [3]: (int) lower boundary for irrational part
// [4]: (int) upper boundary for irrational part
int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("Usage: %s <lower_rational> <upper_rational> <lower_irrational> <upper_irrational>\n", argv[0]);
        return 1;
    }

    int lower_rational, upper_rational, lower_irrational, upper_irrational;

    char *endptr; // pointer to the character that ends the conversion
    lower_rational = (int) strtol(argv[1], &endptr, 10);
    if (*endptr != '\0') {
        printf("Invalid lower rational boundary: %s\n", argv[1]);
        return 1;
    }

    upper_rational = (int) strtol(argv[2], &endptr, 10);
    if (*endptr != '\0') {
        printf("Invalid upper rational boundary: %s\n", argv[2]);
        return 1;
    }

    lower_irrational = (int) strtol(argv[3], &endptr, 10);
    if (*endptr != '\0') {
        printf("Invalid lower irrational boundary: %s\n", argv[3]);
        return 1;
    }

    upper_irrational = (int) strtol(argv[4], &endptr, 10);
    if (*endptr != '\0') {
        printf("Invalid upper irrational boundary: %s\n", argv[4]);
        return 1;
    }

    printf("Lower boundary for rational part: %d\n", lower_rational);
    printf("Upper boundary for rational part: %d\n", upper_rational);
    printf("Lower boundary for irrational part: %d\n", lower_irrational);
    printf("Upper boundary for irrational part: %d\n", upper_irrational);

    // we save these values to the file, so we use int32_t to ensure their size
    int32_t n_rows = upper_rational - lower_rational;
    int32_t n_cols = upper_irrational - lower_irrational;

    // allocate the iteration matrix on the stack
    // we will save this matrix to binary file, so we use uint32_t to ensure the size
    uint32_t n_iterations[n_rows * n_cols];

    naive_mandelbrot(n_iterations, lower_rational, upper_rational, lower_irrational, upper_irrational);

    save_matrix(n_iterations, n_rows, n_cols, "result.bin");

    return 0;
}
