#include "stdio.h"
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

struct OpStats {
    uint64_t n_op;
    uint64_t n_ns;
};

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

// Computes the number of algebraic operations that was needed to compute
// mandelbrot set. Does not include helper operations like incrementing the
// index variables.
uint64_t compute_ops(uint32_t const *n_iterations, int height, int width) {
    uint64_t ops = 0;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint32_t iterations = n_iterations[y * width + x];

            // magnitude calculation and check
            ops += 3 * (iterations + 1);

            // calculation of the next element in the sequence
            ops += 5 * iterations;
        }
    }
    return ops;
}

uint64_t get_ns_diff(struct timespec start, struct timespec end) {
    int64_t sec_passed = end.tv_sec - start.tv_sec;
    int64_t ns_passed = end.tv_nsec - start.tv_nsec;  // could be negative!

    if (ns_passed < 0) {
        sec_passed -= 1;
        ns_passed += 1000000000; // 1 billion nanoseconds in a second
    }

    return (sec_passed * 1000000000LL) + ns_passed;
}

struct OpStats naive_mandelbrot(uint32_t *n_iterations,
                                double lower_real,
                                double upper_real,
                                double lower_imaginary,
                                double upper_imaginary,
                                int height,
                                int width,
                                int max_iterations) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // mandelbrot calculation

    double re_factor = (upper_real - lower_real) / (width - 1);
    double im_factor = (upper_imaginary - lower_imaginary) / (height - 1);

    for (int y = 0; y < height; ++y) {
        double c_im = upper_imaginary - y * im_factor;

        for (int x = 0; x < width; ++x) {
            double c_re = lower_real + x * re_factor;

            // z_0 = c
            double z_re = c_re, z_im = c_im;

            int n;

            for (n = 0; n < max_iterations; ++n) {
                // compute the magnitude of z_i
                double z_re2 = z_re * z_re, z_im2 = z_im * z_im;

                // we consider number to reach the bound once |z_i| > 2
                if (z_re2 + z_im2 > 4) {
                    break;
                }

                // z_{i + 1} = z_i ** 2 + c = (re(z_i) ** 2 - im(z_i) ** 2, 2 * re(z_i) * im(z_i))
                z_im = 2 * z_re * z_im + c_im;
                z_re = z_re2 - z_im2 + c_re;
            }

            n_iterations[y * width + x] = n;
        }
    }


    clock_gettime(CLOCK_MONOTONIC, &end);

    struct OpStats res = {compute_ops(n_iterations, height, width), get_ns_diff(start, end)};
    return res;
}

struct OpStats optimized_mandelbrot(uint32_t *n_iterations,
                                    double lower_real,
                                    double upper_real,
                                    double lower_imaginary,
                                    double upper_imaginary,
                                    int height,
                                    int width,
                                    int max_iterations) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // mandelbrot calculation

    clock_gettime(CLOCK_MONOTONIC, &end);

    struct OpStats res = {compute_ops(n_iterations, height, width), get_ns_diff(start, end)};
    return res;
}

// Arguments:
// [0]: (char[]) name of the program
// [1]: (double) lower boundary for real part
// [2]: (double) upper boundary for real part
// [3]: (double) lower boundary for imaginary part
// [4]: (double) upper boundary for imaginary part
// [5]: (int) height
// [6]: (int) width
// [7]: (int) maximum number of iterations
int main(int argc, char *argv[]) {
    // Check if the correct number of arguments is provided
    if (argc != 8) {
        printf("Usage: %s <lower_rational> <upper_rational> <lower_irrational> <upper_irrational> <height> <width> <max_iterations>\n",
               argv[0]);
        return 1;
    }

    double lower_real, upper_real, lower_imaginary, upper_imaginary;
    int height, width, max_iterations;

    // Convert command-line arguments to appropriate types
    char *endptr;

    // Rational boundaries
    lower_real = strtod(argv[1], &endptr);
    if (*endptr != '\0') {
        printf("Invalid lower rational boundary: %s\n", argv[1]);
        return 1;
    }

    upper_real = strtod(argv[2], &endptr);
    if (*endptr != '\0') {
        printf("Invalid upper rational boundary: %s\n", argv[2]);
        return 1;
    }

    // Irrational boundaries
    lower_imaginary = strtod(argv[3], &endptr);
    if (*endptr != '\0') {
        printf("Invalid lower irrational boundary: %s\n", argv[3]);
        return 1;
    }

    upper_imaginary = strtod(argv[4], &endptr);
    if (*endptr != '\0') {
        printf("Invalid upper irrational boundary: %s\n", argv[4]);
        return 1;
    }

    // Image dimensions
    height = (int) strtol(argv[5], &endptr, 10);
    if (*endptr != '\0') {
        printf("Invalid height value: %s\n", argv[5]);
        return 1;
    }

    width = (int) strtol(argv[6], &endptr, 10);
    if (*endptr != '\0') {
        printf("Invalid width value: %s\n", argv[6]);
        return 1;
    }

    // Maximum iterations
    max_iterations = (int) strtol(argv[7], &endptr, 10);
    if (*endptr != '\0') {
        printf("Invalid maximum number of iterations: %s\n", argv[7]);
        return 1;
    }

    // Print the parsed values
    printf("Lower boundary for rational part: %f\n", lower_real);
    printf("Upper boundary for rational part: %f\n", upper_real);
    printf("Lower boundary for irrational part: %f\n", lower_imaginary);
    printf("Upper boundary for irrational part: %f\n", upper_imaginary);
    printf("Height: %d\n", height);
    printf("Width: %d\n", width);
    printf("Maximum number of iterations: %d\n", max_iterations);

    // allocate the iteration matrix on the stack
    // we will save this matrix to binary file, so we use uint32_t to ensure the size
    uint32_t *n_iterations = (uint32_t *) malloc(height * width * sizeof(uint32_t));
    if (!n_iterations) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    struct OpStats result = naive_mandelbrot(n_iterations, lower_real, upper_real, lower_imaginary,
                                             upper_imaginary, height, width, max_iterations);

    printf("Operations: %llu, Time: %llu ns, ns/op: %Lf", result.n_op, result.n_ns,
           (long double) result.n_ns / (long double) result.n_op);

    save_matrix(n_iterations, height, width, "result.bin");

    return 0;
}
