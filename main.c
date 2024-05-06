#include "stdio.h"
#include <assert.h>
#include <immintrin.h>
#include <stdalign.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

struct OpStats {
    uint64_t n_op;
    uint64_t n_ns;
};

void* allocate_aligned_memory(size_t alignment, size_t size) {
    void* ptr = NULL;

#if defined(_WIN32) || defined(_WIN64)
// Windows platform using _aligned_malloc
ptr = _aligned_malloc(size, alignment);
if (!ptr) {
fprintf(stderr, "Error: _aligned_malloc failed\n");
}
#elif defined(__APPLE__) || defined(__linux__) || defined(__unix__)
// POSIX platforms using aligned_alloc
    ptr = aligned_alloc(alignment, size); 
    if (!ptr) {
        perror("Error: aligned_alloc failed");
    }
#else
    // Fallback for unsupported platforms (custom or basic alignment handling)
    ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "Error: malloc failed\n");
    }
#endif

return ptr;
}

void free_aligned_memory(void *ptr) {
#if defined(_WIN32) || defined(_WIN64)
    _aligned_free(ptr);
#elif defined(__APPLE__) || defined(__linux__) || defined(__unix__)
    free(n_iterations);
#else
            free(n_iterations);
#endif
}

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

union u256d {
    __m256d v;
    double d[4];
};

#define PRINT_DV(a)                                                            \
{                                                                              \
    union u256d u = {a};                                                       \
    printf("%f %f %f %f\n", u.d[0], u.d[1], u.d[2], u.d[3]);                   \
}

struct OpStats optimized_mandelbrot(uint32_t *n_iterations, double lower_real,
                                    double upper_real, double lower_imaginary,
                                    double upper_imaginary, int height,
                                    int width, int max_iterations) {
    assert(width % 4 == 0); // lazy
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    const double bound_squared = 4;
    const double re_factor = (upper_real - lower_real) / (width - 1);
    const double im_step =
            height > 1 ? ((upper_imaginary - lower_imaginary) / (height - 1)) : 1;

    const double re_inc4s = 4 * re_factor;
    __m256d re_inc4 = _mm256_broadcast_sd(&re_inc4s);

    __m256d c_im = _mm256_broadcast_sd(&upper_imaginary);
    __m256d c_inc = _mm256_broadcast_sd(&im_step);

    alignas(64) const double re_inc_step_arr[4] = {0, 1 * re_factor,
                                                   2 * re_factor, 3 * re_factor};
    __m256d re_step = _mm256_load_pd(re_inc_step_arr);

    for (int y = 0; y < height; ++y) {
        __m256d c_re = _mm256_add_pd(_mm256_broadcast_sd(&lower_real), re_step);
        for (int x = 0; x < width; x += 4) {
            int n;
            __m256i iters = _mm256_set1_epi64x(0);
            __m256i iter_inc = _mm256_set1_epi64x(1);

            __m256d z_re = c_re;
            __m256d z_im = c_im;
            for (n = 0; n < max_iterations; ++n) {

                __m256d z_re2 = _mm256_mul_pd(z_re, z_re);
                __m256d res_re = _mm256_add_pd(z_re2, c_re);
                res_re = _mm256_fnmadd_pd(z_im, z_im, res_re);
                __m256d res_im = _mm256_mul_pd(z_re, z_im);
                res_im = _mm256_add_pd(res_im, res_im);
                res_im = _mm256_add_pd(res_im, c_im);
                z_re = res_re;
                z_im = res_im;

                __m256d mag = _mm256_fmadd_pd(z_im, z_im, z_re2);

                __m256d lt_res = _mm256_cmp_pd(
                        mag, _mm256_broadcast_sd(&bound_squared),
                        _CMP_LT_OQ); // OQ means no signal if one of the operands is NaN

                // mask that decides which positions in iters get incremented.
                // once a position is > bound, we zero out the corresponding mask bits
                // so it no longer gets incremented
                iter_inc = _mm256_castpd_si256(
                        _mm256_and_pd(_mm256_castsi256_pd(iter_inc), lt_res));

                // only increment the 64 bit iter counters in positions where the
                // comparison was true
                iters = _mm256_add_epi64(iters, iter_inc);

                // test if lt_res is all zeros (i.e, all magnitudes are greater than
                // bound)
                int all_gt = _mm256_testz_si256((__m256i) lt_res, (__m256i) lt_res);
                if (all_gt) {
                    break;
                }
            }
            // get iter counts out of ymm register and store to n_iterations array
            // n_iterations[y * width + x + 0] = _mm256_extract_epi32(iters, 0);
            // n_iterations[y * width + x + 1] = _mm256_extract_epi32(iters, 2);
            // n_iterations[y * width + x + 2] = _mm256_extract_epi32(iters, 4);
            // n_iterations[y * width + x + 3] = _mm256_extract_epi32(iters, 6);

            // extract lower 32 bits of every 64 bits in iters, since we want to store
            // 32 bit int iteration counts. this is a bit annoying since vshufpd only
            // works within 128 bit control lanes in 256bit registers, so we have to
            // extract the 128 bit halves and shuffle within those to then do a single
            // store
            __m128 low = _mm256_extractf128_ps(_mm256_castsi256_ps(iters), 0);
            __m128 high = _mm256_extractf128_ps(_mm256_castsi256_ps(iters), 1);
            __m128 packed_32 = _mm_shuffle_ps(low, high, _MM_SHUFFLE(0, 2, 0, 2));
            _mm_store_ps((float *) (&n_iterations[y * width + x]), packed_32);

            c_re = _mm256_add_pd(c_re, re_inc4);
        }
        c_im = _mm256_sub_pd(c_im, c_inc);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    struct OpStats res = {compute_ops(n_iterations, height, width), get_ns_diff(start, end)};
    return res;
}

void benchmark(const char* filename) {
    const int repeat = 10;
    const int versions[] = {0, 1};
    const int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096};
    const int max_iterss[] = {10, 50, 250};

    FILE* output_file = fopen(filename, "w");
    if (!output_file) {
        fprintf(stderr, "Error: Unable to open output file %s\n", filename);
        return;
    }

    // Write the header to the file
    fprintf(output_file, "version,size,max_iters,time_s\n");

    // Iterate through each combination of version, size, and max_iters
    for (int v = 0; v < sizeof(versions) / sizeof(int); ++v) {
        for (int si = 0; si < sizeof(sizes) / sizeof(int); ++si) {
            const int size = sizes[si];

            // Allocate memory with proper alignment
            uint32_t *n_iterations = (uint32_t *) allocate_aligned_memory(64, size * size * sizeof(uint32_t));

            // Check if memory allocation was successful
            if (!n_iterations) {
                fprintf(stderr, "Error: Memory allocation failed for size %d\n", size);
                continue;
            }

            for (int mi = 0; mi < sizeof(max_iterss) / sizeof(int); ++mi) {
                const int max_iters = max_iterss[mi];
                for (int r = 0; r < repeat; ++r) {
                    struct timespec start, end;
                    clock_gettime(CLOCK_MONOTONIC, &start);

                    // Perform the appropriate version of the Mandelbrot set computation
                    if (v == 0) {
                        naive_mandelbrot((uint32_t *) n_iterations, -2, 1, -2, 2, size, size, max_iters);
                    } else if (v == 1) {
                        optimized_mandelbrot(n_iterations, -2, 1, -2, 2, size, size, max_iters);
                    }

                    clock_gettime(CLOCK_MONOTONIC, &end);
                    unsigned long long time_ns = get_ns_diff(start, end);

                    // Write results to the file
                    fprintf(output_file, "%d,%d,%d,%llu\n", v, size, max_iters, time_ns);
                }
            }

            // Free the allocated memory
            free_aligned_memory(n_iterations);
        }
    }

    fclose(output_file);
}

// Arguments:
// [0]: (char[]) name of the program

// 1) Benchmark with cache warmup
// [1]: (char[]) mandatory "bench" string
// [2]: (char[]) csv file where the results should be written to

// 2) Single run
// [1]: (int) version, 0 for naive, 1 for optimized
// [2]: (double) lower boundary for real part
// [3]: (double) upper boundary for real part
// [4]: (double) lower boundary for imaginary part
// [5]: (double) upper boundary for imaginary part
// [6]: (int) height
// [7]: (int) width
// [8]: (int) maximum number of iterations
int main(int argc, char *argv[]) {
    if (argc == 3 && !strcmp(argv[1], "bench")) {
        benchmark(argv[2]);
        return 0;
    }

    // Check if the correct number of arguments is provided
    if (argc != 9) {
        printf("Usage: %s <version> <lower_rational> <upper_rational> <lower_irrational> <upper_irrational> <height> <width> <max_iterations>\n",
               argv[0]);
        return 1;
    }

    double lower_real, upper_real, lower_imaginary, upper_imaginary;
    int height, width, max_iterations;

    // Convert command-line arguments to appropriate types
    char *endptr;

    unsigned int version = (unsigned int) strtol(argv[1], &endptr, 10);
    if (*endptr != '\0' || version > 1) {
        printf("Invalid version: %s (parsed: %ud)\n", argv[1], version);
        return 1;
    }

    struct OpStats (*impl)(unsigned int *, double, double, double, double, int, int, int);
    impl = (version == 0) ? naive_mandelbrot : optimized_mandelbrot;

    // Rational boundaries
    lower_real = strtod(argv[2], &endptr);
    if (*endptr != '\0') {
        printf("Invalid lower rational boundary: %s\n", argv[2]);
        return 1;
    }

    upper_real = strtod(argv[3], &endptr);
    if (*endptr != '\0') {
        printf("Invalid upper rational boundary: %s\n", argv[3]);
        return 1;
    }

    // Irrational boundaries
    lower_imaginary = strtod(argv[4], &endptr);
    if (*endptr != '\0') {
        printf("Invalid lower irrational boundary: %s\n", argv[4]);
        return 1;
    }

    upper_imaginary = strtod(argv[5], &endptr);
    if (*endptr != '\0') {
        printf("Invalid upper irrational boundary: %s\n", argv[4]);
        return 1;
    }

    // Image dimensions
    height = (int) strtol(argv[6], &endptr, 10);
    if (*endptr != '\0') {
        printf("Invalid height value: %s\n", argv[6]);
        return 1;
    }

    width = (int) strtol(argv[7], &endptr, 10);
    if (*endptr != '\0') {
        printf("Invalid width value: %s\n", argv[7]);
        return 1;
    }

    // Maximum iterations
    max_iterations = (int) strtol(argv[8], &endptr, 10);
    if (*endptr != '\0') {
        printf("Invalid maximum number of iterations: %s\n", argv[8]);
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

    // allocate the iteration matrix on the heap
    // we will save this matrix to binary file, so we use uint32_t to ensure the size
    uint32_t *n_iterations =
            (uint32_t *) allocate_aligned_memory(64, height * width * sizeof(uint32_t));
    if (!n_iterations) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    struct OpStats result = impl(n_iterations, lower_real, upper_real, lower_imaginary,
                                 upper_imaginary, height, width, max_iterations);

    printf("Operations: %llu, Time: %llu ns, ns/op: %Lf\n", result.n_op, result.n_ns,
           (long double) result.n_ns / (long double) result.n_op);

    save_matrix(n_iterations, height, width, "result.bin");

    free_aligned_memory(n_iterations);

    return 0;
}
