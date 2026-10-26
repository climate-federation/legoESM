#include <stddef.h>

typedef double v2df __attribute__((vector_size(16)));

extern v2df glibc_v2_sin(v2df) __asm__("_ZGVbN2v_sin");

int zdf_glibc_v2_sin_rows(const double *input, double *output,
                          size_t nrow, size_t ncol) {
    if (input == NULL || output == NULL || ncol == 0 || ncol % 2 != 0) {
        return 1;
    }
    for (size_t j = 0; j < nrow; ++j) {
        const size_t row = j * ncol;
        for (size_t i = 0; i < ncol; i += 2) {
            v2df argument = {input[row + i], input[row + i + 1]};
            v2df result = glibc_v2_sin(argument);
            output[row + i] = result[0];
            output[row + i + 1] = result[1];
        }
    }
    return 0;
}
