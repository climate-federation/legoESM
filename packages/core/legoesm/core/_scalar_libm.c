#include <math.h>
#include <stddef.h>

#define DEFINE_UNARY(NAME)                                                   \
    void legoesm_libm_##NAME##_f64(const double *input, double *output,       \
                                    size_t count) {                           \
        for (size_t index = 0; index < count; ++index) {                      \
            output[index] = NAME(input[index]);                              \
        }                                                                     \
    }                                                                         \
    void legoesm_libm_##NAME##_f32(const float *input, float *output,         \
                                    size_t count) {                           \
        for (size_t index = 0; index < count; ++index) {                      \
            output[index] = (float)NAME((double)input[index]);                \
        }                                                                     \
    }

DEFINE_UNARY(exp)
DEFINE_UNARY(log)
DEFINE_UNARY(log10)
DEFINE_UNARY(atan)
DEFINE_UNARY(tanh)
DEFINE_UNARY(sin)
DEFINE_UNARY(cos)

void legoesm_libm_pow_f64(const double *left, const double *right,
                          double *output, size_t count) {
    for (size_t index = 0; index < count; ++index) {
        output[index] = pow(left[index], right[index]);
    }
}

void legoesm_libm_pow_f32(const float *left, const float *right,
                          float *output, size_t count) {
    for (size_t index = 0; index < count; ++index) {
        output[index] = (float)pow((double)left[index], (double)right[index]);
    }
}
