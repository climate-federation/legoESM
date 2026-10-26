/* Minimal STREAM triad: the memory-bandwidth ceiling of the node the benchmark
 * is actually running on. Needed as a DENOMINATOR -- "the lane moves N GB/s" is
 * meaningless without the machine's own ceiling measured the same hour on the
 * same cores, and a published figure is not that.
 *
 * cc -O3 -march=native -fopenmp stream_triad.c -o stream_triad
 * ./stream_triad [n_elements_millions]   prints "STREAM_TRIAD_GBs <value>"
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    size_t n = (argc > 1 ? (size_t)atof(argv[1]) : 200) * 1000000ull;
    double *a = malloc(n * sizeof(double));
    double *b = malloc(n * sizeof(double));
    double *c = malloc(n * sizeof(double));
    if (!a || !b || !c) { fprintf(stderr, "alloc failed\n"); return 2; }
    /* First touch in parallel so the pages land on the thread's own NUMA node;
     * a serial init would measure one socket's bandwidth and halve the answer. */
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.0; }
    double best = 0.0;
    for (int rep = 0; rep < 5; rep++) {
        double t0 = omp_get_wtime();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) c[i] = a[i] + 3.0 * b[i];
        double dt = omp_get_wtime() - t0;
        double gbs = 3.0 * n * sizeof(double) / dt / 1e9;   /* 2 reads + 1 write */
        if (gbs > best) best = gbs;
    }
    printf("STREAM_TRIAD_GBs %.1f threads %d n %zu checksum %.1f\n",
           best, omp_get_max_threads(), n, c[n / 2]);
    free(a); free(b); free(c);
    return 0;
}
