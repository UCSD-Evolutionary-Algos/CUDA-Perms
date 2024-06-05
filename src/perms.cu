#include <iostream>
#include <csignal>
#include <algorithm>
#include <random>
#include <chrono>
#include <queue>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/shuffle.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include "log.h"
#include "config.h"

#define DEVICE_SIZEOF_INT 4

using namespace std;

perms_config cfg;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess)  {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define SIGN(x) (x < 0 ? (-1) : 1)
#define ENFORCE_SAME(a, b, x, y) if (SIGN(perm[b] - perm[a]) != SIGN(pattern[y] - pattern[x])) continue
__global__ void get_score(
    ENTRY *population,
    int *scores,
    int *pattern,
    int permutation_length,
    int pattern_length,
    int population_size
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (index >= population_size) return;
    for (int i = index; i < population_size; i += stride) {
        ENTRY *perm = &population[i * permutation_length];
        int score = 0;

#pragma unroll 1
        for (int j = 0; j < permutation_length; j++) {
#pragma unroll 1
            for (int k = j + 1; k < permutation_length; k++) {
                ENFORCE_SAME(j, k, 0, 1);
                if (pattern_length == 2) {
                    score++;
                    continue;
                }

#pragma unroll 1
                for (int l = k + 1; l < permutation_length; l++) {
                    ENFORCE_SAME(k, l, 1, 2);
                    ENFORCE_SAME(j, l, 0, 2);
                    if (pattern_length == 3) {
                        score++;
                        continue;
                    }

#pragma unroll 1
                    for (int m = l + 1; m < permutation_length; m++) {
                        ENFORCE_SAME(l, m, 2, 3);
                        ENFORCE_SAME(k, m, 1, 3);
                        ENFORCE_SAME(j, m, 0, 3);
                        score++;
                    }
                }
            }
        }
        scores[i] = score;
    }
}

__global__ void breed_children(
    ENTRY *parents, ENTRY *children, int *indices,
    int pattern_length, int permutation_length, int population_size,
    int max_mutations, int fitness_evals
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (index >= population_size) return;

    bool *scratch = new bool[permutation_length];
    ENTRY *pair_keys = new ENTRY[permutation_length];
    ENTRY *pair_vals = new ENTRY[permutation_length];

    thrust::ranlux48 rand;
    rand.seed(fitness_evals + permutation_length);

    int upper_tenth = population_size / 10;
    const ENTRY *parentA = &parents[indices[rand() % upper_tenth] * permutation_length];
    const ENTRY *parentB = &parents[indices[rand() % upper_tenth] * permutation_length];

    for (int i = index; i < population_size; i += stride) {
        rand.seed(fitness_evals + i);
        ENTRY *perm = &children[i * permutation_length];

        // Copy to preserve strong parent
        if (i < upper_tenth) {
            for (int j = 0; j < permutation_length; j++) {
                ENTRY *tmp = &parents[indices[0] * permutation_length];
                perm[j] = tmp[j];
            }
            continue;
        }

        int n = rand() % 3;
        switch (n) {
            // Cut-and-crossfill
            case 0: {
                for (int j = 0; j < permutation_length; j++) scratch[j] = false;

                int from = rand() % permutation_length;
                int to = rand() % permutation_length;
                while (from != to) {
                    ENTRY v = parentA[from];
                    perm[from] = v;
                    scratch[v] = true;
                    from = (from + 1) % permutation_length;
                }

                for (int j = 0; j < permutation_length; j++) {
                    ENTRY v = parentB[j];
                    if (scratch[v]) continue;
                    perm[from] = v;
                    scratch[v] = true;

                    from = (from + 1) % permutation_length;
                }
                break;
            }
            // Flip-and-scan
            case 1: {
                for (int j = 0; j < permutation_length; j++) scratch[j] = false;
                for (int j = 0; j < permutation_length; j++) {
                    const ENTRY *parent = (rand() % 2) == 1 ? parentA : parentB;

                    int k = j;
                    while (scratch[parent[k]]) k = (k + 1) % permutation_length;

                    ENTRY v = parent[k];
                    perm[j] = v;
                    scratch[v] = true;
                }
                break;
            }
            // Flip-and-shift
            case 2: {
                for (int j = 0; j < permutation_length; j++) {
                    if ((rand() % 2) == 1) {
                        pair_keys[j] = (parentA[j] * 3) + 1;
                    } else {
                        pair_keys[j] = (parentB[j] * 3) - 1;
                    }
                    pair_vals[j] = j;
                }
                thrust::sort_by_key(
                    thrust::device,
                    &pair_keys[0],
                    &pair_keys[permutation_length],
                    &pair_vals[0],
                    thrust::greater<int>()
                );
                for (int j = 0; j < permutation_length; j++) {
                    perm[j] = pair_vals[j];
                }
                break;
            }
        }

        // Mutate
        if (max_mutations > 0) {
            int mutationCount = rand() % (max_mutations + 1);
            for (int j = 0; j < mutationCount; j++) {
                int a = rand() % permutation_length;
                int b = rand() % permutation_length;
                while (b == a) b = rand() % permutation_length;

                ENTRY tmp = perm[a];
                perm[a] = perm[b];
                perm[b] = tmp;
            }
        }
    }
    delete[] scratch;
    delete[] pair_keys;
    delete[] pair_vals;
}

__global__ void generate_parents(
    ENTRY *parents, int permutation_length, int population_size
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (index >= population_size) return;

    thrust::default_random_engine generator(population_size);
    for (int i = index; i < population_size; i += stride) {
        thrust::shuffle_copy(
            thrust::device,
            thrust::counting_iterator<ENTRY>(0),
            thrust::counting_iterator<ENTRY>(permutation_length),
            parents + (i * permutation_length),
            generator
        );

        /*
        // Correctness check: 2413 optimizer
        parents[(i * permutation_length) + 0] = 4;
        parents[(i * permutation_length) + 1] = 5;
        parents[(i * permutation_length) + 2] = 7;
        parents[(i * permutation_length) + 3] = 9;
        parents[(i * permutation_length) + 4] = 15;
        parents[(i * permutation_length) + 5] = 14;
        parents[(i * permutation_length) + 6] = 3;
        parents[(i * permutation_length) + 7] = 13;
        parents[(i * permutation_length) + 8] = 2;
        parents[(i * permutation_length) + 9] = 12;
        parents[(i * permutation_length) + 10] = 0;
        parents[(i * permutation_length) + 11] = 1;
        parents[(i * permutation_length) + 12] = 6;
        parents[(i * permutation_length) + 13] = 8;
        parents[(i * permutation_length) + 14] = 10;
        parents[(i * permutation_length) + 15] = 11;
        */
    }
}

void stop(int sig) {
    printf("\x1b[0m\x1b[?25h\nCtrl-C received, shutting down.\n");
    exit(0);
}

#define POP_ARG(i, v) if (argc >= i + 1) { v = atoi(argv[i]); }
int main(int argc, char** argv) {
    std::signal(SIGINT, stop);
    std::signal(SIGTERM, stop);
    std::signal(SIGKILL, stop);

    if (argc < 2) {
        Log::die("Usage: perms [config file]");
    }
    cfg = perms_config(argv[1]);
    Log::info(string("Crossover functions:") +
        string(cfg.crossover_funcs[0] ? " cut-and-crossfill" : "") +
        string(cfg.crossover_funcs[1] ? " flip-and-scan" : "") +
        string(cfg.crossover_funcs[2] ? " flip-and-shift" : "")
    );
    Log::info("Pattern: " + to_string(cfg.raw));
    Log::info("Permutation length: " + to_string(cfg.permutation_length));
    Log::info("Fitness evals: " + to_string(cfg.fitness_evals));
    Log::info("Max mutation count: " + to_string(cfg.max_mutations));
    Log::info("Generating initial parent population...");
    if (cfg.view == FULL) printf("\x1b[2J\x1b[?25l");
    else if (cfg.view == SIMPLE) printf("\x1b[?25l");
    setvbuf(stdout, NULL, _IOFBF, 4096);

    chrono::time_point<chrono::system_clock> start, end;
    start = chrono::system_clock::now();

    int *pattern;
    int *indices;

    cudaMalloc(&pattern, cfg.pattern_length * DEVICE_SIZEOF_INT);
    cudaMalloc(&indices, cfg.population_size * DEVICE_SIZEOF_INT);

    thrust::device_vector<int> scores(cfg.population_size);
    int *raw_scores = thrust::raw_pointer_cast(scores.data());

    thrust::device_vector<ENTRY> parents(cfg.permutation_length * cfg.population_size);
    ENTRY *raw_parents = thrust::raw_pointer_cast(parents.data());

    thrust::device_vector<ENTRY> children(cfg.permutation_length * cfg.population_size);
    ENTRY *raw_children = thrust::raw_pointer_cast(children.data());

    cudaMemcpy(
        pattern,
        cfg.pattern,
        cfg.pattern_length * DEVICE_SIZEOF_INT,
        cudaMemcpyHostToDevice
    );

    generate_parents<<<1024, 256>>>(
        raw_parents,
        cfg.permutation_length,
        cfg.population_size
    );

    get_score<<<1024, 256>>>(
        raw_parents,
        raw_scores,
        pattern,
        cfg.permutation_length,
        cfg.pattern_length,
        cfg.population_size
    );

    thrust::sequence(
        thrust::device,
        &indices[0],
        &indices[cfg.population_size]
    );
    thrust::sort_by_key(
        thrust::device,
        &raw_scores[0],
        &raw_scores[cfg.population_size],
        &indices[0],
        thrust::greater<int>()
    );

    end = chrono::system_clock::now();
    chrono::duration<double> elapsed = end - start;
    printf("\x1b[36m[INFO] \x1b[0mGenerated initial parent population in %.2fs.\x1b[0m\n\n", elapsed.count());
    fflush(stdout);
    // gpuErrchk(cudaPeekAtLastError());

    int best_score, best_idx;
    ENTRY *best_perm = new ENTRY[cfg.permutation_length];
    start = chrono::system_clock::now();
    for (int i = 0; i < cfg.fitness_evals; i += cfg.population_size) {
        // cudaDeviceSynchronize();
        breed_children<<<1024, 256>>>(
            raw_parents,
            raw_children,
            indices,
            cfg.pattern_length,
            cfg.permutation_length,
            cfg.population_size,
            cfg.max_mutations,
            i
        );
        // cudaDeviceSynchronize();
        // gpuErrchk(cudaPeekAtLastError());

        parents.swap(children);
        ENTRY *tmp = raw_parents;
        raw_parents = raw_children;
        raw_children = tmp;

        get_score<<<1024, 256>>>(
            raw_parents,
            raw_scores,
            pattern,
            cfg.permutation_length,
            cfg.pattern_length,
            cfg.population_size
        );
        // cudaDeviceSynchronize();
        // gpuErrchk(cudaPeekAtLastError());
        thrust::sequence(
            thrust::device,
            &indices[0],
            &indices[cfg.population_size]
        );
        thrust::sort_by_key(
            thrust::device,
            &scores[0],
            &scores[cfg.population_size],
            &indices[0],
            thrust::greater<int>()
        );
        // cudaDeviceSynchronize();
        // gpuErrchk(cudaPeekAtLastError());

        best_idx = 0;
        best_score = 0;
        cudaMemcpy(
            &best_idx,
            indices,
            DEVICE_SIZEOF_INT,
            cudaMemcpyDeviceToHost
        );
        // gpuErrchk(cudaPeekAtLastError());
        cudaMemcpy(
            best_perm,
            raw_parents + (best_idx * cfg.permutation_length),
            cfg.permutation_length * sizeof(ENTRY),
            cudaMemcpyDeviceToHost
        );
        // gpuErrchk(cudaPeekAtLastError());
        cudaMemcpy(
            &best_score,
            raw_scores,
            DEVICE_SIZEOF_INT,
            cudaMemcpyDeviceToHost
        );
        // gpuErrchk(cudaPeekAtLastError());

        end = chrono::system_clock::now();
        elapsed = end - start;
        double rate = static_cast<double>(i) / elapsed.count();
        switch (cfg.view) {
            case FULL: {
                printf("\x1b[38;2;255;255;255m");
                for (int j = 0; j < cfg.permutation_length; j++) {
                    printf("\x1b[%i;%iH\x1b[2K██", (cfg.permutation_length - best_perm[j]) + 1, (2 * j) + 1);
                }
                printf(
                    "\x1b[0m\x1b[%d;1H\x1b[34mPattern: \x1b[33m%d\x1b[0K\n\x1b[34mEvals: \x1b[33m%d\x1b[0K\n\x1b[34mFitness: \x1b[33m%d\x1b[0K\n\x1b[34mRate: \x1b[33m%.2f evals/sec\x1b[0K\n\x1b[34mPermutation: \x1b[33m",
                    cfg.permutation_length + 3,
                    cfg.raw,
                    i + 1,
                    best_score,
                    rate
                );
                for (int j = 0; j < cfg.permutation_length; j++) {
                    printf("%d ", best_perm[j]);
                }
                printf("\x1b[0m\x1b[0K\n");
                fflush(stdout);
                break;
            }
            case SIMPLE: {
                printf("\x1b[1E\x1b[2ABest fitness after \x1b[33m%i/%i\x1b[0m evals: \x1b[34m%i\x1b[35m (%.2f evals/sec)\x1b[0m\x1b[0K\n", i + 1, cfg.fitness_evals, best_score, rate);
                fflush(stdout);
                break;
            }
            case NONE: break;
        }
    }

    best_idx = 0;
    best_score = 0;
    cudaMemcpy(
        &best_idx,
        indices,
        DEVICE_SIZEOF_INT,
        cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        best_perm,
        raw_parents + (best_idx * cfg.permutation_length),
        cfg.permutation_length * sizeof(ENTRY),
        cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        &best_score,
        raw_scores,
        DEVICE_SIZEOF_INT,
        cudaMemcpyDeviceToHost
    );
    printf("\x1b[36m[INFO] \x1b[0mBest fitness after \x1b[33m%i\x1b[0m evals: \x1b[34m%i\n\x1b[36m[INFO] \x1b[0mPermutation: \x1b[33m", cfg.fitness_evals, best_score);
    for (int i = 0; i < cfg.permutation_length; i++) {
        printf("%d%c", best_perm[i], (i == cfg.permutation_length ? ' ' : ','));
    }
    printf("\x1b[0m\x1b[?25h\n");
    fflush(stdout);

    delete[] best_perm;
    cudaFree(pattern);
    // cudaFree(parents);
    // cudaFree(children);

    return 0;
}

