#ifndef __CONFIG_H
#define __CONFIG_H

#include <stdint.h>
#include <fstream>
#include <sstream>
#include <climits>

#include "log.h"

using namespace std;

#define MAX_PATTERN_LEN 10
#define N_CROSSOVER_FUNCS 3
#define ENTRY int
#define MAX_PERM_LEN INT_MAX

enum display_mode {
    FULL,
    SIMPLE,
    NONE
};

struct perms_config {
    vector<bool> crossover_funcs;
    ENTRY pattern[MAX_PATTERN_LEN];
    int pattern_length = 0;
    int permutation_length = 10; 
    int population_size = 1000;
    int fitness_evals = 10000;
    int max_mutations = 10;

    int block_size = 256;
    int total_threads = 1024;

    display_mode view = FULL;

    int raw;
    perms_config() {}
    perms_config(char *filename) {
        string str, cmd, func;
        ifstream in;
      
        in.open(filename);
        if (!in.is_open()) {
            Log::die("Error: Failed to open file \"" + string(filename) + "\" for reading.");
        }

        do {
            getline(in, str);

            if (str.find_first_not_of(" \t\r\n") == string::npos || str[0] == '#') continue;

            stringstream s(str);
            s >> cmd;

            if (cmd == "crossover") {
                crossover_funcs = vector<bool>(N_CROSSOVER_FUNCS, false);
                while (!s.eof()) {
                    s >> func;
                    if (func == "crossfill") crossover_funcs[0] = true;
                    else if (func == "flip") crossover_funcs[1] = true;
                    else if (func == "shift") crossover_funcs[2] = true;
                    else {
                        Log::warn("Invalid crossover function \"" + func + "\" detected, ignoring.");
                    }
                }
            } else if (cmd == "pattern") {
                s >> raw;

                int tmp = raw;
                pattern_length = (int)log10((double)tmp) + 1;
                if (pattern_length < 2 || pattern_length > 4) {
                    Log::die("Error: Pattern length must be between 2 and 4 (inclusive).");
                }
                for (int i = 0; i < pattern_length; i++) {
                    pattern[pattern_length - i - 1] = tmp % 10;
                    tmp /= 10;
                }
            } else if (cmd == "length") {
                s >> permutation_length;
                if (permutation_length < 1 || permutation_length > MAX_PERM_LEN) {
                    Log::die("Error: Permutation length must be between 1 and " + to_string(MAX_PERM_LEN) + " (inclusive).");
                }
            } else if (cmd == "population") {
                s >> population_size;
            } else if (cmd == "evals") {
                s >> fitness_evals;
            } else if (cmd == "blocksize") {
                s >> block_size;
            } else if (cmd == "threads") {
                s >> total_threads;
            } else if (cmd == "mutations") {
                s >> max_mutations;
            } else if (cmd == "display") {
                s >> func;
                if (func == "full") view = FULL;
                else if (func == "simple") view = SIMPLE;
                else if (func == "none") view = NONE;
                else {
                    Log::warn("Invalid display mode \"" + func + "\" detected, ignoring.");
                }
            }
        } while(in);

        if (pattern_length > permutation_length) {
            Log::die("Error: Pattern cannot be longer than permutation.");
        }
        int cnt = 0;
        for (bool x: crossover_funcs) {
            if (x) cnt++;
        }
        if (cnt == 0) {
            Log::die("Error: At least one crossover function must be enabled.");
        }
    }
};

#endif

