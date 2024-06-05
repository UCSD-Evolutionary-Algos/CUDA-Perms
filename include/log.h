#ifndef __LOG_H
#define __LOG_H

#include <iostream>

using namespace std;

struct Log {
    static void die(string msg) {
        Log::err(msg);
        exit(1);
    }
    static void err(string msg) {
        cerr << "\x1b[31m[FAIL] \x1b[0m" << msg << endl;
    }
    static void warn(string msg) {
        cout << "\x1b[33m[WARN] \x1b[0m" << msg << endl;
    }
    static void info(string msg) {
        cout << "\x1b[36m[INFO] \x1b[0m" << msg << endl;
    }
#ifdef DEBUG
    static void debug(string msg) {
        cout << "\x1b[34m[DEBG] \x1b[0m" << msg << endl;
    }
#else
    static inline void __attribute__((always_inline)) debug(string msg) {}
#endif // DEBUG
};

#endif // __CONFIG_H

