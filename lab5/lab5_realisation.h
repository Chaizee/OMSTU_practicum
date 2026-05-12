#ifndef PRIME_H
#define PRIME_H

#ifdef _WIN32
    #define DLL_EXPORT __declspec(dllexport)
#else
    #define DLL_EXPORT
#endif

DLL_EXPORT int prime_numbers_count(int L, int R);

#endif