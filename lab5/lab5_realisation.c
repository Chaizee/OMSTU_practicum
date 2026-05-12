#include <stdio.h>

#include "lab5_realisation.h"

static int is_prime(int number)
{
    if (number < 2) return 0;

    int prime_status = 1;

    for (int digit = 2; digit * digit <= number; digit++) {
        if (number % digit == 0) {
            prime_status = 0;
            break;
        }
    }
    
    return prime_status;
}

DLL_EXPORT int prime_numbers_count(int L, int R) 
{
    int count = 0;
    
    for (int num = L; num <= R; num++) {
        if (is_prime(num)) count++;
    }
    return count;
}

int main(void) 
{
    int count = prime_numbers_count(-1, 3);
    printf("%d", count);
    return 0;
}