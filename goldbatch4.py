# -*- coding: utf-8 -*-
"""
Created on Fri May 28 17:17:27 2021

@author: nv199
"""

import numpy
import timeit
import datetime 
import numba
import pickle

def sieve (n: int) -> numpy.ndarray:    #Erathostenes sieve for prime numbers
    primes = numpy.full(n, True)
    primes[0], primes[1] = False, False
    for i in range(2, int(numpy.sqrt(n) + 1)):
        if primes[i]:
            primes[i*i::i] = False
    return numpy.flatnonzero(primes)

def even(n: int) -> numpy.ndarray:    #Generates even numbers
    evens = numpy.full(n, True)
    evens[0] = False
    i=1
    if evens[i]:
        evens[2*i-1::2] = False
    return numpy.flatnonzero(evens)

@numba.jit(nopython = True)
def checker(evens, primes, k):   #Checks for the existence of a pair of Goldbach
    lista = []                #primes for each even
    for i in numba.prange(0,k):
        half = evens[i]/2

            
        half_index = numpy.searchsorted(primes,half)
        
        for j in primes[half_index:len(primes)]:
            
            m = evens[i]-j 
            prime_index = numpy.searchsorted(primes,m, side = 'left')
            
            if (primes[prime_index] == m):
                lista.append([m,j,evens[i]])
            
    return lista


if __name__ == "__main__":
    
    n = 100000
    timestart = timeit.default_timer()
    primes = sieve(n)
    print(primes)
    evens = even(n)
    print(evens)
    k = len(evens)
    
    triplets = checker(evens, primes,k)
    
    
    
    timestop = timeit.default_timer()
    timedelta = (timestop - timestart)
    print(f"Time Elapsed: {datetime.timedelta(seconds = timedelta)}")
    
    with open('goldbach_list.mat', 'wb') as f:
        pickle.dump(triplets, f)

else:
    pass