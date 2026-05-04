import hashlib
import math


class BloomFilter:
    def __init__(self, n=None, epsilon=None, m=None, k=None):
        if m is not None and k is not None:
            self.m = m
            self.k = k
        elif n is not None and epsilon is not None:
            self.m = self._calculate_m(n, epsilon)
            self.k = self._calculate_k(self.m, n)
        else:
            raise ValueError("Either (n, epsilon) or (m, k) must be provided")
        
        self.bit_array = bytearray(self.m)
        self.hash_limit = 2 ** 30
    
    @staticmethod
    def _calculate_m(n, epsilon):
        m = -1 * (n * math.log(epsilon)) / (math.log(2) ** 2)
        return int(math.ceil(m))
    
    @staticmethod
    def _calculate_k(m, n):
        k = (m / n) * math.log(2)
        return max(1, int(math.ceil(k)))
    
    def _hash(self, item, seed):
        data = f"{item}{seed}".encode('utf-8')
        hash_value = int(hashlib.sha3_256(data).hexdigest(), 16)
        return (hash_value % self.hash_limit) % self.m
    
    def add(self, item):
        for i in range(self.k):
            index = self._hash(item, i)
            self.bit_array[index] = 1
    
    def contains(self, item):
        for i in range(self.k):
            index = self._hash(item, i)
            if self.bit_array[index] == 0:
                return False
        return True
    
    def __add__(self, other):
        if self.m != other.m or self.k != other.k:
            raise TypeError("Bloom filters must have the same parameters (m, k)")
        
        result = BloomFilter(m=self.m, k=self.k)
        result.bit_array = bytearray(self.bit_array[i] | other.bit_array[i] for i in range(self.m))
        return result
    
    def __sub__(self, other):
        if self.m != other.m or self.k != other.k:
            raise TypeError("Bloom filters must have the same parameters (m, k)")
        
        result = BloomFilter(m=self.m, k=self.k)
        result.bit_array = bytearray(self.bit_array[i] & other.bit_array[i] for i in range(self.m))
        return result


class CountingBloomFilter(BloomFilter):
    def __init__(self, n=None, epsilon=None, m=None, k=None):
        if m is not None and k is not None:
            self.m = m
            self.k = k
        elif n is not None and epsilon is not None:
            self.m = BloomFilter._calculate_m(n, epsilon)
            self.k = BloomFilter._calculate_k(self.m, n)
        else:
            raise ValueError("Either (n, epsilon) or (m, k) must be provided")
        
        self.counters = bytearray(self.m)
        self.hash_limit = 2 ** 30
    
    def _hash(self, item, seed):
        data = f"{item}{seed}".encode('utf-8')
        hash_value = int(hashlib.sha3_256(data).hexdigest(), 16)
        return (hash_value % self.hash_limit) % self.m
    
    def add(self, item):
        for i in range(self.k):
            index = self._hash(item, i)
            self.counters[index] += 1
    
    def remove(self, item):
        if not self.contains(item):
            return
        
        for i in range(self.k):
            index = self._hash(item, i)
            if self.counters[index] > 0:
                self.counters[index] -= 1
    
    def contains(self, item):
        for i in range(self.k):
            index = self._hash(item, i)
            if self.counters[index] == 0:
                return False
        return True
    
    def __add__(self, other):
        if self.m != other.m or self.k != other.k:
            raise TypeError("Counting Bloom filters must have the same parameters (m, k)")
        
        result = CountingBloomFilter(m=self.m, k=self.k)
        result.counters = bytearray(max(self.counters[i], other.counters[i]) for i in range(self.m))
        return result
    
    def __sub__(self, other):
        if self.m != other.m or self.k != other.k:
            raise TypeError("Counting Bloom filters must have the same parameters (m, k)")
        
        result = CountingBloomFilter(m=self.m, k=self.k)
        result.counters = bytearray(min(self.counters[i], other.counters[i]) for i in range(self.m))
        return result
