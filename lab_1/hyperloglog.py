import math
import cityhash


class HyperLogLog:
    def __init__(self, epsilon=None, p=None):
        if p is not None:
            self.p = p
        elif epsilon is not None:
            self.p = self._calculate_p(epsilon)
        else:
            raise ValueError("Either epsilon or p must be provided")
        self.m = 2 ** self.p
        self.registers = [0] * self.m
        self.alpha = self._get_alpha(self.m)
    
    @staticmethod
    def _calculate_p(epsilon):
        p = math.log2((1.04 / epsilon) ** 2)
        return max(4, min(16, int(math.ceil(p))))
    
    @staticmethod
    def _get_alpha(m):
        if m == 16:
            return 0.673
        elif m == 32:
            return 0.697
        elif m == 64:
            return 0.709
        elif m >= 128:
            return 0.7213 / (1 + 1.079 / m)
        else:
            return 0.5
    
    def _hash(self, item):
        return cityhash.CityHash64(item.encode('utf-8'))
    
    @staticmethod
    def _leading_zeros(value, max_bits=64):
        if value == 0:
            return max_bits
        return max_bits - value.bit_length()
    
    def add(self, item):
        hash_value = self._hash(item)
        
        j = hash_value & ((1 << self.p) - 1)
        
        w = hash_value >> self.p
        
        leading_zeros = self._leading_zeros(w, 64 - self.p) + 1
        
        self.registers[j] = max(self.registers[j], leading_zeros)
    
    def count(self):
        raw_estimate = self.alpha * (self.m ** 2) / sum(2 ** (-x) for x in self.registers)
        
        if raw_estimate <= 2.5 * self.m:
            zeros = self.registers.count(0)
            if zeros != 0:
                return int(self.m * math.log(self.m / zeros))
        
        if raw_estimate <= (1/30) * (2 ** 32):
            return int(raw_estimate)
        else:
            return int(-1 * (2 ** 32) * math.log(1 - raw_estimate / (2 ** 32)))
    
    def __add__(self, other):
        if self.p != other.p:
            raise TypeError("HyperLogLog instances must have the same p parameter")
        
        result = HyperLogLog(p=self.p)
        result.registers = [max(self.registers[i], other.registers[i]) for i in range(self.m)]
        return result
    
    def merge(self, other):
        if self.p != other.p:
            raise TypeError("HyperLogLog instances must have the same p parameter")
        
        for i in range(self.m):
            self.registers[i] = max(self.registers[i], other.registers[i])
