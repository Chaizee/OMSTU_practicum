import math
import cityhash

class CountMinSketch:
    def __init__(self, epsilon=None, delta=None, w=None, d=None):
        if w is not None and d is not None:
            self.w = w
            self.d = d
        elif epsilon is not None and delta is not None:
            self.w = self._calculate_w(epsilon)
            self.d = self._calculate_d(delta)
        else:
            raise ValueError("Either (epsilon, delta) or (w, d) must be provided")
        
        self.table = [[0] * self.w for _ in range(self.d)]
        self.epsilon = epsilon
        self.delta = delta
        self.total_count = 0
    
    @staticmethod
    def _calculate_w(epsilon):
        return int(math.ceil(math.e / epsilon))
    
    @staticmethod
    def _calculate_d(delta):
        return int(math.ceil(math.log(1 / delta)))
    
    def _hash(self, item, seed):
        data = f"{item}{seed}".encode('utf-8')
        return cityhash.CityHash64(data) % self.w
    
    def add(self, item, count=1):
        for i in range(self.d):
            j = self._hash(item, i)
            self.table[i][j] += count
        self.total_count += count
    
    def estimate(self, item):
        return min(self.table[i][self._hash(item, i)] for i in range(self.d))
    
    def __add__(self, other):
        if self.w != other.w or self.d != other.d:
            raise TypeError("Count-Min Sketch instances must have same parameters (w, d)")
        
        result = CountMinSketch(w=self.w, d=self.d)
        for i in range(self.d):
            for j in range(self.w):
                result.table[i][j] = self.table[i][j] + other.table[i][j]
        result.total_count = self.total_count + other.total_count
        return result