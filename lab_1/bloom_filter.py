import hashlib
import math
import random
import string
import statistics

N = 15_000
EPSILON = 0.5


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
        hash_obj = hashlib.blake2b(data, digest_size=4)
        hash_value = int.from_bytes(hash_obj.digest(), byteorder='big')
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
        super().__init__(n=n, epsilon=epsilon, m=m, k=k)
        self.counters = bytearray(self.m)

    def add(self, item):
        for i in range(self.k):
            index = self._hash(item, i)
            if self.counters[index] < 255:
                self.counters[index] += 1
    
    def remove(self, item):
        if not self.contains(item):
            return False
        for i in range(self.k):
            index = self._hash(item, i)
            if self.counters[index] > 0:
                self.counters[index] -= 1
        return True
    
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
        result.counters = bytearray(min(max(self.counters[i], other.counters[i]), 255) for i in range(self.m))
        return result
    
    def __sub__(self, other):
        if self.m != other.m or self.k != other.k:
            raise TypeError("Counting Bloom filters must have the same parameters (m, k)")
        result = CountingBloomFilter(m=self.m, k=self.k)
        result.counters = bytearray(min(self.counters[i], other.counters[i]) for i in range(self.m))
        return result


def generate_random_string(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def test_bloom_filter():
    print()
    print("ТЕСТИРОВАНИЕ ФИЛЬТРА БЛУМА")
    print(f"Параметры: n={N}, ε={EPSILON}")
    
    bf = BloomFilter(n=N, epsilon=EPSILON)
    print(f"Рассчитано: m={bf.m}, k={bf.k}")
    print()
    
    print("\n1. Базовые тесты:")
    test_items = [generate_random_string() for _ in range(100)]
    for item in test_items[:50]:
        bf.add(item)
    
    true_positives = sum(1 for item in test_items[:50] if bf.contains(item))
    print(f"  True Positive: {true_positives}/50 (должно быть 50)")
    
    false_positives = sum(1 for item in test_items[50:] if bf.contains(item))
    print(f"  False Positive на новых данных: {false_positives}/50")
    
    print("\n2. Оценка процента ложноположительных срабатываний:")
    percentages = [0.25, 0.50, 0.75, 0.95]
    
    for pct in percentages:
        n_current = int(N * pct)
        fp_rates = []
        
        for _ in range(3):
            cbf = CountingBloomFilter(n=N, epsilon=EPSILON)
            added = set()
            while len(added) < n_current:
                added.add(generate_random_string())
            for item in added:
                cbf.add(item)
            
            test_set = set()
            while len(test_set) < n_current:
                item = generate_random_string()
                if item not in added:
                    test_set.add(item)
            
            fp_count = sum(1 for item in test_set if cbf.contains(item))
            fp_rates.append(fp_count / len(test_set))
        
        avg_fp = statistics.mean(fp_rates)
        theory_fp = (1 - math.exp(-bf.k * n_current / bf.m)) ** bf.k
        
        print(f"  Наполненность {pct*100:>3.0f}%: "
              f"Опытный FP = {avg_fp:.4f}, "
              f"Теоретический FP = {theory_fp:.4f}")
    
    print("\n3. Тест операций объединения (+) и пересечения (-):")
    bf1 = BloomFilter(m=bf.m, k=bf.k)
    bf2 = BloomFilter(m=bf.m, k=bf.k)
    
    common = [generate_random_string() for _ in range(20)]
    only1 = [generate_random_string() for _ in range(20)]
    only2 = [generate_random_string() for _ in range(20)]
    
    for item in common + only1: bf1.add(item)
    for item in common + only2: bf2.add(item)
    
    bf_union = bf1 + bf2
    bf_inter = bf1 - bf2
    
    print(f"  Объединение:")
    print(f"    'common_0' в union: {bf_union.contains(common[0])} (True)")
    print(f"    'only1_0' в union: {bf_union.contains(only1[0])} (True)")
    print(f"    'only2_0' в union: {bf_union.contains(only2[0])} (True)")
    
    print(f"  Пересечение:")
    print(f"    'common_0' в inter: {bf_inter.contains(common[0])} (True)")
    print(f"    'only1_0' в inter: {bf_inter.contains(only1[0])} (False)")
    
    print("\n4. Тест CountingBloomFilter удаления:")
    cbf_test = CountingBloomFilter(n=100, epsilon=0.1)
    item_to_remove = "unique_item_123"
    cbf_test.add(item_to_remove)
    print(f"  Добавлен '{item_to_remove}'. Содержится: {cbf_test.contains(item_to_remove)}")
    cbf_test.remove(item_to_remove)
    print(f"  Удален '{item_to_remove}'. Содержится: {cbf_test.contains(item_to_remove)} (False)")


if __name__ == "__main__":
    test_bloom_filter()