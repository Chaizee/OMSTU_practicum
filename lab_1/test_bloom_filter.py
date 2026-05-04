import random
import string
import math
from bloom_filter import BloomFilter, CountingBloomFilter

N_CONFIG = 15_000
EPSILON_CONFIG = 0.5



def generate_random_string(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def calculate_theoretical_fpr(m, k, n_current):
    """Формула 1: (1 - e^(-kn/m))^k"""
    if n_current == 0:
        return 0.0
    return (1 - math.exp(-k * n_current / m)) ** k


def test_bloom_filter_fpr(n, epsilon, runs=3):
    print(f"\nТестирование FPR | n={n}, целевой epsilon={epsilon}")
    print()
    
    for fill_rate in [0.25, 0.50, 0.75, 0.95]:
        fprs = []
        items_to_add = int(n * fill_rate)
        
        for run in range(runs):
            bf = BloomFilter(n=n, epsilon=epsilon)
            added_items = set()
            
            for _ in range(items_to_add):
                item = generate_random_string()
                added_items.add(item)
                bf.add(item)
            
            false_positives = 0
            for _ in range(items_to_add):
                test_item = generate_random_string()
                while test_item in added_items:
                    test_item = generate_random_string()
                if bf.contains(test_item):
                    false_positives += 1
            
            fprs.append(false_positives / items_to_add)
        
        avg_fpr = sum(fprs) / len(fprs)

        theory_fpr = calculate_theoretical_fpr(bf.m, bf.k, items_to_add)
        
        print(f"Заполнение {int(fill_rate*100):>2}%: "
              f"Опытный FPR = {avg_fpr:.4f} | "
              f"Теоретический FPR = {theory_fpr:.4f}")


def test_operators():
    
    bf1 = BloomFilter(n=1000, epsilon=0.01)
    bf2 = BloomFilter(n=1000, epsilon=0.01)
    
    items1 = [f"item1_{i}" for i in range(50)]
    items2 = [f"item2_{i}" for i in range(50)]
    common = [f"common_{i}" for i in range(20)]
    
    for item in items1 + common:
        bf1.add(item)
    for item in items2 + common:
        bf2.add(item)
    
    bf_union = bf1 + bf2
    bf_inter = bf1 - bf2
    
    union_correct = all(bf_union.contains(item) for item in items1 + items2 + common)
    print(f"Объединение (union) работает: {union_correct}")
    
    common_in_inter = sum(1 for item in common if bf_inter.contains(item))
    only1_in_inter = sum(1 for item in items1 if bf_inter.contains(item))
    only2_in_inter = sum(1 for item in items2 if bf_inter.contains(item))
    
    print(f"Пересечение (inter):")
    print(f"    Найдено общих элементов: {common_in_inter}/{len(common)} (ожидалось {len(common)})")
    print(f"    Ложных срабатываний из bf1: {only1_in_inter}/{len(items1)} (ожидалось ~0)")
    print(f"    Ложных срабатываний из bf2: {only2_in_inter}/{len(items2)} (ожидалось ~0)")


def test_counting():
    print("Тест Counting Bloom Filter (добавление/удаление)")
    
    cbf = CountingBloomFilter(n=500, epsilon=0.01)
    items = [f"count_item_{i}" for i in range(100)]
    
    for item in items:
        cbf.add(item)
        
    initial_check = sum(1 for item in items if cbf.contains(item))
    print(f"После добавления 100 элементов найдено: {initial_check}/100")
    
    for i in range(50):
        cbf.remove(items[i])
        
    removed_found = sum(1 for i in range(50) if cbf.contains(items[i]))
    kept_found = sum(1 for i in range(50, 100) if cbf.contains(items[i]))
    
    print(f"После удаления 50 элементов:")
    print(f"  Найдено удаленных: {removed_found}/50 (ожидалось ~0, возможны коллизии)")
    print(f"  Найдено оставшихся: {kept_found}/50 (ожидалось 50)")


if __name__ == "__main__":
    random.seed(42)
    
    test_bloom_filter_fpr(n=N_CONFIG, epsilon=EPSILON_CONFIG, runs=3)
    
    test_operators()
    test_counting()