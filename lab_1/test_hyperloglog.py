import random
from hyperloglog import HyperLogLog
from data_generator import DataGenerator


def test_hyperloglog_accuracy(unique_count, total_count, epsilon, test_name, runs=3):
    print()
    print(f"Тест: {test_name}")
    print()
    print(f"Целевая мощность (cardinality): {unique_count}")
    print(f"Размер потока (с дубликатами): {total_count}")
    
    errors = []
    
    for run in range(runs):
        gen = DataGenerator(seed=42 + run)
        hll = HyperLogLog(epsilon=epsilon)
        
        stream = gen.generate_stream_with_duplicates(unique_count, total_count)
        
        for item in stream:
            hll.add(item)
        
        estimated = hll.count()
        actual = unique_count
        error = abs(estimated - actual) / actual
        errors.append(error)
        
        print(f"  Запуск {run + 1}:")
        print(f"    Реальное: {actual}")
        print(f"    Оценка HLL: {estimated}")
        print(f"    Погрешность: {error:.4f} ({error*100:.2f}%)")
    
    avg_error = sum(errors) / len(errors)
    print(f"\n  Итоговая средняя погрешность: {avg_error:.4f} ({avg_error*100:.2f}%)")
    print(f"  Ожидаемая точность (epsilon): {epsilon}")
    print(f"  Параметры HLL: p={hll.p}, m={hll.m}")
    print("-" * 50)
    
    return avg_error


def test_hyperloglog_merge():
    print()
    print("Тест объединения (Merge) двух структур HLL")
    print()
    
    hll1 = HyperLogLog(epsilon=0.01)
    hll2 = HyperLogLog(epsilon=0.01)
    
    gen = DataGenerator(seed=42)
    
    set1 = list(gen.generate_unique_dates(5000))
    set2 = list(gen.generate_unique_dates(5000))
    
    for item in set1:
        hll1.add(item)
    for item in set2:
        hll2.add(item)
    
    hll_merged = hll1 + hll2
    
    actual_union = len(set(set1 + set2))
    estimated_union = hll_merged.count()
    error = abs(estimated_union - actual_union) / actual_union
    
    print(f"  Set1 size: {len(set1)}")
    print(f"  Set2 size: {len(set2)}")
    print(f"  Реальный размер объединения: {actual_union}")
    print(f"  Оценка HLL объединения: {estimated_union}")
    print(f"  Погрешность объединения: {error:.4f} ({error*100:.2f}%)")


def test_hyperloglog_small_sets():
    print("\n" + "="*50)
    print("Тест на малых множествах (Linear Counting)")
    print("="*50)
    
    hll = HyperLogLog(epsilon=0.01)
    gen = DataGenerator(seed=42)
    
    items = list(gen.generate_unique_dates(100))
    
    for item in items:
        hll.add(item)
    
    estimated = hll.count()
    actual = len(items)
    error = abs(estimated - actual) / actual
    
    print(f"  Реальный размер: {actual}")
    print(f"  Оценка HLL: {estimated}")
    print(f"  Погрешность: {error:.4f} ({error*100:.2f}%)")
    
    is_linear = estimated < (2.5 * hll.m)
    print(f"  Используется Linear Counting: {is_linear}")


if __name__ == "__main__":
    random.seed(42)
    
    test_hyperloglog_accuracy(
        unique_count=12_000,       
        total_count=36_000,       
        epsilon=0.05,       
        test_name="Small (12,000)"
    )
    
    test_hyperloglog_accuracy(
        unique_count=279_000,     
        total_count=837_000,      
        epsilon=0.01,
        test_name="Norm (279,000)"
    )
    
    test_hyperloglog_accuracy(
        unique_count=1_150_460,   
        total_count=3_451_380,    
        epsilon=0.01,
        test_name="Big (1,150,460)"
    )

    test_hyperloglog_merge()
    test_hyperloglog_small_sets()