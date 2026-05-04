import random
from collections import Counter
from count_min_sketch import CountMinSketch
from data_generator import DataGenerator

SMALL_COUNT = 12_000
NORM_COUNT = 279_000
BIG_COUNT = 1_150_460

SMALL_EPSILON = 0.05
SMALL_DELTA = 0.05

NORM_EPSILON = 0.01
NORM_DELTA = 0.01

BIG_EPSILON = 0.005
BIG_DELTA = 0.01


def test_cms_accuracy(total_count, epsilon, delta, test_name, runs=3):
    print()
    print(f"Тест точности: {test_name}")
    print()
    print(f"Параметры: n={total_count}, ε={epsilon}, δ={delta}")
    
    errors = []
    
    for run in range(runs):
        gen = DataGenerator(seed=42 + run)
        cms = CountMinSketch(epsilon=epsilon, delta=delta)
        
        stream = list(gen.generate_stream_with_duplicates(int(total_count * 0.5), total_count))
        actual_counts = Counter(stream)
        
        for item in stream:
            cms.add(item)
        
        top_items = actual_counts.most_common(10)
        run_errors = []
        
        for item, actual_count in top_items:
            estimated_count = cms.estimate(item)
            error = abs(estimated_count - actual_count) / actual_count if actual_count > 0 else 0
            run_errors.append(error)
        
        avg_error = sum(run_errors) / len(run_errors) if run_errors else 0
        errors.append(avg_error)
        
        print(f"  Запуск {run + 1}: средняя ошибка = {avg_error:.4f} ({avg_error*100:.2f}%), "
              f"w={cms.w}, d={cms.d}, память={cms.w*cms.d*8/1024:.2f} KB")
    
    final_error = sum(errors) / len(errors)
    print(f"\n  Итоговая средняя ошибка: {final_error:.4f} ({final_error*100:.2f}%)")
    print(f"  Ожидаемая точность ε: {epsilon:.4f}")
    print(f"  {'OK' if final_error <= epsilon * 2 else 'WARNING: ошибка выше ожидаемой'}")
    
    return final_error


def test_cms_merge():
    print("Тест объединения (merge) Count-Min Sketch")
    
    cms1 = CountMinSketch(epsilon=0.01, delta=0.01)
    cms2 = CountMinSketch(epsilon=0.01, delta=0.01)
    
    gen = DataGenerator(seed=42)
    
    stream1 = list(gen.generate_stream_with_duplicates(5000, 25000))
    stream2 = list(gen.generate_stream_with_duplicates(5000, 25000))
    
    for item in stream1:
        cms1.add(item)
    for item in stream2:
        cms2.add(item)
    
    cms_merged = cms1 + cms2
    
    combined = stream1 + stream2
    actual_counts = Counter(combined)
    
    top_items = actual_counts.most_common(5)
    errors = []
    
    for item, actual_count in top_items:
        estimated = cms_merged.estimate(item)
        error = abs(estimated - actual_count) / actual_count if actual_count > 0 else 0
        errors.append(error)
    
    print(f"  Поток 1: {cms1.total_count} элементов")
    print(f"  Поток 2: {cms2.total_count} элементов")
    print(f"  Объединено: {cms_merged.total_count} элементов")
    print(f"  Средняя ошибка: {sum(errors)/len(errors):.4f} ({sum(errors)/len(errors)*100:.2f}%)")
    print(f"  {'OK' if sum(errors)/len(errors) < 0.05 else 'WARNING'}")


def test_hyperparameter_sensitivity():
    print()
    print("Исследование чувствительности к параметрам")
    print()
    
    gen = DataGenerator(seed=42)
    stream = list(gen.generate_stream_with_duplicates(5000, 50000))
    actual_counts = Counter(stream)
    
    print("\n  Варьирование Epsilon (δ=0.01):")
    print(f"  {'Epsilon':>10} {'w':>8} {'d':>8} {'Avg Error':>12} {'Memory KB':>12}")
    print("  " + "-" * 65)
    
    for eps in [0.001, 0.005, 0.01, 0.05, 0.1]:
        cms = CountMinSketch(epsilon=eps, delta=0.01)
        for item in stream:
            cms.add(item)
        
        top_items = actual_counts.most_common(20)
        errors = [abs(cms.estimate(item) - count) / count for item, count in top_items if count > 0]
        avg_error = sum(errors) / len(errors) if errors else 0
        memory_kb = (cms.w * cms.d * 8) / 1024
        
        print(f"  {eps:>10.3f} {cms.w:>8} {cms.d:>8} {avg_error:>11.2%} {memory_kb:>12.2f}")
    
    print("\n  Варьирование Delta (ε=0.01):")
    print(f"  {'Delta':>10} {'w':>8} {'d':>8} {'Avg Error':>12} {'Memory KB':>12}")
    print("  " + "-" * 65)
    
    for dlt in [0.001, 0.01, 0.05, 0.1]:
        cms = CountMinSketch(epsilon=0.01, delta=dlt)
        for item in stream:
            cms.add(item)
        
        top_items = actual_counts.most_common(20)
        errors = [abs(cms.estimate(item) - count) / count for item, count in top_items if count > 0]
        avg_error = sum(errors) / len(errors) if errors else 0
        memory_kb = (cms.w * cms.d * 8) / 1024
        
        print(f"  {dlt:>10.3f} {cms.w:>8} {cms.d:>8} {avg_error:>11.2%} {memory_kb:>12.2f}")


def test_all_sizes():
    print("ТЕСТИРОВАНИЕ ВСЕХ РАЗМЕРОВ ИЗ ТАБЛИЦЫ")
    
    results = {}
    
    results['small'] = test_cms_accuracy(
        total_count=SMALL_COUNT,
        epsilon=SMALL_EPSILON,
        delta=SMALL_DELTA,
        test_name="Small (12,000)"
    )
    
    results['norm'] = test_cms_accuracy(
        total_count=NORM_COUNT,
        epsilon=NORM_EPSILON,
        delta=NORM_DELTA,
        test_name="Norm (279,000)"
    )
    
    results['big'] = test_cms_accuracy(
        total_count=BIG_COUNT,
        epsilon=BIG_EPSILON,
        delta=BIG_DELTA,
        test_name="Big (1,150,460)"
    )


    print(f"{'Размер':>10} {'Элементов':>12} {'Epsilon':>10} {'Ошибка':>12} {'Статус':>10}")
    print()
    
    configs = [
        ('small', SMALL_COUNT, SMALL_EPSILON),
        ('norm', NORM_COUNT, NORM_EPSILON),
        ('big', BIG_COUNT, BIG_EPSILON)
    ]
    
    for name, count, eps in configs:
        error = results[name]
        status = "OK" if error <= eps * 2 else "WARNING"
        print(f"{name:>10} {count:>12,} {eps:>10.3f} {error:>11.2%} {status:>10}")
    
    print("="*60)


if __name__ == "__main__":
    random.seed(42)
    
    test_all_sizes()
    
    test_cms_merge()
    test_hyperparameter_sensitivity()