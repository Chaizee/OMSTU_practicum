

import hashlib
import math
import random
import string
import matplotlib.pyplot as plt
import pandas as pd
import os

os.makedirs('plots', exist_ok=True)

DEFAULT_N = 15_000
DEFAULT_EPSILON = 0.5

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = '#fafafa'
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['font.family'] = 'sans-serif'


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
    
    def get_fill_ratio(self):
        return sum(self.bit_array) / self.m * 100


def generate_random_string(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def measure_fp_rate(bf, added_items, test_items):
    fp_count = sum(1 for item in test_items 
                   if item not in added_items and bf.contains(item))
    total_test = len([item for item in test_items if item not in added_items])
    return fp_count / total_test if total_test > 0 else 0


def experiment_varying_epsilon(n=1000, trials=1):
    epsilons = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    results = []
    
    for epsilon in epsilons:
        bf = BloomFilter(n=n, epsilon=epsilon)
        
        added_items = set()
        while len(added_items) < n:
            added_items.add(generate_random_string())
        for item in added_items:
            bf.add(item)
        
        test_items = set()
        while len(test_items) < n:
            item = generate_random_string()
            if item not in added_items:
                test_items.add(item)
        
        empirical_fp = measure_fp_rate(bf, added_items, test_items)
        theoretical_fp = (1 - math.exp(-bf.k * n / bf.m)) ** bf.k
        
        results.append({
            'epsilon': epsilon,
            'm': bf.m,
            'k': bf.k,
            'fill_ratio': bf.get_fill_ratio(),
            'empirical_fp': empirical_fp,
            'theoretical_fp': theoretical_fp
        })
    
    return pd.DataFrame(results)


def experiment_varying_n(epsilon=0.1):
    n_values = [100, 500, 1000, 2000, 5000, 10000, 15000]
    results = []
    
    for n in n_values:
        bf = BloomFilter(n=n, epsilon=epsilon)
        
        added_items = set()
        while len(added_items) < n:
            added_items.add(generate_random_string())
        for item in added_items:
            bf.add(item)
        
        test_items = set()
        while len(test_items) < n:
            item = generate_random_string()
            if item not in added_items:
                test_items.add(item)
        
        empirical_fp = measure_fp_rate(bf, added_items, test_items)
        theoretical_fp = (1 - math.exp(-bf.k * n / bf.m)) ** bf.k
        
        results.append({
            'n': n,
            'm': bf.m,
            'k': bf.k,
            'fill_ratio': bf.get_fill_ratio(),
            'empirical_fp': empirical_fp,
            'theoretical_fp': theoretical_fp
        })
    
    return pd.DataFrame(results)


def experiment_varying_k(n=5000, epsilon=0.1):
    bf_base = BloomFilter(n=n, epsilon=epsilon)
    m = bf_base.m
    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]
    results = []
    
    for k in k_values:
        bf = BloomFilter(m=m, k=k)
        
        added_items = set()
        while len(added_items) < n:
            added_items.add(generate_random_string())
        for item in added_items:
            bf.add(item)
        
        test_items = set()
        while len(test_items) < n:
            item = generate_random_string()
            if item not in added_items:
                test_items.add(item)
        
        empirical_fp = measure_fp_rate(bf, added_items, test_items)
        theoretical_fp = (1 - math.exp(-k * n / m)) ** k
        
        results.append({
            'k': k,
            'm': m,
            'fill_ratio': bf.get_fill_ratio(),
            'empirical_fp': empirical_fp,
            'theoretical_fp': theoretical_fp
        })
    
    return pd.DataFrame(results)


def plot_epsilon_dependency(df, filename='plots/01_fp_vs_epsilon.png'):
    plt.figure(figsize=(11, 7))
    
    plt.plot(df['epsilon'], df['empirical_fp'], 
             color='#2E86AB', marker='o', markersize=8, 
             linewidth=2.5, label='Экспериментальный FP', markerfacecolor='#A23B72')
    plt.plot(df['epsilon'], df['theoretical_fp'], 
             color='#F18F01', marker='s', markersize=7, 
             linewidth=2, linestyle='--', label='Теоретический FP', 
             markerfacecolor='#C73E1D')
    
    plt.fill_between(df['epsilon'], 
                     [x*0.85 for x in df['theoretical_fp']], 
                     [x*1.15 for x in df['theoretical_fp']], 
                     color='#2E86AB', alpha=0.08, label='Допуск ±15%')
    
    plt.xlabel('Epsilon (целевой уровень ошибок)', fontsize=12, fontweight='500')
    plt.ylabel('Вероятность ложного срабатывания', fontsize=12, fontweight='500')
    plt.title('Зависимость FP rate от параметра epsilon\n(n = 1 000 элементов)', 
              fontsize=14, fontweight='bold', pad=15, color='#1a1a2e')
    
    plt.legend(fontsize=10, frameon=True, shadow=False, 
               edgecolor='#cccccc', facecolor='#ffffff')
    plt.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    plt.xticks(df['epsilon'], fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_color('#999999')
    plt.gca().spines['bottom'].set_color('#999999')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                facecolor='#fafafa', edgecolor='none')
    plt.close()


def plot_n_dependency(df, filename='plots/02_fp_vs_n.png'):
    fig, ax1 = plt.subplots(figsize=(11, 7))
    
    color_fp = '#2E86AB'
    color_fill = '#06A77D'
    color_theory = '#F18F01'
    
    ax1.plot(df['n'], df['empirical_fp'], 
             color=color_fp, marker='o', markersize=7, 
             linewidth=2.5, label='Экспериментальный FP', markerfacecolor='#A23B72')
    ax1.plot(df['n'], df['theoretical_fp'], 
             color=color_theory, marker='s', markersize=6, 
             linewidth=2, linestyle='--', label='Теоретический FP',
             markerfacecolor='#C73E1D')
    
    ax1.set_xlabel('Количество элементов (n)', fontsize=12, fontweight='500')
    ax1.set_ylabel('Вероятность ложного срабатывания', fontsize=12, fontweight='500', color='black')
    ax1.tick_params(axis='y', labelcolor='black', labelsize=10)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    ax2 = ax1.twinx()
    ax2.plot(df['n'], df['fill_ratio'], 
             color=color_fill, marker='^', markersize=7, 
             linewidth=2.5, linestyle='-.', label='Заполнение', 
             markerfacecolor='#047D5E', markeredgecolor='#034d39')
    ax2.set_ylabel('Заполнение битового массива (%)', fontsize=12, fontweight='500', color=color_fill)
    ax2.tick_params(axis='y', labelcolor=color_fill, labelsize=10)
    
    plt.title('Зависимость FP rate и заполнения от количества элементов\n(ε = 0,1)', 
              fontsize=14, fontweight='bold', pad=20, color='#1a1a2e')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, 
               loc='upper left', frameon=True, edgecolor='#cccccc')
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#999999')
    ax1.spines['bottom'].set_color('#999999')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                facecolor='#fafafa', edgecolor='none')
    plt.close()


def plot_k_dependency(df, filename='plots/03_fp_vs_k.png'):
    plt.figure(figsize=(11, 7))
    
    optimal_k = (df['m'].iloc[0] / 5000) * math.log(2)
    
    plt.plot(df['k'], df['empirical_fp'], 
             color='#6A4C93', marker='o', markersize=8, 
             linewidth=2.5, label='Экспериментальный FP', markerfacecolor='#4A2C6A')
    plt.plot(df['k'], df['theoretical_fp'], 
             color='#FF6B6B', marker='s', markersize=7, 
             linewidth=2, linestyle='--', label='Теоретический FP',
             markerfacecolor='#CC4444')
    
    plt.axvline(x=optimal_k, color='#FFD93D', linestyle=':', 
                linewidth=2.5, label=f'Оптимальное k ≈ {optimal_k:.1f}',
                alpha=0.9)
    
    plt.axvspan(optimal_k - 0.5, optimal_k + 0.5, 
                color='#FFD93D', alpha=0.15)
    
    plt.xlabel('Количество хеш-функций (k)', fontsize=12, fontweight='500')
    plt.ylabel('Вероятность ложного срабатывания', fontsize=12, fontweight='500')
    plt.title('Зависимость FP rate от количества хеш-функций\n(n = 5 000, ε = 0,1, m фиксировано)', 
              fontsize=14, fontweight='bold', pad=15, color='#1a1a2e')
    
    plt.legend(fontsize=10, frameon=True, shadow=False, 
               edgecolor='#cccccc', facecolor='#ffffff')
    plt.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    plt.xticks(df['k'], fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_color('#999999')
    plt.gca().spines['bottom'].set_color('#999999')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                facecolor='#fafafa', edgecolor='none')
    plt.close()


def print_summary_tables(df_eps, df_n, df_k):
    print("\n" + "="*90)
    print("ТАБЛИЦА 1: Зависимость от epsilon (n=1000)")
    print("="*90)
    display_df = df_eps.copy()
    display_df['epsilon'] = display_df['epsilon'].apply(lambda x: f"{x:.2f}")
    display_df['empirical_fp'] = display_df['empirical_fp'].apply(lambda x: f"{x*100:.2f}%")
    display_df['theoretical_fp'] = display_df['theoretical_fp'].apply(lambda x: f"{x*100:.2f}%")
    print(display_df[['epsilon', 'm', 'k', 'fill_ratio', 'empirical_fp', 'theoretical_fp']]
          .to_string(index=False))
    
    print("\n" + "="*90)
    print("ТАБЛИЦА 2: Зависимость от n (epsilon=0.1)")
    print("="*90)
    display_df = df_n.copy()
    display_df['empirical_fp'] = display_df['empirical_fp'].apply(lambda x: f"{x*100:.2f}%")
    display_df['theoretical_fp'] = display_df['theoretical_fp'].apply(lambda x: f"{x*100:.2f}%")
    print(display_df[['n', 'm', 'k', 'fill_ratio', 'empirical_fp', 'theoretical_fp']]
          .to_string(index=False))
    
    print("\n" + "="*90)
    print("ТАБЛИЦА 3: Зависимость от k (n=5000, epsilon=0.1)")
    print("="*90)
    display_df = df_k.copy()
    display_df['empirical_fp'] = display_df['empirical_fp'].apply(lambda x: f"{x*100:.2f}%")
    display_df['theoretical_fp'] = display_df['theoretical_fp'].apply(lambda x: f"{x*100:.2f}%")
    print(display_df[['k', 'm', 'fill_ratio', 'empirical_fp', 'theoretical_fp']]
          .to_string(index=False))


def main():

    
    print("\nЗапуск экспериментов...")
    df_epsilon = experiment_varying_epsilon(n=1000)
    df_n = experiment_varying_n(epsilon=0.1)
    df_k = experiment_varying_k(n=5000, epsilon=0.1)
    
    print_summary_tables(df_epsilon, df_n, df_k)
    
    print("\nПостроение графиков...")
    plot_epsilon_dependency(df_epsilon)
    plot_n_dependency(df_n)
    plot_k_dependency(df_k)
    


if __name__ == "__main__":
    random.seed(42)
    main()
