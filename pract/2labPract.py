import numpy as np
import pandas as pd
import os
import warnings

def allocate_feature_types(n_features: int) -> dict:

    if n_features < 4:
        return {'binary': 0, 'nominal': 0, 'ordinal': 0, 'quantitative': n_features}
    
    alloc = {'binary': 1, 'nominal': 1, 'ordinal': 1, 'quantitative': 1}
    remaining = n_features - 4
    
    alloc['quantitative'] += remaining // 2
    alloc['nominal'] += remaining // 4
    alloc['ordinal'] += remaining - (remaining // 2 + remaining // 4)
    
    return alloc

def generate_state_features(n_samples: int, alloc: dict, prefix: str) -> pd.DataFrame:

    cols_data = {}
    
    for i in range(alloc['binary']):
        cols_data[f"{prefix}_bin_{i}"] = np.random.choice([0, 1], size=n_samples)
        
    nom_cats = ['circle', 'square', 'triangle', 'hexagon', 'star']
    for i in range(alloc['nominal']):
        cols_data[f"{prefix}_nom_{i}"] = np.random.choice(nom_cats, size=n_samples)
        
    ord_cats = ['low', 'medium', 'high']
    for i in range(alloc['ordinal']):
        cols_data[f"{prefix}_ord_{i}"] = np.random.choice(ord_cats, size=n_samples)
        
    for i in range(alloc['quantitative']):
        cols_data[f"{prefix}_qnt_{i}"] = np.round(np.random.uniform(0, 100, size=n_samples), 2)
        
    return pd.DataFrame(cols_data)

def custom_collision_function(df: pd.DataFrame) -> pd.Series:
    ord_map = {'low': 1, 'medium': 2, 'high': 3}
    
    q1_cols = [c for c in df.columns if c.startswith('obj1_qnt')]
    q2_cols = [c for c in df.columns if c.startswith('obj2_qnt')]
    
    x1 = df[q1_cols[0]].values
    y1 = df[q1_cols[1]].values if len(q1_cols) > 1 else x1
    x2 = df[q2_cols[0]].values
    y2 = df[q2_cols[1]].values if len(q2_cols) > 1 else x2
    
    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    m1 = df.get('obj1_bin_0', pd.Series(0, index=df.index)).values
    m2 = df.get('obj2_bin_0', pd.Series(0, index=df.index)).values
    is_moving = (m1 | m2).astype(bool)
    
    s1 = df.get('obj1_nom_0', pd.Series('unknown', index=df.index)).values
    s2 = df.get('obj2_nom_0', pd.Series('unknown', index=df.index)).values
    shape_diff = (s1 != s2).astype(int)
    
    o1 = df.get('obj1_ord_0', pd.Series('low', index=df.index)).map(ord_map).fillna(1).values
    o2 = df.get('obj2_ord_0', pd.Series('low', index=df.index)).map(ord_map).fillna(1).values
    urgency = o1 + o2
    
    condition = (dist < 60.0) | (is_moving & (urgency >= 3) & (shape_diff == 1))
    
    result = np.where(condition, "Да", "Нет")
    
    if np.mean(result == "Да") < 0.15:
        n_add = int(len(result) * (0.20 - np.mean(result == "Да")))
        if n_add > 0:
            idx = np.random.choice(np.where(result == "Нет")[0], size=min(n_add, np.sum(result == "Нет")), replace=False)
            result[idx] = "Да"
    
    return pd.Series(result, index=df.index, name="Collision")

def generate_all_datasets():
    sample_configs = [50, 320, 600, 1200]
    feature_configs = [6, 10, 12]
    
    os.makedirs("datasets", exist_ok=True)
    np.random.seed(42)
    
    ds_id = 1
    for n_samples in sample_configs:
        for n_features in feature_configs:
            
            alloc = allocate_feature_types(n_features)
            
            df_obj1 = generate_state_features(n_samples, alloc, 'obj1')
            df_obj2 = generate_state_features(n_samples, alloc, 'obj2')
            
            df = pd.concat([df_obj1, df_obj2], axis=1)
            df['Collision'] = custom_collision_function(df)
            
            print(df['Collision'].value_counts())

            filename = f"datasets/dataset_{ds_id:02d}_samples{n_samples}_feat{n_features}.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            
            ds_id += 1


df_summary = generate_all_datasets()


import os
import glob
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import re

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

F1_SCORER = make_scorer(f1_score, pos_label='Да', zero_division=0)

# LogisticRegression: базовый линейный классификатор. Работает быстро, 
# даёт вероятности, хорошо масштабируется, служит сильным baseline'ом.
# RandomForestClassifier: ансамбль деревьев. Устойчив к переобучению, 
# не требует масштабирования, автоматически обрабатывает нелинейности.
# DecisionTreeClassifier: интерпретируемый, быстрый, хорошо работает 
# с табличными данными и смешанными типами признаков после кодирования.
# KNeighborsClassifier (KNN): непараметрический метод, учитывает локальную 
# структуру данных. Требует масштабирования, но эффективен на малых выборках.

MODELS = {
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
    'DecisionTree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    'KNN': KNeighborsClassifier(n_jobs=-1)
}

# Основная метрика: F1-score (для класса "Да")
# Обоснование: В задачах обнаружения коллизий класс "Да" обычно миноритарный. 
# Accuracy может вводить в заблуждение (95% точности при 5% коллизий бесполезны). 
# F1 балансирует Precision и Recall, фокусируясь на корректном обнаружении именно коллизий.
PRIMARY_METRIC = 'f1'
POS_LABEL = 'Да'

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    bin_cols = [c for c in df.columns if re.match(r'obj\d+_bin_\d+', c)]
    nom_cols = [c for c in df.columns if re.match(r'obj\d+_nom_\d+', c)]
    ord_cols = [c for c in df.columns if re.match(r'obj\d+_ord_\d+', c)]
    qnt_cols = [c for c in df.columns if re.match(r'obj\d+_qnt_\d+', c)]
    
    transformers = []
    if bin_cols: 
        transformers.append(('bin', 'passthrough', bin_cols))
    if qnt_cols: 
        transformers.append(('qnt', StandardScaler(), qnt_cols))
    if nom_cols: 
        transformers.append(('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nom_cols))
    if ord_cols: 
        transformers.append(('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ord_cols))
    
    return ColumnTransformer(transformers, remainder='drop')

from sklearn.model_selection import StratifiedKFold, cross_validate

def train_and_evaluate_all():
    dataset_files = sorted(glob.glob('datasets/*.csv'))
    results = []
    
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for fpath in dataset_files:
        df = pd.read_csv(fpath)
        X = df.drop('Collision', axis=1)
        y = df['Collision']
        
        match = re.search(r'samples(\d+)', os.path.basename(fpath))
        n_samples = int(match.group(1)) if match else len(df)
        
        dataset_res = {'file': os.path.basename(fpath), 'n_samples': n_samples}
        
        preprocessor = build_preprocessor(X)
        
        for name, model in MODELS.items():
            pipe = Pipeline([('pre', preprocessor), ('clf', model)])
            
            cv_out = cross_validate(
                pipe, X, y, cv=cv_strategy,
                scoring={'f1': F1_SCORER, 'acc': 'accuracy'},
                n_jobs=-1, return_train_score=False
            )
            
            dataset_res[f'{name}_f1'] = cv_out['test_f1'].mean()
            dataset_res[f'{name}_f1_std'] = cv_out['test_f1'].std()
            dataset_res[f'{name}_acc'] = cv_out['test_acc'].mean()
            
        results.append(dataset_res)
        rf_f1, rf_std = dataset_res['RandomForest_f1'], dataset_res['RandomForest_f1_std']
        
        print(f"{dataset_res['file']} | Samples: {n_samples:4d} | F1(RF): {rf_f1:.3f} ± {rf_std:.3f}")

    df_res = pd.DataFrame(results)
    
    avg_all = {f'{m}_f1': df_res[f'{m}_f1'].mean() for m in MODELS}
    avg_all['n_datasets'] = len(df_res)
    
    small_mask = df_res['n_samples'] <= 320
    avg_small = {f'{m}_f1': df_res.loc[small_mask, f'{m}_f1'].mean() for m in MODELS} if small_mask.any() else {m: np.nan for m in MODELS}
    avg_small['n_datasets'] = small_mask.sum()
    
    print(f"{'Модель':<25} | {'Все 12 датасетов':<20} | {'Малые (<320 строк)':<20}")
    for m in MODELS:
        all_val = f"{avg_all[f'{m}_f1']:.3f}"
        small_val = f"{avg_small[f'{m}_f1']:.3f}" if not np.isnan(avg_small[f'{m}_f1']) else "N/A"
        print(f"{m:<25} | {all_val:<20} | {small_val:<20}")
        
    return df_res

# Обоснование
# 1. LogisticRegression: O(N*D) память, быстрая оптимизация, минимум оверхеда.
# 2. GaussianNB: O(1) хранение модели после обучения, считает только mean/var.
# 3. DecisionTreeClassifier: Не требует масштабирования, хранит только пороги разбиений, 
#    предсказание за O(log N).
FAST_MODELS = {
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    'GaussianNB': GaussianNB(),
    'DecisionTree': DecisionTreeClassifier(random_state=42, class_weight='balanced')
}

def tune_fast_models(df_res):
    smallest = df_res.loc[df_res['n_samples'].idxmin(), 'file']
    largest = df_res.loc[df_res['n_samples'].idxmax(), 'file']
    
    grids = {
        'LogisticRegression': {'clf__C': [0.1, 1.0, 10.0]},
        'GaussianNB': {'clf__var_smoothing': [1e-9, 1e-7, 1e-5]},
        'DecisionTree': {'clf__max_depth': [3, 5, None], 'clf__min_samples_split': [2, 5, 10]}
    }
    
    cv_tune = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    timing_report = []
    best_models = {}
    
    for ds_name in [smallest, largest]:
        df = pd.read_csv(f'datasets/{ds_name}')
        X = df.drop('Collision', axis=1)
        y = df['Collision']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        preprocessor = build_preprocessor(X_train)
        
        for name, model in FAST_MODELS.items():
            pipe = Pipeline([('pre', preprocessor), ('clf', model)])
            
            grid = GridSearchCV(
                pipe, grids[name], 
                cv=cv_tune, scoring=F1_SCORER, n_jobs=-1, refit=True
            )
            
            t0 = time.perf_counter()
            grid.fit(X_train, y_train)
            t1 = time.perf_counter()
            
            elapsed = t1 - t0
            f1_test = f1_score(y_test, grid.predict(X_test), pos_label='Да', zero_division=0)
            
            timing_report.append({
                'Dataset': os.path.basename(ds_name),
                'Model': name,
                'Best_Params': grid.best_params_,
                'Time_sec': round(elapsed, 4),
                'Test_F1': round(f1_test, 3)
            })
            
            if ds_name == largest:
                best_models[name] = grid.best_estimator_
                
    df_time = pd.DataFrame(timing_report)
    print("\nВремя и результаты подбора гиперпараметров:")
    print(df_time.to_string(index=False))
    
    return best_models

def save_models(best_models: dict):
    os.makedirs('models', exist_ok=True)
    saved_paths = []
    for name, model in best_models.items():
        path = f'models/{name}_tuned.joblib'
        joblib.dump(model, path)
        saved_paths.append(path)

df_results = train_and_evaluate_all()
best_models = tune_fast_models(df_results)
save_models(best_models)
