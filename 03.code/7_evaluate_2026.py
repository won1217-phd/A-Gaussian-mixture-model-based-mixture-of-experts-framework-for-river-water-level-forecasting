import os
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

LEADS  = range(1, 10)
COMBOS = [(10, 10),
          (4, 10), (4, 20), (4, 30),
          (6, 10), (6, 20), (6, 30),
          (8, 10), (8, 20), (8, 30),
          (10, 20), (10, 30)]
HIGH_STAGES = [4.0, 5.0]

BASE_DIR   = '02.data'
RESULT_DIR = '04.results'

TIME_STEPS = 7
BATCH_SIZE = 1024
INPUT_COLS = [
    'Gongju_Bongryong', 'Gongju_Banpo',
    'Nonsan_Hwangsan',  'Nonsan_Toyang',
    'Nonsan_Yeonsan',   'Daejeon_Dugye',
    'Iksan_Hamyol',     'Iksan_Yeondong',
    'Nonsan_Dongsan',   'Wanju_Jangseon',
    'Geumsan',          'Jinan_Juyang',
    'water level'
]
START = {'2025': pd.Timestamp('2025-01-01 00:00'), '2026': pd.Timestamp('2026-01-01 00:00')}
SPLIT = {'2025': 'forecasting', '2026': 'test2026'}

def gmm_file(K, PC, name):
    return os.path.join(RESULT_DIR, f'gmm_K{K}_PC{PC}_{name}')

def expert_file(K, PC, lead, label):
    return os.path.join(RESULT_DIR, f'expert_K{K}_PC{PC}_{lead}hr_label{label}.keras')

def moe_file(K, PC, lead, mode):
    return os.path.join(RESULT_DIR, f'moe_K{K}_PC{PC}_{lead}hr_{mode}.csv')

def single_file(lead, suffix):
    return os.path.join(RESULT_DIR, f'single_lstm_{lead}hr{suffix}')

def load_split(split, lead):
    X = pd.read_csv(os.path.join(BASE_DIR, f'input_{split}_{lead}hr.csv'), encoding='utf-8-sig')[INPUT_COLS]
    y = pd.read_csv(os.path.join(BASE_DIR, f'output_{split}_{lead}hr.csv'), encoding='utf-8-sig').values.flatten()
    return X.values, y

def make_windows(X_scaled, n_out):
    return np.array([X_scaled[i: i + TIME_STEPS, :] for i in range(n_out)])

def predict(model_path, X, scaler_y):
    model = load_model(model_path)
    p = model.predict(X, batch_size=BATCH_SIZE, verbose=0).reshape(-1, 1)
    tf.keras.backend.clear_session()
    return scaler_y.inverse_transform(p).flatten()

def router_proba(K, PC, split):
    scaler, pca, gmm = (joblib.load(gmm_file(K, PC, f)) for f in ['scaler.pkl', 'pca.pkl', 'model.pkl'])
    X1, y1 = load_split(split, 1)
    W = np.array([w.flatten() for w in make_windows(scaler.transform(X1), len(y1))])
    proba = gmm.predict_proba(pca.transform(W))
    label_map = pd.read_csv(gmm_file(K, PC, 'label_map.csv'), encoding='utf-8-sig')
    order = label_map.sort_values('label')['cluster'].values
    return proba[:, order]

def metrics(obs, pred):
    err = pred - obs
    r = {'RMSE': np.sqrt(np.mean(err ** 2)), 'MAE': np.mean(np.abs(err)),
         'NSE': 1 - np.sum(err ** 2) / np.sum((obs - obs.mean()) ** 2),
         'PBIAS': np.sum(obs - pred) / np.sum(obs) * 100}
    cc = np.corrcoef(obs, pred)[0, 1]
    r['KGE'] = 1 - np.sqrt((cc - 1) ** 2 + (pred.std() / obs.std() - 1) ** 2 + (pred.mean() / obs.mean() - 1) ** 2)
    ip = int(np.argmax(obs))
    r['peak_obs'], r['peak_err'] = obs[ip], abs(err[ip])
    for thr in HIGH_STAGES:
        k = obs >= thr
        r[f'n_ge{thr:g}'] = int(k.sum())
        r[f'RMSE_ge{thr:g}'] = np.sqrt(np.mean(err[k] ** 2)) if k.any() else np.nan
    return r

def saved_2025(lead):
    out = {}
    f = single_file(lead, '_test.csv')
    if os.path.exists(f):
        out['Single LSTM'] = pd.read_csv(f, encoding='utf-8-sig')['predicted'].values
    for K, PC in COMBOS:
        for mode in ['Hard', 'Soft']:
            f = moe_file(K, PC, lead, mode.lower())
            if os.path.exists(f):
                out[f'{mode} MoE K{K}_PC{PC}'] = pd.read_csv(f, encoding='utf-8-sig')['predicted'].values
    return out

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    proba = {(K, PC, yr): router_proba(K, PC, SPLIT[yr]) for K, PC in COMBOS for yr in ['2025', '2026']}
    for K, PC in COMBOS:
        saved = pd.read_csv(gmm_file(K, PC, 'proba_forecasting.csv'), encoding='utf-8-sig')
        diff = np.abs(saved[[f'proba_{i}' for i in range(1, K + 1)]].values - proba[(K, PC, '2025')]).max()
        assert diff < 1e-8, f'2025 라우터 재현 실패 K{K} PC{PC}: {diff}'
    lab = proba[(10, 10, '2026')].argmax(1) + 1
    print('2026 라우터 라벨 분포 (K=10, PC=10):', {l: int((lab == l).sum()) for l in range(1, 11)})

    rows, checks = [], []
    for lead in LEADS:
        if not os.path.exists(single_file(lead, '.keras')):
            print(f'\n[{lead}hr] Single LSTM 모델 없음 → 건너뜀')
            continue
        print(f'\n[{lead}hr]')
        scaler_X = joblib.load(os.path.join(RESULT_DIR, f'scaler_X_{lead}hr.pkl'))
        scaler_y = joblib.load(os.path.join(RESULT_DIR, f'scaler_y_{lead}hr.pkl'))

        data = {}
        for yr in ['2025', '2026']:
            X, y = load_split(SPLIT[yr], lead)
            data[yr] = (make_windows(scaler_X.transform(X), len(y)), y)

        X26, y26 = data['2026']
        pred = {'Single LSTM': predict(single_file(lead, '.keras'), X26, scaler_y)}
        for K, PC in COMBOS:
            paths = [expert_file(K, PC, lead, l) for l in range(1, K + 1)]
            if not all(os.path.exists(p) for p in paths):
                continue
            E = np.column_stack([predict(p, X26, scaler_y) for p in paths])
            P = proba[(K, PC, '2026')]
            pred[f'Hard MoE K{K}_PC{PC}'] = E[np.arange(len(E)), P.argmax(1)]
            pred[f'Soft MoE K{K}_PC{PC}'] = (E * P).sum(axis=1)
            if (K, PC) == (10, 10):
                pred['Uniform experts (1/n) K10_PC10'] = E.mean(axis=1)
                router_label = P.argmax(1) + 1
        out = pd.DataFrame({'time': pd.date_range(START['2026'], periods=len(y26), freq='h'), 'observed': y26})
        for name, p in pred.items():
            out[name] = p
        if 'Hard MoE K10_PC10' in pred:
            out['router_label K10_PC10'] = router_label
        out.to_csv(os.path.join(RESULT_DIR, f'pred_2026_{lead}hr.csv'), index=False, encoding='utf-8-sig')

        X25, y25 = data['2025']
        old = saved_2025(lead)
        p25 = predict(single_file(lead, '.keras'), X25, scaler_y)
        checks.append({'lead': lead, 'model': 'Single LSTM', 'max_abs_diff_m': np.abs(p25 - old['Single LSTM']).max()})
        if 'Hard MoE K10_PC10' in old:
            E25 = np.column_stack([predict(expert_file(10, 10, lead, l), X25, scaler_y)
                                   for l in range(1, 11)])
            h25 = E25[np.arange(len(E25)), proba[(10, 10, '2025')].argmax(1)]
            checks.append({'lead': lead, 'model': 'Hard MoE K10_PC10',
                           'max_abs_diff_m': np.abs(h25 - old['Hard MoE K10_PC10']).max()})
        print(f"  2025 재현 오차: Single {checks[-2]['max_abs_diff_m']:.1e} m, Hard MoE {checks[-1]['max_abs_diff_m']:.1e} m")

        for name in pred:
            K, PC = (int(name.split('K')[-1].split('_')[0]), int(name.split('PC')[-1])) if 'K' in name.split()[-1] else ('-', '-')
            series = {'2026': pred[name]}
            if name in old:
                series['2025'] = old[name]
            for yr, p in series.items():
                obs = y25 if yr == '2025' else y26
                rows.append({'year': yr, 'lead': lead, 'model': name.split(' K')[0], 'K': K, 'PC': PC, **metrics(obs, p)})
            if '2025' in series:
                rows.append({'year': '2025+2026', 'lead': lead, 'model': name.split(' K')[0], 'K': K, 'PC': PC,
                             **metrics(np.concatenate([y25, y26]), np.concatenate([series['2025'], series['2026']]))})
        r = pd.DataFrame(rows)
        sel = r[(r['lead'] == lead) & (r['model'].isin(['Single LSTM', 'Hard MoE'])) & (r['K'].isin(['-', 10])) & (r['PC'].isin(['-', 10]))]
        print(sel[['year', 'model', 'RMSE', 'MAE', 'NSE', 'KGE', 'peak_err', 'RMSE_ge4', 'RMSE_ge5']].round(4).to_string(index=False))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULT_DIR, 'metrics_2025_2026.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(checks).to_csv(os.path.join(RESULT_DIR, 'check_2025_reproduction.csv'), index=False, encoding='utf-8-sig')
    print(f'\n저장: {RESULT_DIR}')

if __name__ == '__main__':
    main()
