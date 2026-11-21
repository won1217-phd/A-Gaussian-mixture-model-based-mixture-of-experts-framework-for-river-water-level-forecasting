import os

import numpy as np
import pandas as pd

DATA_DIR   = '02.data'
RESULT_DIR = '04.results'
SINGLE     = os.path.join(RESULT_DIR, 'single_lstm_{h}hr_test.csv')
HARD       = os.path.join(RESULT_DIR, 'moe_K10_PC10_{h}hr_hard.csv')
PRED2026   = os.path.join(RESULT_DIR, 'pred_2026_{h}hr.csv')
OUT_CSV    = os.path.join(RESULT_DIR, 'regime_2025_2026.csv')

LEADS   = range(1, 10)
EDGES   = [1.79, 2.01, 2.10]
REGIMES = ['DWL', 'LWL', 'OWL', 'AWL']

def regime(y):
    return np.digitize(np.round(y, 2), EDGES, right=True) + 1

y25 = pd.read_csv(SINGLE.format(h=1), encoding='utf-8-sig')['observed'].values
lab = pd.read_csv(os.path.join(DATA_DIR, 'Label_forecasting.csv'), encoding='utf-8-sig').values.ravel()
assert (regime(y25) == lab).all(), '2025 구간이 Label_forecasting.csv와 다름'

rows = []
for h in LEADS:
    s25 = pd.read_csv(SINGLE.format(h=h), encoding='utf-8-sig')
    m25 = pd.read_csv(HARD.format(h=h), encoding='utf-8-sig')
    p26 = pd.read_csv(PRED2026.format(h=h), encoding='utf-8-sig')
    sets = {'2025': (s25['observed'].values, s25['predicted'].values, m25['predicted'].values),
            '2026': (p26['observed'].values, p26['Single LSTM'].values, p26['Hard MoE K10_PC10'].values)}
    sets['2025+2026'] = tuple(np.concatenate([sets['2025'][i], sets['2026'][i]]) for i in range(3))
    for yr, (obs, single, moe) in sets.items():
        g = regime(obs)
        for k, name in enumerate(REGIMES, 1):
            idx = g == k
            for model, pred in [('Single LSTM', single), ('Hard MoE', moe)]:
                err = pred[idx] - obs[idx]
                rows.append({'year': yr, 'lead': h, 'regime': name, 'n': int(idx.sum()), 'model': model,
                             'RMSE': np.sqrt(np.mean(err ** 2)), 'MAE': np.mean(np.abs(err))})

res = pd.DataFrame(rows)
res.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
piv = res.pivot_table(index=['year', 'regime', 'lead'], columns='model', values='RMSE')
imp = ((piv['Single LSTM'] - piv['Hard MoE']) / piv['Single LSTM'] * 100).unstack('lead').round(2)
pd.set_option('display.width', 200)
print('RMSE 개선율 (%)')
print(imp.to_string())
print('\n저장:', OUT_CSV)
