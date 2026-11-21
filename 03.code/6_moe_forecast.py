import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

N_CLUSTERS  = 10
PCA_COMPONENTS = 30
LABELS      = list(range(1, N_CLUSTERS + 1))
LABEL_NAMES = {1: '갈수', 2: '저수', 3: '평수', 4: '풍수', 5 : '5', 6 :'6', 7: '7', 8:'8', 9:'9', 10:'10'}

for xx in [1,2,3,4,5,6,7,8,9]:

    LEAD_TIME  = xx
    TIME_STEPS = 7

    DATA_DIR       = '02.data'
    RESULT_DIR     = '04.results'
    GMM_PREFIX     = os.path.join(RESULT_DIR, f'gmm_K{N_CLUSTERS}_PC{PCA_COMPONENTS}_')
    EXPERT_PREFIX  = os.path.join(RESULT_DIR, f'expert_K{N_CLUSTERS}_PC{PCA_COMPONENTS}_{LEAD_TIME}hr')
    MOE_PREFIX     = os.path.join(RESULT_DIR, f'moe_K{N_CLUSTERS}_PC{PCA_COMPONENTS}_{LEAD_TIME}hr')

    INPUT_PATH     = os.path.join(DATA_DIR, f'input_forecasting_{LEAD_TIME}hr.csv')
    OUTPUT_PATH    = os.path.join(DATA_DIR, f'output_forecasting_{LEAD_TIME}hr.csv')
    LABEL_PATH     = GMM_PREFIX + 'label_forecasting.csv'
    PROBA_PATH     = GMM_PREFIX + 'proba_forecasting.csv'
    SCALER_Y_PATH  = os.path.join(RESULT_DIR, f'scaler_y_{LEAD_TIME}hr.pkl')
    SCALER_X_PATH  = os.path.join(RESULT_DIR, f'scaler_X_{LEAD_TIME}hr.pkl')
    GMM_MODEL_PATH = GMM_PREFIX + 'model.pkl'
    GMM_PCA_PATH   = GMM_PREFIX + 'pca.pkl'
    GMM_SCALER     = GMM_PREFIX + 'scaler.pkl'
    LABEL_MAP_PATH = GMM_PREFIX + 'label_map.csv'

    INPUT_COLS = [
        'Gongju_Bongryong', 'Gongju_Banpo',
        'Nonsan_Hwangsan',  'Nonsan_Toyang',
        'Nonsan_Yeonsan',   'Daejeon_Dugye',
        'Iksan_Hamyol',     'Iksan_Yeondong',
        'Nonsan_Dongsan',   'Wanju_Jangseon',
        'Geumsan',          'Jinan_Juyang',
        'water level'
    ]

    print(f"\n{'='*60}")
    print(f"  GMM MoE {LEAD_TIME}hr 예측 시작")
    print(f"{'='*60}")

    X_raw  = pd.read_csv(INPUT_PATH,  encoding='utf-8-sig')
    y_obs  = pd.read_csv(OUTPUT_PATH, encoding='utf-8-sig').values.flatten()
    labels = pd.read_csv(LABEL_PATH,  encoding='utf-8-sig').values.flatten()
    n_data = len(y_obs)
    print(f"  데이터: {n_data}개")
    print(f"  라벨 분포: {dict(zip(*np.unique(labels, return_counts=True)))}")

    if os.path.exists(PROBA_PATH):
        print("\nGMM 확률 파일 로드 중...")
        proba_df     = pd.read_csv(PROBA_PATH, encoding='utf-8-sig')
        router_proba = proba_df[[f'proba_{i}' for i in LABELS]].values
        labels_pred  = proba_df['label'].values
        print(f"  GMM 확률 로드 완료: {router_proba.shape}")

    else:
        print("\nGMM 확률 파일 없음 → GMM 모델로 직접 계산 중...")

        gmm_data  = joblib.load(GMM_MODEL_PATH)
        pca       = joblib.load(GMM_PCA_PATH)
        scaler_gm = joblib.load(GMM_SCALER)

        X_scaled_gm = scaler_gm.transform(X_raw[INPUT_COLS].values)
        X_sw = np.array([X_scaled_gm[i : i + TIME_STEPS, :].flatten()
                         for i in range(n_data)])
        X_pca = pca.transform(X_sw)

        router_proba_raw = gmm_data.predict_proba(X_pca)
        labels_pred_raw  = np.argmax(router_proba_raw, axis=1)

        label_map_df = pd.read_csv(LABEL_MAP_PATH, encoding='utf-8-sig')
        label_map    = dict(zip(label_map_df['cluster'], label_map_df['label']))
        labels_pred  = np.array([label_map[c] for c in labels_pred_raw])

        cluster_order = [None] * N_CLUSTERS
        for c, l in label_map.items():
            cluster_order[l-1] = c
        router_proba = router_proba_raw[:, cluster_order]

        print(f"  GMM 확률 계산 완료: {router_proba.shape}")

    router_acc = accuracy_score(labels, labels_pred)
    router_f1  = f1_score(labels, labels_pred, average='macro')
    router_cm  = confusion_matrix(labels, labels_pred)
    unique_labels = sorted(np.unique(np.concatenate([labels, labels_pred])))

    print(f"  예측 라벨 분포: {dict(zip(*np.unique(labels_pred, return_counts=True)))}")
    print(f"  실제 라벨 분포: {dict(zip(*np.unique(labels,      return_counts=True)))}")
    print(f"  Accuracy: {router_acc:.4f}, F1: {router_f1:.4f}")

    scaler_ex   = joblib.load(SCALER_X_PATH)
    X_scaled_ex = scaler_ex.transform(X_raw[INPUT_COLS].values)
    X_expert    = np.array([X_scaled_ex[i : i + TIME_STEPS, :]
                             for i in range(n_data)])

    print("\n전문가 모델 로드 중...")
    scaler_y = joblib.load(SCALER_Y_PATH)
    experts  = {}
    for label in LABELS:
        experts[label] = load_model(f'{EXPERT_PREFIX}_label{label}.keras')
        print(f"  {EXPERT_PREFIX}_label{label}.keras 로드 완료")

    print("\n[Hard MoE] 예측 중...")
    y_hard_scaled = np.zeros(n_data)
    for label in LABELS:
        idx = np.where(labels_pred == label)[0]
        if len(idx) == 0: continue
        y_hard_scaled[idx] = experts[label].predict(
            X_expert[idx], verbose=0).flatten()
        print(f"  라벨 {label}({LABEL_NAMES[label]}): {len(idx)}개 예측 완료")

    y_hard = scaler_y.inverse_transform(
        y_hard_scaled.reshape(-1, 1)).flatten()

    print("\n[Soft MoE] 전문가별 전체 예측 중...")
    expert_preds = np.zeros((n_data, N_CLUSTERS))
    for j, label in enumerate(LABELS):
        expert_preds[:, j] = experts[label].predict(
            X_expert, verbose=0).flatten()
        print(f"  expert_{label}({LABEL_NAMES[label]}) 예측 완료")

    y_soft_scaled = np.sum(router_proba * expert_preds, axis=1)
    y_soft = scaler_y.inverse_transform(
        y_soft_scaled.reshape(-1, 1)).flatten()

    def calc_metrics(obs, pred):
        rmse  = np.sqrt(np.mean((obs - pred) ** 2))
        mae   = np.mean(np.abs(obs - pred))
        denom = np.sum((obs - np.mean(obs)) ** 2)
        nse   = 1 - np.sum((obs-pred)**2) / denom if denom != 0 else np.nan
        pbias = np.sum(obs - pred) / np.sum(obs) * 100
        return rmse, mae, nse, pbias

    rmse_h, mae_h, nse_h, pb_h = calc_metrics(y_obs, y_hard)
    rmse_s, mae_s, nse_s, pb_s = calc_metrics(y_obs, y_soft)

    print(f"\n{'='*60}")
    print(f"  GMM 라우터 {LEAD_TIME}hr 성능 비교")
    print(f"{'='*60}")
    print(f"  {'':12} {'RMSE':>8} {'MAE':>8} {'NSE':>8} {'PBIAS':>8}")
    print(f"  {'Hard MoE':12} {rmse_h:>8.4f} {mae_h:>8.4f} "
          f"{nse_h:>8.4f} {pb_h:>8.4f}")
    print(f"  {'Soft MoE':12} {rmse_s:>8.4f} {mae_s:>8.4f} "
          f"{nse_s:>8.4f} {pb_s:>8.4f}")

    for mode, y_pred in [('Hard MoE', y_hard), ('Soft MoE', y_soft)]:
        print(f"\n  [{mode} 구간별 성능]")
        for label in LABELS:
            idx = np.where(labels == label)[0]
            if len(idx) == 0: continue
            r, m, n, p = calc_metrics(y_obs[idx], y_pred[idx])
            print(f"    라벨{label}({LABEL_NAMES[label]}): "
                  f"RMSE={r:.4f}, MAE={m:.4f}, NSE={n:.4f} (n={len(idx)})")

    os.makedirs(RESULT_DIR, exist_ok=True)

    pd.DataFrame({
        'observed'    : y_obs,
        'predicted'   : y_hard,
        'true_label'  : labels,
        'router_label': labels_pred
    }).to_csv(MOE_PREFIX + '_hard.csv',
              index=False, encoding='utf-8-sig')

    soft_df = pd.DataFrame({
        'observed'    : y_obs,
        'predicted'   : y_soft,
        'true_label'  : labels,
        'router_label': labels_pred
    })
    for j in range(N_CLUSTERS):
        soft_df[f'proba_{j+1}'] = router_proba[:, j]
    soft_df.to_csv(MOE_PREFIX + '_soft.csv',
                   index=False, encoding='utf-8-sig')

    router_df = pd.DataFrame({
        'true_label'  : labels,
        'router_label': labels_pred
    })
    for j in range(N_CLUSTERS):
        router_df[f'proba_{j+1}'] = router_proba[:, j]
    router_df.to_csv(MOE_PREFIX + '_router.csv',
                     index=False, encoding='utf-8-sig')

    pd.DataFrame(router_cm,
                 index=[f'true_{i}'  for i in unique_labels],
                 columns=[f'pred_{i}' for i in unique_labels]
    ).to_csv(MOE_PREFIX + '_confusion_matrix.csv',
             encoding='utf-8-sig')

    pd.DataFrame([
        {'router':'GMM','lead_time':LEAD_TIME,'mode':'Hard',
         'RMSE':round(rmse_h,4),'MAE':round(mae_h,4),
         'NSE':round(nse_h,4),'PBIAS':round(pb_h,4),
         'router_acc':round(router_acc,4),'router_f1':round(router_f1,4)},
        {'router':'GMM','lead_time':LEAD_TIME,'mode':'Soft',
         'RMSE':round(rmse_s,4),'MAE':round(mae_s,4),
         'NSE':round(nse_s,4),'PBIAS':round(pb_s,4),
         'router_acc':round(router_acc,4),'router_f1':round(router_f1,4)}
    ]).to_csv(MOE_PREFIX + '_summary.csv',
              index=False, encoding='utf-8-sig')

    print(f"\n  결과 저장 완료: {RESULT_DIR}")
