import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import time
import os
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
INPUT_COLS = [
    'Gongju_Bongryong', 'Gongju_Banpo',
    'Nonsan_Hwangsan',  'Nonsan_Toyang',
    'Nonsan_Yeonsan',   'Daejeon_Dugye',
    'Iksan_Hamyol',     'Iksan_Yeondong',
    'Nonsan_Dongsan',   'Wanju_Jangseon',
    'Geumsan',          'Jinan_Juyang',
    'water level'
]

N_CLUSTERS     = 10
PCA_COMPONENTS = 30

for xx in [1,2,3,4,5,6,7,8,9]:

    LEAD_TIME = xx

    BASE_DIR       = '02.data'
    MODEL_DIR      = '04.results'
    OUTPUT_DIR     = '04.results'
    SCALER_DIR     = '04.results'
    GMM_PREFIX     = f'gmm_K{N_CLUSTERS}_PC{PCA_COMPONENTS}_'
    PREFIX         = f'expert_K{N_CLUSTERS}_PC{PCA_COMPONENTS}_{LEAD_TIME}hr'
    LABEL_PATH     = os.path.join(OUTPUT_DIR, GMM_PREFIX + 'label_train.csv')
    LABEL_VAL_PATH = os.path.join(OUTPUT_DIR, GMM_PREFIX + 'label_validation.csv')

    SINGLE_MODEL_PATH = os.path.join(
        MODEL_DIR, f'pretrained_lstm_{LEAD_TIME}hr.keras'
    )

    TIME_STEPS = 7

    LEARNING_RATE = 0.00069
    EPOCHS        = 50
    PATIENCE      = 10
    BATCH_SIZE    = 1024
    RANDOM_STATE  = 42

    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    print(f"[{LEAD_TIME}hr] 데이터 로드 중...")

    X_train_df   = pd.read_csv(os.path.join(BASE_DIR, f'input_learn_{LEAD_TIME}hr.csv'),       encoding='utf-8-sig')
    y_train_df   = pd.read_csv(os.path.join(BASE_DIR, f'output_learn_{LEAD_TIME}hr.csv'),      encoding='utf-8-sig')
    labels_train = pd.read_csv(LABEL_PATH, encoding='utf-8-sig').values.flatten()

    X_val_df     = pd.read_csv(os.path.join(BASE_DIR, f'input_validation_{LEAD_TIME}hr.csv'),  encoding='utf-8-sig')
    y_val_df     = pd.read_csv(os.path.join(BASE_DIR, f'output_validation_{LEAD_TIME}hr.csv'), encoding='utf-8-sig')
    labels_val   = pd.read_csv(LABEL_VAL_PATH, encoding='utf-8-sig').values.flatten()

    print("스케일러 로드 중 (초기 LSTM과 동일 기준)...")
    scaler_X = joblib.load(os.path.join(SCALER_DIR, f'scaler_X_{LEAD_TIME}hr.pkl'))
    scaler_y = joblib.load(os.path.join(SCALER_DIR, f'scaler_y_{LEAD_TIME}hr.pkl'))

    X_train_all = scaler_X.transform(X_train_df[INPUT_COLS].values)
    y_train_all = scaler_y.transform(y_train_df.values).flatten()
    X_val_all   = scaler_X.transform(X_val_df[INPUT_COLS].values)
    y_val_all   = scaler_y.transform(y_val_df.values).flatten()

    print(f"  학습: X={X_train_all.shape}, y={y_train_all.shape}, label={labels_train.shape}")
    print(f"  검증: X={X_val_all.shape},   y={y_val_all.shape},   label={labels_val.shape}")

    BUFFER = TIME_STEPS - 1

    assert len(y_train_all) == len(labels_train), \
        f"학습 y/label 불일치: y={len(y_train_all)}, label={len(labels_train)}"
    assert len(y_val_all) == len(labels_val), \
        f"검증 y/label 불일치: y={len(y_val_all)}, label={len(labels_val)}"
    assert len(X_train_all) == len(y_train_all) + BUFFER, \
        f"학습 X 행 수 오류: X={len(X_train_all)}, y+buffer={len(y_train_all)+BUFFER}"
    assert len(X_val_all) == len(y_val_all) + BUFFER, \
        f"검증 X 행 수 오류: X={len(X_val_all)}, y+buffer={len(y_val_all)+BUFFER}"

    print(f"  라벨 분포 (train): {dict(zip(*np.unique(labels_train, return_counts=True)))}")
    print(f"  라벨 분포 (val)  : {dict(zip(*np.unique(labels_val,   return_counts=True)))}")

    def make_windows_by_label(X_all, y_all, labels, target_label, time_steps, buffer):
        target_indices = np.where(labels == target_label)[0]
        valid_indices  = target_indices[target_indices >= time_steps - 1]

        print(f"  라벨 {target_label}: 전체 {len(target_indices)}개 "
              f"→ 유효 {len(valid_indices)}개")

        X_list, y_list = [], []
        for i in valid_indices:
            X_list.append(X_all[i + buffer - time_steps + 1 : i + buffer + 1, :])
            y_list.append(y_all[i])

        return np.array(X_list), np.array(y_list)

    print(f"\n초기 LSTM 모델 로드: {SINGLE_MODEL_PATH}")
    single_model = load_model(SINGLE_MODEL_PATH)
    single_model.summary()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    LABEL_NAMES = {1: '갈수', 2: '저수', 3: '평수', 4: '풍수', 5:'5', 6 :'6', 7:'7', 8:'8', 9:'9', 10:'10'}
    results = []

    for label in range(1, N_CLUSTERS + 1):

        print(f"\n{'='*60}")
        print(f"  라벨 {label}({LABEL_NAMES[label]}) 추가학습 시작")
        print(f"{'='*60}")

        print("  [학습 데이터]")
        X_label_train, y_label_train = make_windows_by_label(
            X_train_all, y_train_all, labels_train, label, TIME_STEPS, BUFFER
        )
        print("  [검증 데이터]")
        X_label_val, y_label_val = make_windows_by_label(
            X_val_all, y_val_all, labels_val, label, TIME_STEPS, BUFFER
        )

        print(f"  학습: X={X_label_train.shape}, y={y_label_train.shape}")
        print(f"  검증: X={X_label_val.shape},   y={y_label_val.shape}")

        if len(X_label_train) == 0:
            print(f"  ⚠️  유효한 학습 데이터 없음 → 건너뜀")
            continue

        if len(X_label_val) == 0:
            print(f"  ⚠️  유효한 검증 데이터 없음 → validation_split=0.1 사용")
            validation_data = None
            val_split = 0.1
        else:
            validation_data = (X_label_val, y_label_val)
            val_split = 0.0

        expert_model = load_model(SINGLE_MODEL_PATH)
        expert_model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True
        )

        start = time.time()

        history = expert_model.fit(
            X_label_train, y_label_train,
            validation_data=validation_data,
            validation_split=val_split,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping],
            verbose=1
        )

        elapsed       = time.time() - start
        actual_epochs = len(history.history['loss'])

        print(f"\n  학습 완료 (에포크={actual_epochs}, 소요={elapsed:.1f}초)")
        print(f"  최종 train_loss: {history.history['loss'][-1]:.6f}")
        print(f"  최종 val_loss  : {history.history['val_loss'][-1]:.6f}")

        def calc_metrics(obs, pred):
            rmse = np.sqrt(np.mean((obs - pred) ** 2))
            mae  = np.mean(np.abs(obs - pred))
            nse  = 1 - np.sum((obs - pred) ** 2) / np.sum((obs - np.mean(obs)) ** 2)
            return rmse, mae, nse

        pred_train_scaled = expert_model.predict(X_label_train, verbose=0).flatten()
        pred_train = scaler_y.inverse_transform(pred_train_scaled.reshape(-1, 1)).flatten()
        obs_train  = scaler_y.inverse_transform(y_label_train.reshape(-1, 1)).flatten()
        rmse_tr, mae_tr, nse_tr = calc_metrics(obs_train, pred_train)
        print(f"  학습 성능: RMSE={rmse_tr:.4f}m, MAE={mae_tr:.4f}m, NSE={nse_tr:.4f}")

        pd.DataFrame({'observed': obs_train, 'predicted': pred_train}).to_csv(
            os.path.join(OUTPUT_DIR, f'{PREFIX}_label{label}_train.csv'),
            index=False, encoding='utf-8-sig'
        )

        if len(X_label_val) > 0:
            pred_val_scaled = expert_model.predict(X_label_val, verbose=0).flatten()
            pred_val = scaler_y.inverse_transform(pred_val_scaled.reshape(-1, 1)).flatten()
            obs_val  = scaler_y.inverse_transform(y_label_val.reshape(-1, 1)).flatten()
            rmse_val, mae_val, nse_val = calc_metrics(obs_val, pred_val)
            print(f"  검증 성능: RMSE={rmse_val:.4f}m, MAE={mae_val:.4f}m, NSE={nse_val:.4f}")
            pd.DataFrame({'observed': obs_val, 'predicted': pred_val}).to_csv(
                os.path.join(OUTPUT_DIR, f'{PREFIX}_label{label}_val.csv'),
                index=False, encoding='utf-8-sig'
            )
        else:
            rmse_val, mae_val, nse_val = None, None, None

        save_path = os.path.join(OUTPUT_DIR, f'{PREFIX}_label{label}.keras')
        expert_model.save(save_path)
        print(f"  저장 완료: {save_path}")

        results.append({
            'lead_time'       : LEAD_TIME,
            'label'           : label,
            'label_name'      : LABEL_NAMES[label],
            'n_train_samples' : len(X_label_train),
            'n_val_samples'   : len(X_label_val),
            'actual_epochs'   : actual_epochs,
            'final_train_loss': round(history.history['loss'][-1],    6),
            'final_val_loss'  : round(history.history['val_loss'][-1], 6),
            'elapsed_sec'     : round(elapsed, 1)
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(OUTPUT_DIR, f'{PREFIX}_summary.csv'),
        index=False, encoding='utf-8-sig'
    )

    print(f"\n{'='*60}")
    print(f"  {LEAD_TIME}hr 전문가 모델 추가학습 완료")
    print(f"{'='*60}")
    print(f"\n{'label':>8} | {'name':>6} | {'train_n':>8} | {'val_n':>7} | "
          f"{'epochs':>8} | {'train_loss':>12} | {'val_loss':>12}")
    print("-"*75)
    for _, row in results_df.iterrows():
        print(f"  {int(row['label']):>6} | {row['label_name']:>6} | "
              f"{int(row['n_train_samples']):>8} | {int(row['n_val_samples']):>7} | "
              f"{int(row['actual_epochs']):>8} | "
              f"{row['final_train_loss']:>12.6f} | {row['final_val_loss']:>12.6f}")
