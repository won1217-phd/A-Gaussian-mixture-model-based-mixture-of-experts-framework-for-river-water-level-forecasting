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

for xx in [1, 2, 3, 4, 5, 6, 7, 8, 9]:

    LEAD_TIME = xx

    BASE_DIR   = '02.data'
    MODEL_DIR  = '04.results'
    OUTPUT_DIR = '04.results'
    SCALER_DIR = '04.results'

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

    print(f"\n{'='*60}")
    print(f"  [{LEAD_TIME}hr] 전기간 미세조정 시작")
    print(f"{'='*60}")

    X_train_df = pd.read_csv(os.path.join(BASE_DIR, f'input_learn_{LEAD_TIME}hr.csv'),       encoding='utf-8-sig')
    y_train_df = pd.read_csv(os.path.join(BASE_DIR, f'output_learn_{LEAD_TIME}hr.csv'),      encoding='utf-8-sig')
    X_val_df   = pd.read_csv(os.path.join(BASE_DIR, f'input_validation_{LEAD_TIME}hr.csv'),  encoding='utf-8-sig')
    y_val_df   = pd.read_csv(os.path.join(BASE_DIR, f'output_validation_{LEAD_TIME}hr.csv'), encoding='utf-8-sig')
    X_test_df  = pd.read_csv(os.path.join(BASE_DIR, f'input_forecasting_{LEAD_TIME}hr.csv'), encoding='utf-8-sig')
    y_test_df  = pd.read_csv(os.path.join(BASE_DIR, f'output_forecasting_{LEAD_TIME}hr.csv'),encoding='utf-8-sig')

    print(f"  학습: X={X_train_df.shape}, y={y_train_df.shape}")
    print(f"  검증: X={X_val_df.shape},   y={y_val_df.shape}")
    print(f"  예측: X={X_test_df.shape},  y={y_test_df.shape}")

    print("스케일러 로드 중...")
    scaler_X = joblib.load(os.path.join(SCALER_DIR, f'scaler_X_{LEAD_TIME}hr.pkl'))
    scaler_y = joblib.load(os.path.join(SCALER_DIR, f'scaler_y_{LEAD_TIME}hr.pkl'))

    X_train_s = scaler_X.transform(X_train_df.values)
    y_train_s = scaler_y.transform(y_train_df.values).flatten()
    X_val_s   = scaler_X.transform(X_val_df.values)
    y_val_s   = scaler_y.transform(y_val_df.values).flatten()
    X_test_s  = scaler_X.transform(X_test_df.values)
    y_test_s  = scaler_y.transform(y_test_df.values).flatten()

    BUFFER = TIME_STEPS - 1

    def make_sliding_window(X_scaled, y_scaled, time_steps):
        n_out = len(y_scaled)
        X_list, y_list = [], []
        for i in range(n_out):
            X_list.append(X_scaled[i : i + time_steps, :])
            y_list.append(y_scaled[i])
        return np.array(X_list), np.array(y_list)

    print("슬라이딩 윈도우 적용 중...")
    X_train, y_train = make_sliding_window(X_train_s, y_train_s, TIME_STEPS)
    X_val,   y_val   = make_sliding_window(X_val_s,   y_val_s,   TIME_STEPS)
    X_test,  y_test  = make_sliding_window(X_test_s,  y_test_s,  TIME_STEPS)

    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    print(f"\n초기 LSTM 모델 로드: {SINGLE_MODEL_PATH}")
    model = load_model(SINGLE_MODEL_PATH)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True
    )

    print(f"\n전기간 미세조정 시작 (LR={LEARNING_RATE}, 최대 {EPOCHS} 에포크)...")
    start = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    elapsed       = time.time() - start
    actual_epochs = len(history.history['loss'])

    print(f"\n  미세조정 완료 (에포크={actual_epochs}, 소요={elapsed:.1f}초)")
    print(f"  최종 train_loss: {history.history['loss'][-1]:.6f}")
    print(f"  최종 val_loss  : {history.history['val_loss'][-1]:.6f}")

    def calc_metrics(obs, pred):
        rmse  = np.sqrt(np.mean((obs - pred) ** 2))
        mae   = np.mean(np.abs(obs - pred))
        denom = np.sum((obs - np.mean(obs)) ** 2)
        nse   = 1 - np.sum((obs - pred) ** 2) / denom if denom != 0 else np.nan
        pbias = np.sum(obs - pred) / np.sum(obs) * 100
        return rmse, mae, nse, pbias

    pred_train = scaler_y.inverse_transform(model.predict(X_train, verbose=0)).flatten()
    pred_val   = scaler_y.inverse_transform(model.predict(X_val,   verbose=0)).flatten()
    pred_test  = scaler_y.inverse_transform(model.predict(X_test,  verbose=0)).flatten()

    obs_train  = scaler_y.inverse_transform(y_train.reshape(-1,1)).flatten()
    obs_val    = scaler_y.inverse_transform(y_val.reshape(-1,1)).flatten()
    obs_test   = scaler_y.inverse_transform(y_test.reshape(-1,1)).flatten()

    rmse_tr,  mae_tr,  nse_tr,  pb_tr  = calc_metrics(obs_train, pred_train)
    rmse_val, mae_val, nse_val, pb_val = calc_metrics(obs_val,   pred_val)
    rmse_te,  mae_te,  nse_te,  pb_te  = calc_metrics(obs_test,  pred_test)

    print(f"\n  [성능 평가 - 역정규화 기준]")
    print(f"  학습: RMSE={rmse_tr:.4f}m, MAE={mae_tr:.4f}m, NSE={nse_tr:.4f}, PBIAS={pb_tr:.4f}%")
    print(f"  검증: RMSE={rmse_val:.4f}m, MAE={mae_val:.4f}m, NSE={nse_val:.4f}, PBIAS={pb_val:.4f}%")
    print(f"  예측: RMSE={rmse_te:.4f}m,  MAE={mae_te:.4f}m,  NSE={nse_te:.4f},  PBIAS={pb_te:.4f}%")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_path = os.path.join(OUTPUT_DIR, f'single_lstm_{LEAD_TIME}hr.keras')
    model.save(model_path)
    print(f"\n  모델 저장: {model_path}")

    for fname, obs, pred in [
        (os.path.join(OUTPUT_DIR, f'single_lstm_{LEAD_TIME}hr_train.csv'), obs_train, pred_train),
        (os.path.join(OUTPUT_DIR, f'single_lstm_{LEAD_TIME}hr_val.csv'),   obs_val,   pred_val),
        (os.path.join(OUTPUT_DIR, f'single_lstm_{LEAD_TIME}hr_test.csv'),  obs_test,  pred_test),
    ]:
        pd.DataFrame({'observed': obs, 'predicted': pred}).to_csv(
            fname, index=False, encoding='utf-8-sig')

    pd.DataFrame([{
        'lead_time'    : LEAD_TIME,
        'actual_epochs': actual_epochs,
        'train_RMSE'   : round(rmse_tr,  4), 'train_MAE': round(mae_tr,  4),
        'train_NSE'    : round(nse_tr,   4), 'train_PBIAS': round(pb_tr, 4),
        'val_RMSE'     : round(rmse_val, 4), 'val_MAE'  : round(mae_val, 4),
        'val_NSE'      : round(nse_val,  4), 'val_PBIAS': round(pb_val,  4),
        'test_RMSE'    : round(rmse_te,  4), 'test_MAE' : round(mae_te,  4),
        'test_NSE'     : round(nse_te,   4), 'test_PBIAS': round(pb_te, 4),
        'elapsed_sec'  : round(elapsed,  1)
    }]).to_csv(os.path.join(OUTPUT_DIR, f'single_lstm_{LEAD_TIME}hr_summary.csv'),
               index=False, encoding='utf-8-sig')

    print(f"\n{'='*60}")
    print(f"  {LEAD_TIME}hr 전기간 미세조정 완료")
    print(f"{'='*60}")
