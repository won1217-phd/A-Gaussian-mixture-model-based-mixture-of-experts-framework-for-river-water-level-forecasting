import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

for xx in [1,2,3,4,5,6,7,8,9]:

    LEAD_TIME = xx

    BASE_DIR   = '02.data'
    OUTPUT_DIR = '04.results'
    SCALER_DIR = '04.results'

    N_LAYERS   = 4
    UNITS      = [31, 41, 56, 21]
    DROPOUT    = 0.1
    LR         = 0.0069
    EPOCHS     = 500
    PATIENCE   = 100
    BATCH_SIZE = 1024
    TIME_STEPS = 7
    N_FEATURES = 13

    RANDOM_STATE = 42

    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    print(f"[{LEAD_TIME}hr] 데이터 로드 중...")

    X_train_df = pd.read_csv(os.path.join(BASE_DIR, f'input_learn_{LEAD_TIME}hr.csv'),       encoding='utf-8-sig')
    y_train_df = pd.read_csv(os.path.join(BASE_DIR, f'output_learn_{LEAD_TIME}hr.csv'),      encoding='utf-8-sig')
    X_val_df   = pd.read_csv(os.path.join(BASE_DIR, f'input_validation_{LEAD_TIME}hr.csv'),  encoding='utf-8-sig')
    y_val_df   = pd.read_csv(os.path.join(BASE_DIR, f'output_validation_{LEAD_TIME}hr.csv'), encoding='utf-8-sig')
    X_test_df  = pd.read_csv(os.path.join(BASE_DIR, f'input_forecasting_{LEAD_TIME}hr.csv'), encoding='utf-8-sig')
    y_test_df  = pd.read_csv(os.path.join(BASE_DIR, f'output_forecasting_{LEAD_TIME}hr.csv'),encoding='utf-8-sig')

    print(f"  학습: X={X_train_df.shape}, y={y_train_df.shape}")
    print(f"  검증: X={X_val_df.shape},   y={y_val_df.shape}")
    print(f"  예측: X={X_test_df.shape},  y={y_test_df.shape}")

    print("정규화 적용 중...")
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train_df.values)
    y_train_scaled = scaler_y.fit_transform(y_train_df.values)

    X_val_scaled   = scaler_X.transform(X_val_df.values)
    y_val_scaled   = scaler_y.transform(y_val_df.values)
    X_test_scaled  = scaler_X.transform(X_test_df.values)
    y_test_scaled  = scaler_y.transform(y_test_df.values)

    os.makedirs(SCALER_DIR, exist_ok=True)
    joblib.dump(scaler_X, os.path.join(SCALER_DIR, f'scaler_X_{LEAD_TIME}hr.pkl'))
    joblib.dump(scaler_y, os.path.join(SCALER_DIR, f'scaler_y_{LEAD_TIME}hr.pkl'))
    print(f"  스케일러 저장 완료: {SCALER_DIR}")

    BUFFER = TIME_STEPS - 1

    def make_sliding_window(X_scaled, y_scaled, time_steps, buffer):
        n_out = len(y_scaled)
        X_list, y_list = [], []
        for i in range(n_out):
            X_list.append(X_scaled[i : i + time_steps, :])
            y_list.append(y_scaled[i])
        return np.array(X_list), np.array(y_list)

    print("슬라이딩 윈도우 적용 중...")
    X_train, y_train = make_sliding_window(X_train_scaled, y_train_scaled, TIME_STEPS, BUFFER)
    X_val,   y_val   = make_sliding_window(X_val_scaled,   y_val_scaled,   TIME_STEPS, BUFFER)
    X_test,  y_test  = make_sliding_window(X_test_scaled,  y_test_scaled,  TIME_STEPS, BUFFER)

    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    def build_model(time_steps, n_features, n_layers, units, dropout, lr):
        model = Sequential()
        for i in range(n_layers):
            return_seq = (i < n_layers - 1)
            if i == 0:
                model.add(LSTM(units[i],
                               return_sequences=return_seq,
                               input_shape=(time_steps, n_features)))
            else:
                model.add(LSTM(units[i], return_sequences=return_seq))
            model.add(Dropout(dropout))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
        return model

    print("\nLSTM 모델 구성...")
    model = build_model(TIME_STEPS, N_FEATURES, N_LAYERS, UNITS, DROPOUT, LR)
    model.summary()

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True
    )

    print(f"\n학습 시작 (최대 {EPOCHS} 에포크, patience={PATIENCE})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    actual_epochs = len(history.history['loss'])
    print(f"학습 완료 (실제 에포크: {actual_epochs})")

    def calc_metrics(obs, pred):
        rmse = np.sqrt(np.mean((obs - pred) ** 2))
        mae  = np.mean(np.abs(obs - pred))
        nse  = 1 - np.sum((obs - pred) ** 2) / np.sum((obs - np.mean(obs)) ** 2)
        return rmse, mae, nse

    pred_train = scaler_y.inverse_transform(model.predict(X_train, verbose=0))
    pred_val   = scaler_y.inverse_transform(model.predict(X_val,   verbose=0))
    pred_test  = scaler_y.inverse_transform(model.predict(X_test,  verbose=0))

    obs_train  = scaler_y.inverse_transform(y_train.reshape(-1, 1))
    obs_val    = scaler_y.inverse_transform(y_val.reshape(-1, 1))
    obs_test   = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    rmse_tr, mae_tr, nse_tr = calc_metrics(obs_train, pred_train)
    rmse_val, mae_val, nse_val = calc_metrics(obs_val, pred_val)
    rmse_te, mae_te, nse_te = calc_metrics(obs_test, pred_test)

    print(f"\n[{LEAD_TIME}hr 초기 LSTM 성능]")
    print(f"  학습: RMSE={rmse_tr:.4f}m, MAE={mae_tr:.4f}m, NSE={nse_tr:.4f}")
    print(f"  검증: RMSE={rmse_val:.4f}m, MAE={mae_val:.4f}m, NSE={nse_val:.4f}")
    print(f"  예측: RMSE={rmse_te:.4f}m, MAE={mae_te:.4f}m, NSE={nse_te:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_path = os.path.join(OUTPUT_DIR, f'pretrained_lstm_{LEAD_TIME}hr.keras')
    model.save(model_path)
    print(f"\n모델 저장: {model_path}")

    for fname, obs, pred in [
        (os.path.join(OUTPUT_DIR, f'pretrained_lstm_{LEAD_TIME}hr_train.csv'), obs_train, pred_train),
        (os.path.join(OUTPUT_DIR, f'pretrained_lstm_{LEAD_TIME}hr_val.csv'),   obs_val,   pred_val),
        (os.path.join(OUTPUT_DIR, f'pretrained_lstm_{LEAD_TIME}hr_test.csv'),  obs_test,  pred_test),
    ]:
        pd.DataFrame({'observed': obs.flatten(), 'predicted': pred.flatten()}).to_csv(
            fname, index=False, encoding='utf-8-sig')

    pd.DataFrame([{
        'lead_time'   : LEAD_TIME,
        'actual_epochs': actual_epochs,
        'train_RMSE'  : round(rmse_tr,  4), 'train_MAE': round(mae_tr,  4), 'train_NSE': round(nse_tr,  4),
        'val_RMSE'    : round(rmse_val, 4), 'val_MAE'  : round(mae_val, 4), 'val_NSE'  : round(nse_val, 4),
        'test_RMSE'   : round(rmse_te,  4), 'test_MAE' : round(mae_te,  4), 'test_NSE' : round(nse_te,  4),
    }]).to_csv(os.path.join(OUTPUT_DIR, f'pretrained_lstm_{LEAD_TIME}hr_summary.csv'), index=False, encoding='utf-8-sig')

    print(f"\n{'='*50}")
    print(f"  {LEAD_TIME}hr 초기 LSTM 학습 완료")
    print(f"{'='*50}")
