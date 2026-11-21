import pandas as pd
import numpy as np
import os

DATA_DIR   = '02.data'
TRAIN_CSV  = os.path.join(DATA_DIR, 'raw_train.csv')
VAL_CSV    = os.path.join(DATA_DIR, 'raw_val.csv')
TEST_CSV   = os.path.join(DATA_DIR, 'raw_test.csv')

OUTPUT_DIR = DATA_DIR

TIME_STEPS = 7
LEAD_TIMES = [1, 2, 3, 4, 5, 6, 7, 8, 9]

TRAIN_OUT_START = '2006-01-01 00:00:00'
TRAIN_OUT_END   = '2023-12-31 23:00:00'
VAL_OUT_START   = '2024-01-01 00:00:00'
VAL_OUT_END     = '2024-12-31 23:00:00'
TEST_OUT_START  = '2025-01-01 00:00:00'
TEST_OUT_END    = '2025-12-31 23:00:00'

INPUT_COLS = [
    'Gongju_Bongryong', 'Gongju_Banpo',
    'Nonsan_Hwangsan',  'Nonsan_Toyang',
    'Nonsan_Yeonsan',   'Daejeon_Dugye',
    'Iksan_Hamyol',     'Iksan_Yeondong',
    'Nonsan_Dongsan',   'Wanju_Jangseon',
    'Geumsan',          'Jinan_Juyang',
    'water level'
]
OUTPUT_COL = 'water level'

print("데이터 로드 중...")
df_train = pd.read_csv(TRAIN_CSV, encoding='cp949')
df_val   = pd.read_csv(VAL_CSV,   encoding='cp949')
df_test  = pd.read_csv(TEST_CSV,  encoding='cp949')

df_train['시간'] = pd.to_datetime(df_train['시간'])
df_val['시간']   = pd.to_datetime(df_val['시간'])
df_test['시간']  = pd.to_datetime(df_test['시간'])

print(f"학습: {len(df_train)}개 | {df_train['시간'].iloc[0]} ~ {df_train['시간'].iloc[-1]}")
print(f"검증: {len(df_val)}개   | {df_val['시간'].iloc[0]} ~ {df_val['시간'].iloc[-1]}")
print(f"예측: {len(df_test)}개  | {df_test['시간'].iloc[0]} ~ {df_test['시간'].iloc[-1]}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_start_idx(df, datetime_str):
    mask = df['시간'] >= pd.to_datetime(datetime_str)
    return df[mask].index[0]

def get_end_idx(df, datetime_str):
    mask = df['시간'] <= pd.to_datetime(datetime_str)
    return df[mask].index[-1]

train_out_s = get_start_idx(df_train, TRAIN_OUT_START)
train_out_e = get_end_idx(df_train,   TRAIN_OUT_END)
val_out_s   = get_start_idx(df_val,   VAL_OUT_START)
val_out_e   = get_end_idx(df_val,     VAL_OUT_END)
test_out_s  = get_start_idx(df_test,  TEST_OUT_START)
test_out_e  = get_end_idx(df_test,    TEST_OUT_END)

print(f"\n[출력 범위 확인]")
print(f"학습: idx {train_out_s}({df_train['시간'].iloc[train_out_s]}) ~ idx {train_out_e}({df_train['시간'].iloc[train_out_e]})")
print(f"검증: idx {val_out_s}  ({df_val['시간'].iloc[val_out_s]}) ~ idx {val_out_e}  ({df_val['시간'].iloc[val_out_e]})")
print(f"예측: idx {test_out_s} ({df_test['시간'].iloc[test_out_s]}) ~ idx {test_out_e} ({df_test['시간'].iloc[test_out_e]})")

max_lead       = max(LEAD_TIMES)
max_buffer_req = TIME_STEPS + max_lead - 1

print(f"\n[버퍼 충분성 확인]")
print(f"  최대 필요 버퍼: TIME_STEPS({TIME_STEPS}) + max_lead({max_lead}) - 1 = {max_buffer_req}개")
print(f"  학습 버퍼 보유: {train_out_s}개 → {'✅ OK' if train_out_s >= max_buffer_req else '❌ 부족'}")
print(f"  검증 버퍼 보유: {val_out_s}개   → {'✅ OK' if val_out_s >= max_buffer_req else '❌ 부족'}")
print(f"  예측 버퍼 보유: {test_out_s}개  → {'✅ OK' if test_out_s >= max_buffer_req else '❌ 부족'}")

def make_lead_time_data(df, lead, out_s, out_e):
    in_s = out_s - (TIME_STEPS + lead - 1)
    in_e = out_e - lead

    assert in_s >= 0, f"버퍼 부족: in_s={in_s} < 0 (lead={lead})"

    X = df[INPUT_COLS].iloc[in_s : in_e + 1].reset_index(drop=True)
    y = df[[OUTPUT_COL]].iloc[out_s : out_e + 1].reset_index(drop=True)

    return X, y

print(f"\n출력 경로: {OUTPUT_DIR}")
print("="*70)

for lead in LEAD_TIMES:

    in_s_train = train_out_s - (TIME_STEPS + lead - 1)
    in_s_val   = val_out_s   - (TIME_STEPS + lead - 1)
    in_s_test  = test_out_s  - (TIME_STEPS + lead - 1)

    print(f"\n[{lead}hr 선행시간]")
    print(f"  학습 input 시작: {df_train['시간'].iloc[in_s_train]} "
          f"→ output 시작: {df_train['시간'].iloc[train_out_s]}")
    print(f"  검증 input 시작: {df_val['시간'].iloc[in_s_val]}   "
          f"→ output 시작: {df_val['시간'].iloc[val_out_s]}")
    print(f"  예측 input 시작: {df_test['시간'].iloc[in_s_test]}   "
          f"→ output 시작: {df_test['시간'].iloc[test_out_s]}")

    X_train, y_train = make_lead_time_data(df_train, lead, train_out_s, train_out_e)
    X_val,   y_val   = make_lead_time_data(df_val,   lead, val_out_s,   val_out_e)
    X_test,  y_test  = make_lead_time_data(df_test,  lead, test_out_s,  test_out_e)

    print(f"  학습: X={X_train.shape}, y={y_train.shape}")
    print(f"  검증: X={X_val.shape},   y={y_val.shape}")
    print(f"  예측: X={X_test.shape},  y={y_test.shape}")

    X_train.to_csv(os.path.join(OUTPUT_DIR, f'input_learn_{lead}hr.csv'),
                   index=False, encoding='utf-8-sig')
    y_train.to_csv(os.path.join(OUTPUT_DIR, f'output_learn_{lead}hr.csv'),
                   index=False, encoding='utf-8-sig')
    X_val.to_csv(os.path.join(OUTPUT_DIR, f'input_validation_{lead}hr.csv'),
                 index=False, encoding='utf-8-sig')
    y_val.to_csv(os.path.join(OUTPUT_DIR, f'output_validation_{lead}hr.csv'),
                 index=False, encoding='utf-8-sig')
    X_test.to_csv(os.path.join(OUTPUT_DIR, f'input_forecasting_{lead}hr.csv'),
                  index=False, encoding='utf-8-sig')
    y_test.to_csv(os.path.join(OUTPUT_DIR, f'output_forecasting_{lead}hr.csv'),
                  index=False, encoding='utf-8-sig')

    print(f"  → 저장 완료")

print("\n" + "="*70)
print("  전체 선행시간 데이터 생성 완료")
print("="*70)
print(f"\n생성 파일: {len(LEAD_TIMES) * 6}개 (선행시간 {len(LEAD_TIMES)}개 × 6개)")
