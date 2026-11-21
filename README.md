# GMM-based mixture of experts (MoE) for river water level forecasting

Code and data for:

> Lee, W. J., & Lee, E. H. A Gaussian mixture model-based mixture of experts framework for river water level forecasting. *Journal of Hydrology* (under review).

## Overview

- Study site: Nonsan Bridge water level station, Nonsan River, South Korea
- Target: hourly water level (EL. m) at lead times of 1–9 h
- Inputs: hourly rainfall at 12 gauges and the water level at the station over a 7-h window (t−6 to t), i.e., 7 time steps × 13 variables
- Periods: training 2006–2023, validation 2024, test 2025, additional test January–October 2026
- Models
  - Pre-trained LSTM: 4 hidden layers (31, 41, 56, 21 nodes), dropout 0.10, learning rate 0.0069, trained on the entire training period
  - Single LSTM: the pre-trained LSTM fine-tuned on the entire training period
  - MoE: the 7-h input windows are scaled, reduced by PCA (PC components) and clustered by a GMM (K clusters); an expert is obtained by fine-tuning the pre-trained LSTM on the samples of each cluster, and each forecast is made by the expert of the assigned cluster (hard routing). K = 4, 6, 8, 10 and PC = 10, 20, 30 were compared, and K = 10, PC = 10 was used for the detailed analyses

## Repository structure

```
README.md
02.data/                                  data
    raw_train.csv, raw_val.csv, raw_test.csv
    input_{split}_{L}hr.csv               model input files (lead times L = 1–9 h)
    output_{split}_{L}hr.csv              model output files
    Label_forecasting.csv                 water level regimes of the 2025 test hours
03.code/                                  code, numbered in the order it is run
    1_make_input_output.py                input/output files
    2_pretrain_lstm.py                    pre-trained LSTM
    3_single_lstm.py                      single LSTM
    4_gmm_router.py                       GMM router
    5_train_experts.py                    experts
    6_moe_forecast.py                     MoE forecasts (test year 2025)
    7_evaluate_2026.py                    additional test year 2026
    8_evaluate_regimes.py                 2025 and 2026 by water level regime
04.results/                               outputs of the scripts (empty until the code is run)
```

## Data

### Sources

Hourly water level and rainfall data were obtained from the Water Resources Management Information System (WAMIS, http://www.wamis.go.kr), South Korea. Rainfall at Geumsan is from the Korea Meteorological Administration (ASOS station 238). The water level is given in EL. m (WAMIS stage + gauge datum 0.96 m). Missing values were filled using linear interpolation.

| Column | Station | Code |
|---|---|---|
| Gongju_Bongryong | Gongju-si (Bongryong-ri) | WAMIS 30124080 |
| Gongju_Banpo | Gongju-si (Banpo M.S.) | WAMIS 30124070 |
| Nonsan_Hwangsan | Nonsan-si (Hwangsan-ri) | WAMIS 30144010 |
| Nonsan_Toyang | Nonsan-si (Toyang-ri) | WAMIS 30134050 |
| Nonsan_Yeonsan | Nonsan-si (Yeonsan M.S.) | WAMIS 30134010 |
| Daejeon_Dugye | Daejeon (Dugye Bridge) | WAMIS 30094030 |
| Iksan_Hamyol | Iksan-si (Hamyol E.S.) | WAMIS 30144020 |
| Iksan_Yeondong | Iksan-si (Yeondong-ri) | WAMIS 30144050 |
| Nonsan_Dongsan | Nonsan-si (Dongsan E.S.) | WAMIS 30134040 |
| Wanju_Jangseon | Wanju-gun (Unju E.S.) | WAMIS 30134020 |
| Geumsan | Geumsan | KMA ASOS 238 |
| Jinan_Juyang | Jinan-gun (Juyang-ri) | WAMIS 30014210 |
| water level | Nonsan-si (Nonsan Bridge) | WAMIS 3013670 |

Rainfall is in mm/h. The raw files also contain hourly temperature and humidity at three weather stations, which were not used in this study.

### Files

- `raw_train.csv`, `raw_val.csv`, `raw_test.csv` (encoding cp949; first column `시간` = time): hourly series for 2006–2023, 2024 and 2025, each with preceding hours so that the first input window of each period is complete.
- `input_{split}_{L}hr.csv` and `output_{split}_{L}hr.csv`, L = 1–9
  - split: `learn` (2006–2023, 157,776 targets), `validation` (2024, 8,784), `forecasting` (2025, 8,760), `test2026` (1 January–31 October 2026, 7,296)
  - The `learn`, `validation` and `forecasting` files are made from the raw files by step 1.
  - `output` row i is the water level at target time i. The `input` file has 6 more rows; input rows i to i+6 form the 7-h window that ends L hours before target i.
- `Label_forecasting.csv`: water level regime of each 2025 test hour based on the water level duration curve of the training period: 1 = DWL (h ≤ EL. 1.79 m), 2 = LWL (1.79 < h ≤ 2.01 m), 3 = OWL (2.01 < h ≤ 2.10 m), 4 = AWL (h > 2.10 m). Step 8 uses it to check its own regime classification.

## Requirements

Python 3 with TensorFlow (Keras 3), scikit-learn, NumPy, pandas, joblib and Matplotlib. The models in the paper were saved with Keras 3.13.2 (1–8 h) and Keras 3.14.1 with TensorFlow 2.21 (9 h).

## How to run

Run the scripts in `03.code/` in the order of their numbers from the repository root (the folder containing this README), for example `python 03.code/1_make_input_output.py`. Steps 1–3, 5 and 6 loop over the lead times 1–9 h. Trained models are not included. From step 2 on, every script writes its outputs to `04.results/` (empty in this repository), and later steps read them from there. The output file names contain the lead time L and, from step 4 on, the number of clusters K and PCA components PC.

1. **Input/output files** (already provided): `1_make_input_output.py` makes the `learn`, `validation` and `forecasting` files in `02.data/` from the raw files.
2. **Pre-training**: `2_pretrain_lstm.py`. The LSTM hyperparameters of Table 4 in the paper (`N_LAYERS`, `UNITS`, `DROPOUT`, `LR`) are set at the top of the script; the GA search is not included. Saves the input/output scalers (`scaler_X_{L}hr.pkl`, `scaler_y_{L}hr.pkl`) and the pre-trained LSTMs (`pretrained_lstm_{L}hr.keras`).
3. **Single LSTM**: `3_single_lstm.py`. Results: `single_lstm_{L}hr_summary.csv`; forecasts for 2025: `single_lstm_{L}hr_test.csv`.
4. **GMM router**: `4_gmm_router.py`. Saves the fitted scaler, PCA and GMM, the cluster labels and the cluster probabilities (`gmm_K{K}_PC{PC}_*`).
5. **Experts**: `5_train_experts.py`. Fine-tunes one expert per cluster (`expert_K{K}_PC{PC}_{L}hr_label{k}.keras`).
6. **MoE forecasts** (test year 2025): `6_moe_forecast.py`. Results: `moe_K{K}_PC{PC}_{L}hr_summary.csv`, where the `Hard` row is the MoE in the paper; forecasts: `moe_K{K}_PC{PC}_{L}hr_hard.csv`.
7. **Additional test year 2026**: `7_evaluate_2026.py`. Applies the trained single LSTMs, GMM routers and experts, without retraining, to the 2026 files. It also recomputes the 2025 forecasts and writes their largest difference from the outputs of steps 3 and 6 to `check_2025_reproduction.csv`. Results: `metrics_2025_2026.csv` and `pred_2026_{L}hr.csv`.
8. **Water level regimes**: `8_evaluate_regimes.py`. RMSE and MAE of the single LSTM and the MoE (K = 10, PC = 10) in each regime for 2025, 2026 and both years. Results: `regime_2025_2026.csv`.

Steps 4–6 are saved with K = 10 and PC = 30. For another combination of K (4, 6, 8, 10) and PC (10, 20, 30), set `N_CLUSTERS` and `PCA_COMPONENTS` at the top of steps 4, 5 and 6; the output file names follow these settings.

Steps 7 and 8 need the outputs of steps 4–6 for K = 10, PC = 10. Step 7 evaluates every combination in `COMBOS` (all 12 as saved), so either run steps 4–6 for all 12 combinations or set `COMBOS = [(10, 10)]`.

LSTM training is not fully deterministic, so retrained models may give slightly different values from those in the paper.

## License

To be added.
