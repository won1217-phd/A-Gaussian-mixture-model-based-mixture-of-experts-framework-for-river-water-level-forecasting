import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

for xx in [1]:
    LEAD_TIME    = xx
    TIME_STEPS   = 7
    N_CLUSTERS   = 10
    RANDOM_STATE = 42
    N_INIT       = 10
    MAX_ITER     = 200
    COVARIANCE   = 'full'

    PCA_COMPONENTS = 30

    BASE_DIR   = '02.data'
    OUTPUT_DIR = '04.results'
    PREFIX     = f'gmm_K{N_CLUSTERS}_PC{PCA_COMPONENTS}_'

    TRAIN_INPUT_PATH  = os.path.join(BASE_DIR, 'input_learn_{}hr.csv'.format(LEAD_TIME))
    VAL_INPUT_PATH    = os.path.join(BASE_DIR, 'input_validation_{}hr.csv'.format(LEAD_TIME))
    TEST_INPUT_PATH   = os.path.join(BASE_DIR, 'input_forecasting_{}hr.csv'.format(LEAD_TIME))
    TRAIN_OUTPUT_PATH = os.path.join(BASE_DIR, 'output_learn_{}hr.csv'.format(LEAD_TIME))

    INPUT_COLS = [
        'Gongju_Bongryong', 'Gongju_Banpo',
        'Nonsan_Hwangsan',  'Nonsan_Toyang',
        'Nonsan_Yeonsan',   'Daejeon_Dugye',
        'Iksan_Hamyol',     'Iksan_Yeondong',
        'Nonsan_Dongsan',   'Wanju_Jangseon',
        'Geumsan',          'Jinan_Juyang',
        'water level'
    ]

    print("데이터 로드 중...")
    train_X = pd.read_csv(TRAIN_INPUT_PATH, encoding='utf-8-sig')
    val_X   = pd.read_csv(VAL_INPUT_PATH,   encoding='utf-8-sig')
    test_X  = pd.read_csv(TEST_INPUT_PATH,  encoding='utf-8-sig')
    train_y = pd.read_csv(TRAIN_OUTPUT_PATH, encoding='utf-8-sig').values.flatten()

    n_train = len(train_y)
    n_val   = len(val_X) - (TIME_STEPS - 1)
    n_test  = len(test_X) - (TIME_STEPS - 1)

    print(f"  학습: {n_train}개 / 검증: {n_val}개 / 예측: {n_test}개")

    print("\n정규화 적용 중...")
    scaler = MinMaxScaler()
    train_X_s = scaler.fit_transform(train_X[INPUT_COLS].values)
    val_X_s   = scaler.transform(val_X[INPUT_COLS].values)
    test_X_s  = scaler.transform(test_X[INPUT_COLS].values)

    print("슬라이딩 윈도우 적용 중 (7hr × 19vars = 133dim)...")

    def make_sliding_window(X_scaled, n_samples, time_steps):
        X_list = []
        for i in range(n_samples):
            X_list.append(X_scaled[i : i + time_steps, :].flatten())
        return np.array(X_list)

    X_train_sw = make_sliding_window(train_X_s, n_train, TIME_STEPS)
    X_val_sw   = make_sliding_window(val_X_s,   n_val,   TIME_STEPS)
    X_test_sw  = make_sliding_window(test_X_s,  n_test,  TIME_STEPS)

    print(f"  학습: {X_train_sw.shape}")

    print(f"\nPCA 전처리 중 (133 → {PCA_COMPONENTS}차원)...")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_sw)
    X_val_pca   = pca.transform(X_val_sw)
    X_test_pca  = pca.transform(X_test_sw)

    var_ratio = pca.explained_variance_ratio_.sum() * 100
    print(f"  누적 설명 분산: {var_ratio:.1f}%")

    print("\n최적 K 탐색 중 (K=2~8)...")
    np.random.seed(RANDOM_STATE)
    sample_idx = np.random.choice(len(X_train_pca),
                                   min(5000, len(X_train_pca)), replace=False)
    X_sample   = X_train_pca[sample_idx]

    k_range  = range(2, 9)
    bic_list = []
    aic_list = []
    sil_list = []

    for k in k_range:
        gm = GaussianMixture(n_components=k, covariance_type=COVARIANCE,
                             n_init=3, max_iter=100, random_state=RANDOM_STATE)
        gm.fit(X_sample)
        labels_tmp = gm.predict(X_sample)
        bic_list.append(gm.bic(X_sample))
        aic_list.append(gm.aic(X_sample))
        try:
            sil = silhouette_score(X_sample, labels_tmp,
                                   sample_size=3000, random_state=RANDOM_STATE)
        except:
            sil = 0
        sil_list.append(sil)
        print(f"  K={k}: BIC={gm.bic(X_sample):.1f}, "
              f"AIC={gm.aic(X_sample):.1f}, Silhouette={sil:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(list(k_range), bic_list, 'bo-', linewidth=2)
    axes[0].axvline(x=N_CLUSTERS, color='r', linestyle='--',
                    label=f'Selected K={N_CLUSTERS}')
    axes[0].set_xlabel('Number of Components (K)')
    axes[0].set_ylabel('BIC')
    axes[0].set_title('BIC (낮을수록 좋음)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(list(k_range), aic_list, 'go-', linewidth=2)
    axes[1].axvline(x=N_CLUSTERS, color='r', linestyle='--',
                    label=f'Selected K={N_CLUSTERS}')
    axes[1].set_xlabel('Number of Components (K)')
    axes[1].set_ylabel('AIC')
    axes[1].set_title('AIC (낮을수록 좋음)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(list(k_range), sil_list, 'ro-', linewidth=2)
    axes[2].axvline(x=N_CLUSTERS, color='b', linestyle='--',
                    label=f'Selected K={N_CLUSTERS}')
    axes[2].set_xlabel('Number of Components (K)')
    axes[2].set_ylabel('Silhouette Score')
    axes[2].set_title('Silhouette (높을수록 좋음)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, PREFIX + 'optimal_k.png'), dpi=150)
    plt.close()
    print(f"  최적 K 그래프 저장 완료")

    print(f"\nGMM 학습 중 (K={N_CLUSTERS}, cov={COVARIANCE}, n_init={N_INIT})...")
    gmm = GaussianMixture(
        n_components    = N_CLUSTERS,
        covariance_type = COVARIANCE,
        n_init          = N_INIT,
        max_iter        = MAX_ITER,
        random_state    = RANDOM_STATE,
        verbose         = 1
    )
    gmm.fit(X_train_pca)
    print(f"  GMM 학습 완료 (수렴: {gmm.converged_})")
    print(f"  BIC: {gmm.bic(X_train_pca):.2f}")
    print(f"  AIC: {gmm.aic(X_train_pca):.2f}")

    print("\n분류 중...")

    train_labels_raw = gmm.predict(X_train_pca)
    val_labels_raw   = gmm.predict(X_val_pca)
    test_labels_raw  = gmm.predict(X_test_pca)

    train_proba = gmm.predict_proba(X_train_pca)
    val_proba   = gmm.predict_proba(X_val_pca)
    test_proba  = gmm.predict_proba(X_test_pca)

    cluster_wl_mean = {}
    for c in range(N_CLUSTERS):
        idx = train_labels_raw == c
        cluster_wl_mean[c] = train_y[idx].mean() if idx.sum() > 0 else 0

    sorted_clusters = sorted(cluster_wl_mean, key=lambda x: cluster_wl_mean[x])
    label_map = {old: new+1 for new, old in enumerate(sorted_clusters)}

    print(f"\n  클러스터별 평균 수위 및 라벨 재정렬:")
    for old_c, new_l in label_map.items():
        print(f"    클러스터 {old_c} → 라벨 {new_l} "
              f"(평균 수위: {cluster_wl_mean[old_c]:.3f}m)")

    train_labels = np.array([label_map[c] for c in train_labels_raw])
    val_labels   = np.array([label_map[c] for c in val_labels_raw])
    test_labels  = np.array([label_map[c] for c in test_labels_raw])

    cluster_order = [None] * N_CLUSTERS
    for c, l in label_map.items():
        cluster_order[l-1] = c
    train_proba = train_proba[:, cluster_order]
    val_proba   = val_proba[:,   cluster_order]
    test_proba  = test_proba[:,  cluster_order]

    print(f"\n  [라벨 분포]")
    for split, labels in [('학습', train_labels),
                           ('검증', val_labels),
                           ('예측', test_labels)]:
        print(f"  [{split}]")
        for l in range(1, N_CLUSTERS+1):
            n = (labels == l).sum()
            print(f"    라벨{l}: {n:>6}개 ({n/len(labels)*100:>5.1f}%)")

    print("\n결과 저장 중...")

    joblib.dump(gmm, os.path.join(OUTPUT_DIR, PREFIX + 'model.pkl'))
    joblib.dump(pca, os.path.join(OUTPUT_DIR, PREFIX + 'pca.pkl'))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, PREFIX + 'scaler.pkl'))
    print(f"  모델 저장 완료")

    pd.DataFrame([{'cluster': k, 'label': v,
                    'wl_mean': round(cluster_wl_mean[k], 4)}
                  for k, v in label_map.items()]
                 ).to_csv(os.path.join(OUTPUT_DIR, PREFIX + 'label_map.csv'),
                          index=False, encoding='utf-8-sig')

    for fname, labels in [
        ('label_train.csv',       train_labels),
        ('label_validation.csv',  val_labels),
        ('label_forecasting.csv', test_labels),
    ]:
        pd.DataFrame({'label': labels}).to_csv(
            os.path.join(OUTPUT_DIR, PREFIX + fname), index=False, encoding='utf-8-sig')
    print(f"  label_*.csv 저장 완료")

    for split, labels, proba in [
        ('train',       train_labels, train_proba),
        ('validation',  val_labels,   val_proba),
        ('forecasting', test_labels,  test_proba),
    ]:
        prob_df = pd.DataFrame(
            proba, columns=[f'proba_{i+1}' for i in range(N_CLUSTERS)])
        prob_df.insert(0, 'label', labels)
        prob_df.to_csv(os.path.join(OUTPUT_DIR, PREFIX + f'proba_{split}.csv'),
                       index=False, encoding='utf-8-sig')
    print(f"  proba_*.csv 저장 완료 (Soft MoE 가중치용)")

    pd.DataFrame([{
        'n_clusters'  : N_CLUSTERS,
        'covariance'  : COVARIANCE,
        'pca_dim'     : PCA_COMPONENTS,
        'pca_var_pct' : round(var_ratio, 2),
        'bic'         : round(gmm.bic(X_train_pca), 2),
        'aic'         : round(gmm.aic(X_train_pca), 2),
        'converged'   : gmm.converged_,
        'n_train'     : n_train,
        'n_val'       : n_val,
        'n_test'      : n_test,
    }]).to_csv(os.path.join(OUTPUT_DIR, PREFIX + 'summary.csv'),
               index=False, encoding='utf-8-sig')

    print(f"\n{'='*60}")
    print(f"  GMM 클러스터링 완료")
    print(f"{'='*60}")
    print(f"  BIC       : {gmm.bic(X_train_pca):.2f}")
    print(f"  AIC       : {gmm.aic(X_train_pca):.2f}")
    print(f"  수렴 여부 : {gmm.converged_}")
    print(f"\n  저장 파일 ({OUTPUT_DIR}):")
    print(f"    {PREFIX}model.pkl          ← GMM 모델")
    print(f"    {PREFIX}pca.pkl            ← PCA 모델")
    print(f"    {PREFIX}scaler.pkl         ← 정규화 스케일러")
    print(f"    {PREFIX}label_map.csv      ← 클러스터-라벨 매핑")
    print(f"    {PREFIX}label_*.csv        ← Hard 라벨")
    print(f"    {PREFIX}proba_*.csv        ← Soft 확률 (Soft MoE용)")
