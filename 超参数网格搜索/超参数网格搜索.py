import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 全局设置
# ==============================================================================
RANDOM_STATE = 42          # 固定随机种子，确保可复现
TEST_SIZE = 0.2            # 测试集比例
CV_FOLDS = 5               # 交叉验证折数
SCORING = 'roc_auc'        # 评估指标

# 数据路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'semiconductor_quality_control.csv')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, '超参数搜索结果.xlsx')

# 工艺参数列表
PROCESS_FEATURES = [
    'Chamber_Temperature', 'Gas_Flow_Rate', 'RF_Power', 'Etch_Depth',
    'Rotation_Speed', 'Vacuum_Pressure', 'Stage_Alignment_Error',
    'Vibration_Level', 'UV_Exposure_Intensity', 'Particle_Count'
]

# ==============================================================================
# 1. 数据加载与预处理
# ==============================================================================
print("=" * 70)
print("【三模型统一超参数网格搜索】")
print(f"随机种子: {RANDOM_STATE} | 交叉验证: {CV_FOLDS}折 | 评估指标: {SCORING}")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"\n数据加载: {df.shape[0]} 行 × {df.shape[1]} 列")

X = df[PROCESS_FEATURES].copy()
y = df['Defect'].values

print(f"特征数量: {X.shape[1]}")
print(f"缺陷率: {y.mean():.1%} ({sum(y)}/{len(y)})")

# 数据集划分（所有模型共用同一划分）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"训练集: {X_train.shape[0]} 条 | 测试集: {X_test.shape[0]} 条")

# 标准化（逻辑回归和SVM需要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# 2. 随机森林 网格搜索
# ==============================================================================
print("\n" + "=" * 70)
print("【1/3】随机森林 (RandomForest) 网格搜索")
print("=" * 70)

rf_param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    rf_param_grid, cv=CV_FOLDS, scoring=SCORING, n_jobs=-1, verbose=1
)
rf_grid.fit(X_train, y_train)

rf_best = rf_grid.best_estimator_
print(f"\n最佳参数: {rf_grid.best_params_}")
print(f"最佳交叉验证 AUC: {rf_grid.best_score_:.4f}")

# 测试集性能
rf_test_score = rf_best.score(X_test, y_test)
print(f"测试集准确率: {rf_test_score:.4f}")

# ==============================================================================
# 3. 逻辑回归 网格搜索
# ==============================================================================
print("\n" + "=" * 70)
print("【2/3】逻辑回归 (LogisticRegression) 网格搜索")
print("=" * 70)

lr_param_grid = {
    'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100],
    'penalty': ['l1', 'l2', None],
    'solver': ['liblinear', 'saga', 'lbfgs'],
    'class_weight': ['balanced', None],
    'max_iter': [1000, 5000]
}

lr_grid = GridSearchCV(
    LogisticRegression(random_state=RANDOM_STATE),
    lr_param_grid, cv=CV_FOLDS, scoring=SCORING, n_jobs=-1, verbose=1
)
lr_grid.fit(X_train_scaled, y_train)

lr_best = lr_grid.best_estimator_
print(f"\n最佳参数: {lr_grid.best_params_}")
print(f"最佳交叉验证 AUC: {lr_grid.best_score_:.4f}")

# 测试集性能
lr_test_score = lr_best.score(X_test_scaled, y_test)
print(f"测试集准确率: {lr_test_score:.4f}")

# ==============================================================================
# 4. SVM 网格搜索（统一使用 linear 核）
# ==============================================================================
print("\n" + "=" * 70)
print("【3/3】支持向量机 (SVM) 网格搜索")
print("    kernel 统一使用 'linear'（确保与PCA特征重要性分析一致）")
print("=" * 70)

svm_param_grid = {
    'C': [0.01, 0.1, 0.5, 1, 5, 10, 50],
    'class_weight': ['balanced', None],
    'max_iter': [5000, 10000]
}

svm_grid = GridSearchCV(
    SVC(kernel='linear', random_state=RANDOM_STATE, probability=True),
    svm_param_grid, cv=CV_FOLDS, scoring=SCORING, n_jobs=-1, verbose=1
)
svm_grid.fit(X_train_scaled, y_train)

svm_best = svm_grid.best_estimator_
print(f"\n最佳参数: {svm_grid.best_params_}")
print(f"最佳交叉验证 AUC: {svm_grid.best_score_:.4f}")

# 测试集性能
svm_test_score = svm_best.score(X_test_scaled, y_test)
print(f"测试集准确率: {svm_test_score:.4f}")

# ==============================================================================
# 5. 结果汇总
# ==============================================================================
print("\n" + "=" * 70)
print("【网格搜索结果汇总】")
print("=" * 70)

# 提取最佳参数（扁平化处理）
def extract_best_params(grid_result, model_name):
    """提取最佳参数并格式化"""
    params = grid_result.best_params_
    return {
        '模型': model_name,
        '最佳交叉验证AUC': f"{grid_result.best_score_:.4f}",
        '测试集准确率': '',
        **{k: str(v) for k, v in params.items()}
    }

# 构建结果表
results = []

# 随机森林
rf_row = {'模型': '随机森林', '最佳交叉验证AUC': f"{rf_grid.best_score_:.4f}"}
rf_row.update({f'RF_{k}': str(v) for k, v in rf_grid.best_params_.items()})
rf_row['测试集准确率'] = f"{rf_test_score:.4f}"
results.append(rf_row)

# 逻辑回归
lr_row = {'模型': '逻辑回归', '最佳交叉验证AUC': f"{lr_grid.best_score_:.4f}"}
lr_row.update({f'LR_{k}': str(v) for k, v in lr_grid.best_params_.items()})
lr_row['测试集准确率'] = f"{lr_test_score:.4f}"
results.append(lr_row)

# SVM
svm_row = {'模型': 'SVM', '最佳交叉验证AUC': f"{svm_grid.best_score_:.4f}"}
svm_row.update({f'SVM_{k}': str(v) for k, v in svm_grid.best_params_.items()})
svm_row['测试集准确率'] = f"{svm_test_score:.4f}"
results.append(svm_row)

results_df = pd.DataFrame(results)

# 输出到控制台
print("\n" + results_df[['模型', '最佳交叉验证AUC', '测试集准确率']].to_string(index=False))

# 输出完整参数
print("\n--- 随机森林最优参数 ---")
for k, v in rf_grid.best_params_.items():
    print(f"  {k}: {v}")

print("\n--- 逻辑回归最优参数 ---")
for k, v in lr_grid.best_params_.items():
    print(f"  {k}: {v}")

print("\n--- SVM最优参数 ---")
for k, v in svm_grid.best_params_.items():
    print(f"  {k}: {v}")

# 保存到 Excel
with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='超参数搜索结果', index=False)

    # 详细参数表
    param_detail = pd.DataFrame([
        {
            '模型': '随机森林',
            '超参数': k,
            '最优值': str(v),
            '搜索范围': str(rf_param_grid.get(k, 'N/A'))
        }
        for k, v in rf_grid.best_params_.items()
    ] + [
        {
            '模型': '逻辑回归',
            '超参数': k,
            '最优值': str(v),
            '搜索范围': str(lr_param_grid.get(k, 'N/A'))
        }
        for k, v in lr_grid.best_params_.items()
    ] + [
        {
            '模型': 'SVM',
            '超参数': k,
            '最优值': str(v),
            '搜索范围': str(svm_param_grid.get(k, 'N/A'))
        }
        for k, v in svm_grid.best_params_.items()
    ])
    param_detail.to_excel(writer, sheet_name='参数详情', index=False)

    # 搜索设置
    pd.DataFrame({
        '项目': ['随机种子', '交叉验证折数', '评估指标', '测试集比例',
                 '数据集样本数', '特征数量', 'SVM核函数'],
        '值': [RANDOM_STATE, f'{CV_FOLDS}折', SCORING, TEST_SIZE,
               f'{len(df)}条', f'{len(PROCESS_FEATURES)}个', 'linear']
    }).to_excel(writer, sheet_name='搜索设置', index=False)

print(f"\n[OK] 结果已保存: {OUTPUT_PATH}")
print("\n" + "=" * 70)
print("超参数网格搜索完成！")
print("请将上述最优参数复制到对应模型训练程序中。")
print("=" * 70)
