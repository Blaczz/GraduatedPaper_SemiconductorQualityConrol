import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             classification_report)
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 原始工艺参数数据路径
RAW_DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'semiconductor_quality_control.csv')
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')

# 10个原始工艺参数
FEATURE_COLS = ['Chamber_Temperature', 'Gas_Flow_Rate', 'RF_Power', 'Etch_Depth',
                'Rotation_Speed', 'Vacuum_Pressure', 'Stage_Alignment_Error',
                'Vibration_Level', 'UV_Exposure_Intensity', 'Particle_Count']


def load_and_preprocess_data():
    """加载原始10个工艺参数数据"""
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"数据加载（原始工艺参数）: {df.shape[0]}行, {df.shape[1]}列")

    X = df[FEATURE_COLS].copy()
    y = df['Defect'].values

    print(f"特征数: {len(FEATURE_COLS)}, 缺陷率: {y.mean():.1%}")
    return X, y, FEATURE_COLS


def build_logistic_regression(X, y):
    """逻辑回归模型训练与超参数优化
    网格搜索 + 5折交叉验证，ROC-AUC为指标
    最优参数通过统一网格搜索确定: C=0.001, penalty='l2', solver='lbfgs',
      max_iter=1000, class_weight='balanced'
    random_state=42 确保结果可复现
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    param_grid = {
        'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100],
        'penalty': ['l1', 'l2', None],
        'solver': ['liblinear', 'saga', 'lbfgs'],
        'class_weight': ['balanced', None],
        'max_iter': [1000, 5000]
    }

    grid = GridSearchCV(
        LogisticRegression(random_state=42),
        param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train_s, y_train)

    print(f"最佳参数: {grid.best_params_}")
    print(f"最佳CV AUC: {grid.best_score_:.4f}")

    model = grid.best_estimator_

    # 系数分析
    feature_imp = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_[0],
        'abs_coefficient': np.abs(model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)

    print("\n系数分析 Top5:")
    print(feature_imp.head(5).to_string(index=False))

    return model, X_train_s, X_test_s, y_train, y_test, scaler, feature_imp


def evaluate(model, X_test_s, y_test):
    """模型性能评估"""
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba)
    }

    print("\n性能指标:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['正常', '缺陷'])}")

    return y_pred, y_proba, metrics


def plot_roc(y_test, y_proba, auc_val):
    """绘制ROC曲线（含AUC积分底纹）"""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC={auc_val:.3f})')
    plt.fill_between(fpr, tpr, alpha=0.15, color='darkorange')  # AUC积分底纹
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('假正率', fontsize=12, fontweight='bold')
    plt.ylabel('真正率', fontsize=12, fontweight='bold')
    plt.title('图4-1 逻辑回归ROC曲线', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    optimal_idx = np.argmax(tpr - fpr)
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '41_逻辑回归ROC曲线.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] ROC曲线已保存")


def plot_confusion_matrix(y_test, y_pred):
    """绘制混淆矩阵热力图"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['预测正常', '预测缺陷'],
                yticklabels=['实际正常', '实际缺陷'])
    plt.xlabel('预测标签', fontsize=12, fontweight='bold')
    plt.ylabel('实际标签', fontsize=12, fontweight='bold')
    plt.title('图4-2 逻辑回归混淆矩阵热力图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '42_逻辑回归混淆矩阵热力图.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] 混淆矩阵已保存")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("【逻辑回归模型构建与性能评估】")

    X, y, _ = load_and_preprocess_data()
    model, X_train_s, X_test_s, y_train, y_test, scaler, feat_imp = build_logistic_regression(X, y)
    y_pred, y_proba, metrics = evaluate(model, X_test_s, y_test)
    plot_roc(y_test, y_proba, metrics['auc'])
    plot_confusion_matrix(y_test, y_pred)

    print(f"\n=== 分析摘要 ===")
    print(f"AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}, "
          f"Top1特征={feat_imp.iloc[0]['feature']}(系数={feat_imp.iloc[0]['coefficient']:.4f})")
    print("[OK] 逻辑回归模型分析完成")


if __name__ == "__main__":
    main()
