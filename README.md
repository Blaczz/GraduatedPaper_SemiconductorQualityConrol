# 半导体晶圆制造缺陷预测分析

## 依赖环境

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn scipy openpyxl joblib
```

Python >= 3.8，推荐使用 conda 或 venv 虚拟环境运行。

基于 SECOM 半导体制造数据集，运用机器学习与统计过程控制方法，对晶圆制造过程中的质量缺陷进行系统性分析与预测。

## 项目结构

```
GraduatedPaper_SemiconductorQualityConrol/
├── semiconductor_quality_control.csv     # 数据集（4219条 × 16字段）
├── 数据统计性分析/                        # 描述性统计与数据探索
├── 超参数网格搜索/                        # 三模型超参数统一搜索
├── 主成分分析/                            # PCA降维 + 特征重要性 + 5M1E
├── SPC控制图分析/                         # 统计过程控制与过程能力
├── 逻辑回归/                              # 逻辑回归模型
├── 随机森林/                              # 随机森林模型（SMOTE过采样）
├── 支持向量机/                            # SVM模型（RBF核 + 类别平衡）
├── 三模型对比/                            # 三模型性能对比
└── README.md
```

## 运行顺序

所有脚本均使用**相对路径**引用数据文件，解压后无需任何配置即可运行。

### 第一步：描述性统计分析

```bash
# 数据探索：描述性统计、分布直方图、箱线图、变异系数
python "数据统计性分析/数据统计分析.py"
python "数据统计性分析/参数分布直方图.py"
python "数据统计性分析/参数分布箱线图.py"
python "数据统计性分析/变异系数对比图.py"
```

### 第二步：超参数网格搜索

```bash
# 对逻辑回归、随机森林、SVM 进行 GridSearchCV 搜索最优参数
python "超参数网格搜索/超参数网格搜索.py"
```

### 第三步：主成分分析与特征重要性

```bash
# PCA 降维 + 三模型融合特征重要性 + 5M1E 维度映射
python "主成分分析/完整PCA与特征重要性分析.py"
```

### 第四步：统计过程控制（SPC）

```bash
# X-MR 控制图 + Cp/Cpk 过程能力分析
python "SPC控制图分析/SPC控制图与过程能力分析.py"
python "SPC控制图分析/SPC子图拼接大图.py"
```

### 第五步：单模型训练与评估

```bash
# 逻辑回归
python "逻辑回归/逻辑回归模型分析.py"
# 随机森林（SMOTE 过采样 + 最优参数写死）
python "随机森林/随机森林模型分析.py"
# 支持向量机（RBF 核 + class_weight=balanced + 最优参数写死）
python "支持向量机/支持向量机模型分析.py"
```

### 第六步：三模型对比

```bash
# 统一加载原始数据，使用各自最优参数，生成 ROC 对比图
python "三模型对比/三模型对比与性能评估.py"
```

## 数据说明

`semiconductor_quality_control.csv` 包含 4219 条晶圆制造工艺记录，涵盖 10 个工艺参数和缺陷标签：

| 字段                  | 含义                       |
| --------------------- | -------------------------- |
| Chamber_Temperature   | 腔体温度（℃）              |
| Gas_Flow_Rate         | 气体流量                   |
| RF_Power              | 射频功率（W）              |
| Etch_Depth            | 刻蚀深度                   |
| Rotation_Speed        | 旋转速度（rpm）            |
| Vacuum_Pressure       | 真空压力                   |
| Stage_Alignment_Error | 晶圆台对准误差             |
| Vibration_Level       | 振动级别                   |
| UV_Exposure_Intensity | UV曝光强度                 |
| Particle_Count        | 颗粒数                     |
| Defect                | 缺陷标签（0=正常, 1=缺陷） |

缺陷率：14.6%（616 缺陷 / 4219 总样本）

## 模型最优参数

| 模型           | 参数                                                         | 说明               |
| -------------- | ------------------------------------------------------------ | ------------------ |
| **逻辑回归**   | C=0.001, penalty=l2, solver=lbfgs, class_weight=balanced     | L2正则化，强正则化 |
| **随机森林**   | n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1, class_weight=balanced | SMOTE过采样后训练  |
| **支持向量机** | kernel=rbf, C=1, gamma=0.001, class_weight=balanced          | RBF核，类别平衡    |

## 关键分析结果

| 模型                | 准确率 | 召回率     | F1     | AUC        |
| ------------------- | ------ | ---------- | ------ | ---------- |
| 逻辑回归            | 0.5308 | 0.4146     | 0.2048 | 0.4556     |
| 随机森林（SMOTE）   | 0.7654 | 0.1138     | 0.1239 | 0.4688     |
| **SVM（RBF+平衡）** | 0.4668 | **0.4390** | 0.1935 | **0.5476** |

### 5M1E 维度重要性

| 排名 | 维度              | 占比  | 关键参数                               |
| ---- | ----------------- | ----- | -------------------------------------- |
| 1    | Measurement（测） | 22.3% | 晶圆台对准误差、真空压力               |
| 2    | Machine（机）     | 22.3% | 腔体温度、射频功率、旋转速度、振动级别 |
| 3    | Material（料）    | 19.9% | 气体流量、刻蚀深度、颗粒数             |
| 4    | Environment（环） | 18.1% | 腔体温度                               |
| 5    | Method（法）      | 17.3% | UV曝光强度                             |

## 输出产物

每个模块运行后会在自身目录下生成对应的图表和报告文件：

- `figures/` — ROC 曲线、混淆矩阵、特征重要性图、PCA 图、5M1E 图、SPC 控制图
- `*.xlsx` — 分析报告汇总表
- `pca_reduced_data/` — PCA 降维后的模型文件和数据

