import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import joblib

# 检查是否有可用的 GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# 加载数据集，指定编码格式为 GB2312
data = pd.read_csv('./new_cleaned.csv', encoding='GB2312', engine='python')

# 打印列名
print("Columns in the dataset:", data.columns)

# 去除列名中的空格
data.columns = data.columns.str.strip()

# 取出 10% 的数据进行实验
data_sample = data.sample(frac=0.1, random_state=42)

# 选择重要特征（包括 floor）
features = [
    'Lng', 'Lat', 'square', 'livingRoom', 'drawingRoom', 'kitchen', 'bathRoom', 'floor',
    'buildingType', 'renovationCondition', 'buildingStructure', 'ladderRatio', 'elevator',
    'DOM', 'followers', 'constructionTime', 'fiveYearsProperty', 'subway', 'district'
]
X = data_sample[features]
y_totalPrice = data_sample['totalPrice']

# 处理缺失值和无效值
X.replace(['#NAME?', 'δ?'], pd.NA, inplace=True)
X.fillna(X.mean(), inplace=True)
X = X.dropna()
y_totalPrice = y_totalPrice.loc[X.index].copy()

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train_totalPrice, y_test_totalPrice = train_test_split(X, y_totalPrice, test_size=0.2, random_state=42)

# 定义 SVR 模型
svr = SVR()

# 定义参数网格
param_grid = {
    'kernel': ['rbf', 'linear'],
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10],
    'epsilon': [0.01, 0.1, 1]
}

# 使用网格搜索优化超参数
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train_totalPrice)

# 打印最佳参数
print("Best parameters found: ", grid_search.best_params_)

# 使用最佳参数训练模型
best_svr = grid_search.best_estimator_
best_svr.fit(X_train, y_train_totalPrice)

# 保存优化后的模型
joblib.dump(best_svr, 'svr_totalPrice_optimized.pkl')

print("Optimized SVR model saved to disk.")