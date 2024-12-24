import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import joblib
import numpy as np

# 检查是否有可用的 GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# 加载数据集，指定编码格式为 GB2312
data = pd.read_csv('./new_cleaned.csv', encoding='GB2312', engine='python')

# 打印列名
print("Columns in the dataset:", data.columns)

# 去除列名中的空格
data.columns = data.columns.str.strip()

# 将 tradeTime 转换为 datetime 格式
data['tradeTime'] = pd.to_datetime(data['tradeTime'], errors='coerce')

# 提取年份、月份等时间特征
data['tradeYear'] = data['tradeTime'].dt.year
data['tradeMonth'] = data['tradeTime'].dt.month

# 取出 50% 的数据进行实验
#data_sample = data.sample(frac=0.9, random_state=42)
data_sample = data
# 选择重要特征（包括 floor 和时间特征）
features = [
    'Lng', 'Lat', 'square', 'livingRoom', 'drawingRoom', 'kitchen', 'bathRoom', 'floor',
    'buildingType', 'renovationCondition', 'buildingStructure', 'ladderRatio', 'elevator',
    'DOM', 'followers', 'constructionTime', 'fiveYearsProperty', 'subway', 'district',
    'tradeYear', 'tradeMonth'
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
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)

# 训练模型
svr.fit(X_train, y_train_totalPrice)

# 预测
y_pred = svr.predict(X_test)

# 打印模型性能
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test_totalPrice, y_pred)
print(f'Mean Absolute Error: {mae}')

# 保存模型和标准化器
joblib.dump(svr, 'svr_totalPrice_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("SVR model and scaler saved to disk.")