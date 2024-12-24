import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import joblib
import matplotlib.pyplot as plt

# 检查是否有可用的 GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# 加载数据集，指定编码格式为 GB2312
data = pd.read_csv('/home/USER2022100821/work/Artificial-Intelligence/北京房价/new_cleaned.csv', encoding='GB2312', engine='python')

# 打印列名
print("Columns in the dataset:", data.columns)

# 去除列名中的空格
data.columns = data.columns.str.strip()

# 取出 10% 的数据进行实验
data_sample = data.sample(frac=0.1, random_state=42)

# 选择重要特征（包括 floor）
features = [
    'Lng', 'Lat', 'square', 'livingRoom', 'drawingRoom', 'kitchen', 'bathRoom', 'floor',
    'buildingType', 'renovationCondition', 'buildingStructure', 'ladderRatio', 'elevator'
]
X = data_sample[features]
y_totalPrice = data_sample['totalPrice']
y_price = data_sample['price']

# 检查并处理缺失值和无效值
print("Checking for missing values...")
print(X.isnull().sum())

# 处理缺失值和无效值（示例：填充缺失值，替换无效值）
X.replace(['#NAME?', 'δ?'], pd.NA, inplace=True)
X.fillna(X.mean(), inplace=True)

# 再次检查缺失值
print("Checking for missing values after filling...")
print(X.isnull().sum())

# 确保所有缺失值都已处理
X = X.dropna()
y_totalPrice = y_totalPrice.loc[X.index]
y_price = y_price.loc[X.index]

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train_totalPrice, y_test_totalPrice = train_test_split(X, y_totalPrice, test_size=0.2, random_state=42)
X_train, X_test, y_train_price, y_test_price = train_test_split(X, y_price, test_size=0.2, random_state=42)

# 定义和训练SVR模型（预测 totalPrice）
svr_totalPrice = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_totalPrice.fit(X_train, y_train_totalPrice)
pred_totalPrice = svr_totalPrice.predict(X_test)
mae_totalPrice = mean_absolute_error(y_test_totalPrice, pred_totalPrice)
print(f"Mean Absolute Error for totalPrice: {mae_totalPrice}")

# 定义和训练SVR模型（预测 price）
svr_price = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_price.fit(X_train, y_train_price)
pred_price = svr_price.predict(X_test)
mae_price = mean_absolute_error(y_test_price, pred_price)
print(f"Mean Absolute Error for price: {mae_price}")

# 保存模型
joblib.dump(svr_totalPrice, 'house_price_model_totalPrice.pkl')
joblib.dump(svr_price, 'house_price_model_price.pkl')
print("Models saved to house_price_model_totalPrice.pkl and house_price_model_price.pkl")

# 可视化预测值与实际值（totalPrice）
plt.figure(figsize=(10, 5))
plt.plot(y_test_totalPrice.values, label='Actual Total Price', color='blue', marker='o')
plt.plot(pred_totalPrice, label='Predicted Total Price', color='red', marker='x')
plt.xlabel('Sample Index')
plt.ylabel('Total Price')
plt.title('Actual vs Predicted Total Price')
plt.legend()
plt.show()

# 可视化预测值与实际值（price）
plt.figure(figsize=(10, 5))
plt.plot(y_test_price.values, label='Actual Price', color='blue', marker='o')
plt.plot(pred_price, label='Predicted Price', color='red', marker='x')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.title('Actual vs Predicted Price')
plt.legend()
plt.show()