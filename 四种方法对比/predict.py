import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# 加载数据集，指定编码格式为 GB2312
data = pd.read_csv('/home/USER2022100821/work/Artificial-Intelligence/北京房价/predict.csv', encoding='GB2312', engine='python')

# 将 tradeTime 转换为 datetime 格式
data['tradeTime'] = pd.to_datetime(data['tradeTime'], errors='coerce')

# 提取年份、月份等时间特征
data['tradeYear'] = data['tradeTime'].dt.year
data['tradeMonth'] = data['tradeTime'].dt.month

# 随机选取二十行数据，并设置随机种子
#data_sample = data.sample(n=20, random_state=51)
data_sample = data.sample(n=20)
# 选择重要特征（包括 floor 和时间特征）
features = [
    'Lng', 'Lat', 'square', 'livingRoom', 'drawingRoom', 'kitchen', 'bathRoom', 'floor',
    'buildingType', 'renovationCondition', 'buildingStructure', 'ladderRatio', 'elevator',
    'DOM', 'followers', 'constructionTime', 'fiveYearsProperty', 'subway', 'district',
    'tradeYear', 'tradeMonth'
]
X = data_sample[features]
y_totalPrice = data_sample['totalPrice']
y_price = data_sample['price']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 加载训练好的模型
svr_totalPrice = joblib.load('svr_totalPrice.pkl')
svr_price = joblib.load('svr_price.pkl')
rf_totalPrice = joblib.load('rf_totalPrice.pkl')
rf_price = joblib.load('rf_price.pkl')
gb_totalPrice = joblib.load('gb_totalPrice.pkl')
gb_price = joblib.load('gb_price.pkl')
mlp_totalPrice = tf.keras.models.load_model('mlp_totalPrice.h5')
mlp_price = tf.keras.models.load_model('mlp_price.h5')

# 进行预测
predicted_totalPrice_svr = svr_totalPrice.predict(X_scaled)
predicted_price_svr = svr_price.predict(X_scaled)
predicted_totalPrice_rf = rf_totalPrice.predict(X_scaled)
predicted_price_rf = rf_price.predict(X_scaled)
predicted_totalPrice_gb = gb_totalPrice.predict(X_scaled)
predicted_price_gb = gb_price.predict(X_scaled)
predicted_totalPrice_mlp = mlp_totalPrice.predict(X_scaled).flatten()
predicted_price_mlp = mlp_price.predict(X_scaled).flatten()

# 计算 MAE
mae_totalPrice_svr = mean_absolute_error(y_totalPrice, predicted_totalPrice_svr)
mae_price_svr = mean_absolute_error(y_price, predicted_price_svr)
mae_totalPrice_rf = mean_absolute_error(y_totalPrice, predicted_totalPrice_rf)
mae_price_rf = mean_absolute_error(y_price, predicted_price_rf)
mae_totalPrice_gb = mean_absolute_error(y_totalPrice, predicted_totalPrice_gb)
mae_price_gb = mean_absolute_error(y_price, predicted_price_gb)
mae_totalPrice_mlp = mean_absolute_error(y_totalPrice, predicted_totalPrice_mlp)
mae_price_mlp = mean_absolute_error(y_price, predicted_price_mlp)

# 打印 MAE
print(f"Mean Absolute Error for totalPrice (SVR): {mae_totalPrice_svr}")
print(f"Mean Absolute Error for price (SVR): {mae_price_svr}")
print(f"Mean Absolute Error for totalPrice (Random Forest): {mae_totalPrice_rf}")
print(f"Mean Absolute Error for price (Random Forest): {mae_price_rf}")
print(f"Mean Absolute Error for totalPrice (Gradient Boosting): {mae_totalPrice_gb}")
print(f"Mean Absolute Error for price (Gradient Boosting): {mae_price_gb}")
print(f"Mean Absolute Error for totalPrice (MLP): {mae_totalPrice_mlp}")
print(f"Mean Absolute Error for price (MLP): {mae_price_mlp}")

# 打印实际值和预测值
results = pd.DataFrame({
    'row_number': data_sample.index + 2,  # 加上偏移量 2
    'actual_totalPrice': y_totalPrice.reset_index(drop=True),
    'predicted_totalPrice_svr': predicted_totalPrice_svr,
    'predicted_totalPrice_rf': predicted_totalPrice_rf,
    'predicted_totalPrice_gb': predicted_totalPrice_gb,
    'predicted_totalPrice_mlp': predicted_totalPrice_mlp,
    'actual_price': y_price.reset_index(drop=True),
    'predicted_price_svr': predicted_price_svr,
    'predicted_price_rf': predicted_price_rf,
    'predicted_price_gb': predicted_price_gb,
    'predicted_price_mlp': predicted_price_mlp
})

print(results[['row_number', 'actual_totalPrice', 'predicted_totalPrice_svr', 'predicted_totalPrice_rf', 'predicted_totalPrice_gb', 'predicted_totalPrice_mlp', 'actual_price', 'predicted_price_svr', 'predicted_price_rf', 'predicted_price_gb', 'predicted_price_mlp']])

# 可视化预测值与实际值（totalPrice）
plt.figure(figsize=(15, 10))
plt.plot(results['actual_totalPrice'], label='Actual Total Price', color='blue', marker='o')
plt.plot(results['predicted_totalPrice_svr'], label='Predicted Total Price (SVR)', color='red', marker='x')
plt.plot(results['predicted_totalPrice_rf'], label='Predicted Total Price (Random Forest)', color='green', marker='s')
plt.plot(results['predicted_totalPrice_gb'], label='Predicted Total Price (Gradient Boosting)', color='purple', marker='d')
plt.plot(results['predicted_totalPrice_mlp'], label='Predicted Total Price (MLP)', color='orange', marker='^')
plt.xlabel('Sample Index')
plt.ylabel('Total Price')
plt.title('Actual vs Predicted Total Price')
plt.legend()

# 可视化预测值与实际值（price）
plt.figure(figsize=(15, 10))
plt.plot(results['actual_price'], label='Actual Price', color='blue', marker='o')
plt.plot(results['predicted_price_svr'], label='Predicted Price (SVR)', color='red', marker='x')
plt.plot(results['predicted_price_rf'], label='Predicted Price (Random Forest)', color='green', marker='s')
plt.plot(results['predicted_price_gb'], label='Predicted Price (Gradient Boosting)', color='purple', marker='d')
plt.plot(results['predicted_price_mlp'], label='Predicted Price (MLP)', color='orange', marker='^')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.title('Actual vs Predicted Price')
plt.legend()

# 显示所有图表
plt.show()