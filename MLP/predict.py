import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 加载数据集，指定编码格式为 GB2312
data = pd.read_csv('/home/USER2022100821/work/Artificial-Intelligence/北京房价/predict.csv', encoding='GB2312', engine='python')

# 随机选取二十行数据，并设置随机种子
data_sample = data.sample(n=30, random_state=42)

# 选择重要特征（包括 floor）
features = [
    'Lng', 'Lat', 'square', 'livingRoom', 'drawingRoom', 'kitchen', 'bathRoom', 'floor',
    'buildingType', 'renovationCondition', 'buildingStructure', 'ladderRatio', 'elevator'
]
X = data_sample[features]
y_totalPrice = data_sample['totalPrice']
y_price = data_sample['price']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 加载训练好的模型
model_totalPrice = tf.keras.models.load_model('house_price_model_totalPrice.h5')
model_price = tf.keras.models.load_model('house_price_model_price.h5')

# 进行预测
predicted_totalPrice = model_totalPrice.predict(X_scaled)
predicted_price = model_price.predict(X_scaled)

# 打印实际值和预测值
results = pd.DataFrame({
    'row_number': data_sample.index + 2,  # 加上偏移量 2
    'actual_totalPrice': y_totalPrice.reset_index(drop=True),
    'predicted_totalPrice': predicted_totalPrice.flatten(),
    'actual_price': y_price.reset_index(drop=True),
    'predicted_price': predicted_price.flatten()
})

print("MLP Model Predictions:")
print(results[['row_number', 'actual_totalPrice', 'predicted_totalPrice', 'actual_price', 'predicted_price']])

# 可视化预测值与实际值（totalPrice）
plt.figure(figsize=(10, 5))
plt.plot(results['actual_totalPrice'], label='Actual Total Price (MLP)', color='blue', marker='o')
plt.plot(results['predicted_totalPrice'], label='Predicted Total Price (MLP)', color='red', marker='x')
plt.xlabel('Sample Index')
plt.ylabel('Total Price')
plt.title('Actual vs Predicted Total Price (MLP)')
plt.legend()
plt.show()

# 可视化预测值与实际值（price）
plt.figure(figsize=(10, 5))
plt.plot(results['actual_price'], label='Actual Price (MLP)', color='blue', marker='o')
plt.plot(results['predicted_price'], label='Predicted Price (MLP)', color='red', marker='x')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.title('Actual vs Predicted Price (MLP)')
plt.legend()
plt.show()