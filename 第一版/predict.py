import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# 加载数据集，指定编码格式为 GB2312
data = pd.read_csv('/home/USER2022100821/work/Artificial-Intelligence/北京房价/predict.csv', encoding='GB2312', engine='python')

# 随机选取五行数据
data_sample = data.sample(n=5, random_state=42)

# 选择重要特征（不包括 floor）
features = [
    'Lng', 'Lat', 'square', 'livingRoom', 'drawingRoom', 'kitchen', 'bathRoom',
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
    'actual_totalPrice': y_totalPrice,
    'predicted_totalPrice': predicted_totalPrice.flatten(),
    'actual_price': y_price,
    'predicted_price': predicted_price.flatten()
})

print(results[['actual_totalPrice', 'predicted_totalPrice', 'actual_price', 'predicted_price']])