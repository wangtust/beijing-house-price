import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# 加载数据集，指定编码格式为 GB2312
data = pd.read_csv('./predict.csv', encoding='GB2312', engine='python')

# 随机选取二十行数据，并设置随机种子
data_sample = data.sample(n=20, random_state=43)

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
X_scaled = scaler.fit_transform(X)

# 加载训练好的模型
svr_totalPrice = joblib.load('svr_totalPrice_optimized.pkl')

# 进行预测
predicted_totalPrice_svr = svr_totalPrice.predict(X_scaled)

# 计算 MAE
mae_totalPrice_svr = mean_absolute_error(y_totalPrice, predicted_totalPrice_svr)

# 打印 MAE
print(f"Mean Absolute Error for totalPrice (SVR): {mae_totalPrice_svr}")

# 打印实际值和预测值
results = pd.DataFrame({
    'row_number': data_sample.index + 2,  # 加上偏移量 2
    'actual_totalPrice': y_totalPrice.reset_index(drop=True),
    'predicted_totalPrice_svr': predicted_totalPrice_svr
})

print(results[['row_number', 'actual_totalPrice', 'predicted_totalPrice_svr']])

# 可视化预测值与实际值（totalPrice）
plt.figure(figsize=(15, 10))
plt.plot(results['actual_totalPrice'], label='Actual Total Price', color='blue', marker='o')
plt.plot(results['predicted_totalPrice_svr'], label='Predicted Total Price (SVR)', color='red', marker='x')
plt.xlabel('Sample Index')
plt.ylabel('Total Price')
plt.title('Actual vs Predicted Total Price (SVR)')
plt.legend()

# 显示图表
plt.show()