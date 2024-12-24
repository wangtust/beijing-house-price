import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib

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
data_sample = data.sample(frac=0.1, random_state=42)

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

# 处理缺失值和无效值
X.replace(['#NAME?', 'δ?'], pd.NA, inplace=True)
X.fillna(X.mean(), inplace=True)
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
joblib.dump(svr_totalPrice, 'svr_totalPrice.pkl')

# 定义和训练SVR模型（预测 price）
svr_price = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_price.fit(X_train, y_train_price)
joblib.dump(svr_price, 'svr_price.pkl')

# 定义和训练随机森林模型（预测 totalPrice）
rf_totalPrice = RandomForestRegressor(n_estimators=100, random_state=42)
rf_totalPrice.fit(X_train, y_train_totalPrice)
joblib.dump(rf_totalPrice, 'rf_totalPrice.pkl')

# 定义和训练随机森林模型（预测 price）
rf_price = RandomForestRegressor(n_estimators=100, random_state=42)
rf_price.fit(X_train, y_train_price)
joblib.dump(rf_price, 'rf_price.pkl')

# 定义和训练梯度提升模型（预测 totalPrice）
gb_totalPrice = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_totalPrice.fit(X_train, y_train_totalPrice)
joblib.dump(gb_totalPrice, 'gb_totalPrice.pkl')

# 定义和训练梯度提升模型（预测 price）
gb_price = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_price.fit(X_train, y_train_price)
joblib.dump(gb_price, 'gb_price.pkl')

# 定义和训练MLP模型（预测 totalPrice）
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])
    return model

model_totalPrice = build_model()
model_totalPrice.fit(X_train, y_train_totalPrice, epochs=50, batch_size=32, validation_split=0.2)
model_totalPrice.save('mlp_totalPrice.h5')

model_price = build_model()
model_price.fit(X_train, y_train_price, epochs=50, batch_size=32, validation_split=0.2)
model_price.save('mlp_price.h5')

print("Models saved to disk.")