import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 检查是否有可用的 GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# 加载数据集，指定编码格式为 GB2312
data = pd.read_csv('/home/USER2022100821/work/Artificial-Intelligence/北京房价/new.csv', encoding='GB2312', engine='python')

# 打印列名
print("Columns in the dataset:", data.columns)

# 去除列名中的空格
data.columns = data.columns.str.strip()

# 取出 10% 的数据进行实验
data_sample = data.sample(frac=0.1, random_state=42)

# 选择重要特征（不包括 floor）
features = [
    'Lng', 'Lat', 'square', 'livingRoom', 'drawingRoom', 'kitchen', 'bathRoom',
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
y_totalPrice = y_totalPrice[X.index]
y_price = y_price[X.index]

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train_totalPrice, y_test_totalPrice = train_test_split(X, y_totalPrice, test_size=0.2, random_state=42)
X_train, X_test, y_train_price, y_test_price = train_test_split(X, y_price, test_size=0.2, random_state=42)

# 定义模型
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 训练和评估模型（预测 totalPrice）
model_totalPrice = build_model()
model_totalPrice.fit(X_train, y_train_totalPrice, epochs=50, batch_size=32, validation_split=0.2)
loss_totalPrice, mae_totalPrice = model_totalPrice.evaluate(X_test, y_test_totalPrice)
print(f"Mean Absolute Error for totalPrice: {mae_totalPrice}")

# 训练和评估模型（预测 price）
model_price = build_model()
model_price.fit(X_train, y_train_price, epochs=50, batch_size=32, validation_split=0.2)
loss_price, mae_price = model_price.evaluate(X_test, y_test_price)
print(f"Mean Absolute Error for price: {mae_price}")

# 保存模型
model_totalPrice.save('house_price_model_totalPrice.h5')
model_price.save('house_price_model_price.h5')
print("Models saved to house_price_model_totalPrice.h5 and house_price_model_price.h5")