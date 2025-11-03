"""
房价预测线性回归实验
基于housing.csv数据集完成线性回归模型的训练、测试与评估
"""

import numpy as np
import csv
from collections import defaultdict

# ==================== 1. 准备数据集并认识数据 ====================
print("=" * 60)
print("1. 准备数据集并认识数据")
print("=" * 60)

def load_data(filename):
    """加载CSV数据"""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            data.append(row)
    return data, headers

# 加载数据
data, headers = load_data('housing.csv')
print(f"\n数据集总样本数: {len(data)}")
print(f"\n数据集特征列: {headers}")

# 数据集各维度特征含义
feature_meanings = {
    'longitude': '经度 - 房屋所在位置的经度坐标',
    'latitude': '纬度 - 房屋所在位置的纬度坐标',
    'housing_median_age': '房屋年龄中位数 - 该区域房屋的中位年龄',
    'total_rooms': '总房间数 - 该区域的总房间数',
    'total_bedrooms': '总卧室数 - 该区域的总卧室数',
    'population': '人口数 - 该区域的人口数量',
    'households': '家庭数 - 该区域的家庭数量',
    'median_income': '收入中位数 - 该区域的收入中位数(单位:万美元)',
    'median_house_value': '房价中位数 - 该区域房价中位数(目标预测值)',
    'ocean_proximity': '海洋距离 - 房屋与海洋的距离类别(分类特征)'
}

print("\n数据集特征含义:")
for feature, meaning in feature_meanings.items():
    print(f"  {feature}: {meaning}")

# 显示前5条样本数据
print("\n前5条样本数据:")
for i in range(min(5, len(data))):
    print(f"\n样本 {i+1}:")
    for key in headers:
        print(f"  {key}: {data[i][key]}")


# ==================== 2. 探索数据并预处理数据 ====================
print("\n" + "=" * 60)
print("2. 探索数据并预处理数据")
print("=" * 60)

# 2.1 观察数据类型和分布
print("\n2.1 数据类型和统计信息:")

# 统计ocean_proximity的类别
ocean_categories = defaultdict(int)
for row in data:
    if row['ocean_proximity']:
        ocean_categories[row['ocean_proximity']] += 1

print(f"\nocean_proximity类别分布:")
for category, count in ocean_categories.items():
    print(f"  {category}: {count} ({count/len(data)*100:.2f}%)")

# 统计数值特征的基本信息
numeric_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
                   'total_bedrooms', 'population', 'households', 'median_income', 
                   'median_house_value']

print(f"\n数值特征统计信息:")
for feature in numeric_features:
    values = []
    for row in data:
        if row[feature] and row[feature] != '':
            try:
                values.append(float(row[feature]))
            except:
                pass
    
    if values:
        values = np.array(values)
        print(f"\n{feature}:")
        print(f"  数量: {len(values)}")
        print(f"  最小值: {np.min(values):.2f}")
        print(f"  最大值: {np.max(values):.2f}")
        print(f"  均值: {np.mean(values):.2f}")
        print(f"  中位数: {np.median(values):.2f}")
        print(f"  标准差: {np.std(values):.2f}")


# 2.2 预处理数据
print("\n" + "-" * 60)
print("2.2 数据预处理:")

# 收集所有ocean_proximity类别
all_categories = sorted(list(ocean_categories.keys()))
print(f"\nocean_proximity将转换为one-hot编码,类别: {all_categories}")

# 处理数据并转换为numpy数组
def preprocess_data(data, all_categories):
    """
    预处理数据:
    1. 处理缺失值
    2. 将ocean_proximity转换为one-hot编码
    3. 分离特征和目标值
    """
    X_list = []
    y_list = []
    
    for row in data:
        # 检查是否有缺失值
        has_missing = False
        for feature in numeric_features:
            if not row[feature] or row[feature] == '':
                has_missing = True
                break
        
        if has_missing:
            continue
        
        # 提取数值特征
        features = []
        for feature in numeric_features[:-1]:  # 不包括median_house_value
            features.append(float(row[feature]))
        
        # one-hot编码ocean_proximity
        ocean_prox = row['ocean_proximity']
        for category in all_categories:
            features.append(1.0 if ocean_prox == category else 0.0)
        
        # 添加偏置项(常数项1)
        features.append(1.0)
        
        X_list.append(features)
        y_list.append(float(row['median_house_value']))
    
    return np.array(X_list), np.array(y_list)

X, y = preprocess_data(data, all_categories)

print(f"\n预处理后:")
print(f"  特征矩阵X形状: {X.shape}")
print(f"  目标向量y形状: {y.shape}")
print(f"  特征数量: {X.shape[1]-1} (包含{len(numeric_features)-1}个数值特征 + {len(all_categories)}个one-hot特征 + 1个偏置项)")


# 2.3 划分训练集和测试集 (70%训练, 30%测试)
print("\n" + "-" * 60)
print("2.3 划分训练集和测试集 (70%训练, 30%测试):")

def train_test_split(X, y, train_ratio=0.7, random_seed=42):
    """划分训练集和测试集"""
    np.random.seed(random_seed)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    train_size = int(n_samples * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_ratio=0.7, random_seed=42)

print(f"\n训练集:")
print(f"  X_train形状: {X_train.shape}")
print(f"  y_train形状: {y_train.shape}")
print(f"\n测试集:")
print(f"  X_test形状: {X_test.shape}")
print(f"  y_test形状: {y_test.shape}")


# ==================== 3. 求解模型参数 ====================
print("\n" + "=" * 60)
print("3. 求解模型参数")
print("=" * 60)

# 3.1 闭合形式解 (正规方程)
print("\n3.1 闭合形式参数求解 (正规方程法):")
print("公式: θ = (X^T X)^(-1) X^T y")

def closed_form_solution(X, y):
    """
    使用闭合形式(正规方程)求解线性回归参数
    θ = (X^T X)^(-1) X^T y
    为了数值稳定性,对特征进行标准化
    """
    # 特征标准化(不包括偏置项)
    X_mean = np.mean(X[:, :-1], axis=0)
    X_std = np.std(X[:, :-1], axis=0) + 1e-8
    X_normalized = X.copy()
    X_normalized[:, :-1] = (X[:, :-1] - X_mean) / X_std
    
    # 目标值标准化
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_normalized = (y - y_mean) / y_std
    
    # 计算 X^T X
    XtX = np.dot(X_normalized.T, X_normalized)
    
    # 为了数值稳定性,添加正则化项(Ridge回归)
    lambda_reg = 1e-5
    XtX_reg = XtX + lambda_reg * np.eye(XtX.shape[0])
    
    # 计算 (X^T X)^(-1)
    XtX_inv = np.linalg.inv(XtX_reg)
    
    # 计算 X^T y
    Xty = np.dot(X_normalized.T, y_normalized)
    
    # 计算标准化后的 θ
    theta = np.dot(XtX_inv, Xty)
    
    # 反标准化参数
    theta_original = theta.copy()
    theta_original[:-1] = theta[:-1] * y_std / X_std
    theta_original[-1] = y_mean + theta[-1] * y_std - np.sum(theta[:-1] * y_std * X_mean / X_std)
    
    return theta_original

print("\n开始求解...")
theta_closed = closed_form_solution(X_train, y_train)
print(f"求解完成!")
print(f"参数向量θ形状: {theta_closed.shape}")
print(f"\n前10个参数值:")
for i in range(min(10, len(theta_closed))):
    print(f"  θ[{i}] = {theta_closed[i]:.6f}")


# 3.2 梯度下降法
print("\n" + "-" * 60)
print("3.2 梯度下降参数优化:")

def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000, verbose=True):
    """
    使用梯度下降法求解线性回归参数
    
    参数:
        X: 特征矩阵
        y: 目标值向量
        learning_rate: 学习率
        n_iterations: 迭代次数
        verbose: 是否打印详细信息
    
    返回:
        theta: 参数向量
        cost_history: 损失函数历史记录
    """
    n_samples, n_features = X.shape
    
    # 初始化参数为0
    theta = np.zeros(n_features)
    
    # 记录损失函数值
    cost_history = []
    
    # 特征标准化(避免梯度爆炸)
    X_mean = np.mean(X[:, :-1], axis=0)  # 不包括偏置项
    X_std = np.std(X[:, :-1], axis=0) + 1e-8  # 避免除0
    X_normalized = X.copy()
    X_normalized[:, :-1] = (X[:, :-1] - X_mean) / X_std
    
    # 目标值标准化
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_normalized = (y - y_mean) / y_std
    
    for iteration in range(n_iterations):
        # 计算预测值
        predictions = np.dot(X_normalized, theta)
        
        # 计算误差
        errors = predictions - y_normalized
        
        # 计算梯度
        gradient = (1 / n_samples) * np.dot(X_normalized.T, errors)
        
        # 更新参数
        theta = theta - learning_rate * gradient
        
        # 计算损失 (均方误差)
        cost = (1 / (2 * n_samples)) * np.sum(errors ** 2)
        cost_history.append(cost)
        
        # 打印进度
        if verbose and (iteration % 100 == 0 or iteration == n_iterations - 1):
            print(f"  迭代 {iteration}/{n_iterations}, 损失: {cost:.6f}")
    
    # 反标准化参数
    theta_original = theta.copy()
    theta_original[:-1] = theta[:-1] * y_std / X_std
    theta_original[-1] = y_mean + theta[-1] * y_std - np.sum(theta[:-1] * y_std * X_mean / X_std)
    
    return theta_original, cost_history, X_mean, X_std, y_mean, y_std

print(f"\n参数设置:")
print(f"  学习率: 0.01")
print(f"  迭代次数: 1000")
print(f"\n开始梯度下降优化...")

theta_gd, cost_history, X_mean, X_std, y_mean, y_std = gradient_descent(
    X_train, y_train, learning_rate=0.01, n_iterations=1000, verbose=True
)

print(f"\n优化完成!")
print(f"参数向量θ形状: {theta_gd.shape}")
print(f"\n前10个参数值:")
for i in range(min(10, len(theta_gd))):
    print(f"  θ[{i}] = {theta_gd[i]:.6f}")

print(f"\n损失函数收敛情况:")
print(f"  初始损失: {cost_history[0]:.6f}")
print(f"  最终损失: {cost_history[-1]:.6f}")
print(f"  损失降低: {((cost_history[0] - cost_history[-1]) / cost_history[0] * 100):.2f}%")


# ==================== 4. 测试和评估模型 ====================
print("\n" + "=" * 60)
print("4. 测试和评估模型")
print("=" * 60)

def calculate_r2_score(y_true, y_pred):
    """
    计算R²决定系数
    R² = 1 - (SS_res / SS_tot)
    其中:
        SS_res = Σ(y_true - y_pred)²  残差平方和
        SS_tot = Σ(y_true - y_mean)²  总平方和
    """
    # 计算残差平方和
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # 计算总平方和
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    
    # 计算R²
    r2 = 1 - (ss_res / ss_tot)
    
    return r2

def calculate_mse(y_true, y_pred):
    """计算均方误差 MSE"""
    return np.mean((y_true - y_pred) ** 2)

def calculate_rmse(y_true, y_pred):
    """计算均方根误差 RMSE"""
    return np.sqrt(calculate_mse(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    """计算平均绝对误差 MAE"""
    return np.mean(np.abs(y_true - y_pred))


# 4.1 评估闭合形式解模型
print("\n4.1 闭合形式解模型评估:")

# 在训练集上的表现
y_train_pred_closed = np.dot(X_train, theta_closed)
r2_train_closed = calculate_r2_score(y_train, y_train_pred_closed)
mse_train_closed = calculate_mse(y_train, y_train_pred_closed)
rmse_train_closed = calculate_rmse(y_train, y_train_pred_closed)
mae_train_closed = calculate_mae(y_train, y_train_pred_closed)

print(f"\n训练集性能:")
print(f"  R² 分数: {r2_train_closed:.6f}")
print(f"  MSE (均方误差): {mse_train_closed:.2f}")
print(f"  RMSE (均方根误差): {rmse_train_closed:.2f}")
print(f"  MAE (平均绝对误差): {mae_train_closed:.2f}")

# 在测试集上的表现
y_test_pred_closed = np.dot(X_test, theta_closed)
r2_test_closed = calculate_r2_score(y_test, y_test_pred_closed)
mse_test_closed = calculate_mse(y_test, y_test_pred_closed)
rmse_test_closed = calculate_rmse(y_test, y_test_pred_closed)
mae_test_closed = calculate_mae(y_test, y_test_pred_closed)

print(f"\n测试集性能:")
print(f"  R² 分数: {r2_test_closed:.6f}")
print(f"  MSE (均方误差): {mse_test_closed:.2f}")
print(f"  RMSE (均方根误差): {rmse_test_closed:.2f}")
print(f"  MAE (平均绝对误差): {mae_test_closed:.2f}")


# 4.2 评估梯度下降法模型
print("\n" + "-" * 60)
print("4.2 梯度下降法模型评估:")

# 在训练集上的表现
y_train_pred_gd = np.dot(X_train, theta_gd)
r2_train_gd = calculate_r2_score(y_train, y_train_pred_gd)
mse_train_gd = calculate_mse(y_train, y_train_pred_gd)
rmse_train_gd = calculate_rmse(y_train, y_train_pred_gd)
mae_train_gd = calculate_mae(y_train, y_train_pred_gd)

print(f"\n训练集性能:")
print(f"  R² 分数: {r2_train_gd:.6f}")
print(f"  MSE (均方误差): {mse_train_gd:.2f}")
print(f"  RMSE (均方根误差): {rmse_train_gd:.2f}")
print(f"  MAE (平均绝对误差): {mae_train_gd:.2f}")

# 在测试集上的表现
y_test_pred_gd = np.dot(X_test, theta_gd)
r2_test_gd = calculate_r2_score(y_test, y_test_pred_gd)
mse_test_gd = calculate_mse(y_test, y_test_pred_gd)
rmse_test_gd = calculate_rmse(y_test, y_test_pred_gd)
mae_test_gd = calculate_mae(y_test, y_test_pred_gd)

print(f"\n测试集性能:")
print(f"  R² 分数: {r2_test_gd:.6f}")
print(f"  MSE (均方误差): {mse_test_gd:.2f}")
print(f"  RMSE (均方根误差): {rmse_test_gd:.2f}")
print(f"  MAE (平均绝对误差): {mae_test_gd:.2f}")


# 4.3 对比两种方法
print("\n" + "-" * 60)
print("4.3 两种方法对比:")

print(f"\n测试集R²分数对比:")
print(f"  闭合形式解: {r2_test_closed:.6f}")
print(f"  梯度下降法: {r2_test_gd:.6f}")
print(f"  差异: {abs(r2_test_closed - r2_test_gd):.6f}")


# 4.4 预测示例
print("\n" + "-" * 60)
print("4.4 预测示例 (使用闭合形式解模型):")

print(f"\n前5个测试样本的预测结果:")
print(f"{'真实值':>12} {'预测值':>12} {'误差':>12} {'误差率':>10}")
print("-" * 50)
for i in range(min(5, len(y_test))):
    true_val = y_test[i]
    pred_val = y_test_pred_closed[i]
    error = pred_val - true_val
    error_rate = (error / true_val) * 100
    print(f"{true_val:>12.2f} {pred_val:>12.2f} {error:>12.2f} {error_rate:>9.2f}%")


# ==================== 总结 ====================
print("\n" + "=" * 60)
print("实验总结")
print("=" * 60)

print(f"""
1. 数据集信息:
   - 总样本数: {len(data)}
   - 有效样本数: {len(X)}
   - 特征维度: {X.shape[1]}
   - 训练集样本: {len(X_train)}
   - 测试集样本: {len(X_test)}

2. 模型性能 (测试集):
   
   闭合形式解 (正规方程):
   - R² 分数: {r2_test_closed:.6f}
   - RMSE: {rmse_test_closed:.2f}
   
   梯度下降法:
   - R² 分数: {r2_test_gd:.6f}
   - RMSE: {rmse_test_gd:.2f}

3. 结论:
   - 两种方法都成功训练了线性回归模型
   - R²分数约为{r2_test_closed:.2f},说明模型能够解释约{r2_test_closed*100:.1f}%的房价变异
   - 闭合形式解计算快速但需要矩阵求逆
   - 梯度下降法更灵活,适合大规模数据
   - 两种方法得到的结果非常接近,验证了实现的正确性
""")

print("=" * 60)
print("实验完成!")
print("=" * 60)
