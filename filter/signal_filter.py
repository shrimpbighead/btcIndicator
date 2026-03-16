"""
AI信号过滤器
使用机器学习模型来过滤和增强交易信号
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from config import ML_CONFIDENCE_THRESHOLD, LOOK_AHEAD_BARS
import joblib
import os


class SignalFilter:
    """AI信号过滤器"""

    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = None

    def prepare_features(self, df, params):
        """
        准备特征矩阵

        Args:
            df: 带有指标的DataFrame
            params: 指标参数

        Returns:
            X: 特征矩阵
            y: 标签 (1=正确信号, 0=错误信号)
        """
        features = pd.DataFrame(index=df.index)

        # 添加技术指标作为特征
        for col in df.columns:
            if col in ['open', 'high', 'low', 'close', 'volume', 'timestamp']:
                continue
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                features[col] = df[col]

        # 添加市场状态特征
        # 波动率
        features['volatility'] = df['close'].pct_change().rolling(20).std()

        # 价格动量
        features['momentum_5'] = df['close'].pct_change(5)
        features['momentum_10'] = df['close'].pct_change(10)

        # 成交量变化
        features['volume_change'] = df['volume'].pct_change()
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

        # 价格位置
        features['price_position'] = (df['close'] - df['low'].rolling(20).min()) / (
            df['high'].rolling(20).max() - df['low'].rolling(20).min()
        )

        # 填充NaN
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)

        self.feature_columns = features.columns.tolist()
        return features

    def create_labels(self, df, signals, look_ahead=LOOK_AHEAD_BARS):
        """
        创建标签：信号是否正确

        Args:
            df: 价格数据
            signals: 交易信号
            look_ahead: 向前看的K线数

        Returns:
            标签数组
        """
        labels = []

        for i in range(len(df) - look_ahead):
            if signals.iloc[i] == 0:
                labels.append(0)  # 无信号
                continue

            # 计算未来价格变化
            future_return = (df['close'].iloc[i + look_ahead] - df['close'].iloc[i]) / df['close'].iloc[i]

            # 信号正确：买入信号且价格上涨，或卖出信号且价格下跌
            if (signals.iloc[i] == 1 and future_return > 0) or (signals.iloc[i] == -1 and future_return < 0):
                labels.append(1)  # 正确信号
            else:
                labels.append(0)  # 错误信号

        # 补充末尾
        while len(labels) < len(df):
            labels.append(0)

        return np.array(labels)

    def train(self, df, params, signals):
        """
        训练模型

        Args:
            df: 带有指标的DataFrame
            params: 指标参数
            signals: 初始交易信号
        """
        print("\nTraining AI Signal Filter...")

        # 准备特征
        X = self.prepare_features(df, params)

        # 创建标签
        y = self.create_labels(df, signals)

        # 分割训练集和测试集
        # 只使用有信号的数据
        mask = y != 0
        X_filtered = X[mask]
        y_filtered = y[mask]

        if len(y_filtered) < 50:
            print("Not enough data for training. Skipping filter.")
            return False

        # 分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
        )

        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 训练模型
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        else:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )

        self.model.fit(X_train_scaled, y_train)

        # 评估
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Filter Training Results:")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")

        # 如果准确率太低，不使用过滤器
        if accuracy < 0.55:
            print("Filter accuracy too low, not using filter.")
            return False

        self.is_trained = True
        return True

    def filter_signals(self, df, signals):
        """
        过滤信号

        Args:
            df: 带有指标的DataFrame
            signals: 原始交易信号

        Returns:
            filtered_signals: 过滤后的信号
            confidence: 置信度
        """
        if not self.is_trained or self.model is None:
            return signals, pd.Series(0.5, index=signals.index)

        # 准备特征
        X = self.prepare_features(df, {})

        # 标准化
        X_scaled = self.scaler.transform(X)

        # 预测概率
        proba = self.model.predict_proba(X_scaled)

        # 获取正确信号的概率
        if proba.shape[1] > 1:
            confidence = proba[:, 1]  # 正确信号的概率
        else:
            confidence = proba[:, 0]

        # 过滤信号
        filtered_signals = signals.copy()

        for i in range(len(signals)):
            if signals.iloc[i] != 0:
                # 只有高置信度时才保留信号
                if confidence[i] < ML_CONFIDENCE_THRESHOLD:
                    filtered_signals.iloc[i] = 0  # 过滤掉

        return filtered_signals, pd.Series(confidence, index=signals.index)

    def save(self, filepath):
        """保存模型"""
        if self.model is not None:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'feature_columns': self.feature_columns
            }, filepath)
            print(f"Model saved to {filepath}")

    def load(self, filepath):
        """加载模型"""
        if os.path.exists(filepath):
            data = joblib.load(filepath)
            self.model = data['model']
            self.scaler = data['scaler']
            self.model_type = data['model_type']
            self.feature_columns = data['feature_columns']
            self.is_trained = True
            print(f"Model loaded from {filepath}")
            return True
        return False
