"""
技术指标计算模块
包含趋势、动量、波动、成交量指标
"""
import pandas as pd
import numpy as np


# ==================== 趋势类指标 ====================

def calculate_ma(df, period):
    """移动平均线"""
    return df['close'].rolling(window=period).mean()


def calculate_ema(df, period):
    """指数移动平均"""
    return df['close'].ewm(span=period, adjust=False).mean()


def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    MACD指标
    Returns: macd, signal, histogram
    """
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line

    return macd, signal_line, histogram


# ==================== 动量类指标 ====================

def calculate_rsi(df, period=14):
    """相对强弱指数"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_stochastic(df, k_period=14, d_period=3):
    """
    随机指标
    Returns: %K, %D
    """
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()

    k = 100 * (df['close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()

    return k, d


def calculate_cci(df, period=20):
    """商品通道指数"""
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())

    cci = (tp - sma_tp) / (0.015 * mad)
    return cci


# ==================== 波动类指标 ====================

def calculate_bollinger_bands(df, period=20, std_dev=2):
    """
    布林带
    Returns: upper, middle, lower
    """
    middle = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()

    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)

    return upper, middle, lower


def calculate_atr(df, period=14):
    """平均真实波幅"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    atr = true_range.rolling(window=period).mean()
    return atr


# ==================== 成交量类指标 ====================

def calculate_obv(df):
    """能量潮"""
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return obv


def calculate_vwap(df, period=20):
    """成交量加权平均价格"""
    tp = (df['high'] + df['low'] + df['close']) / 3
    cumsum_tp = tp.rolling(window=period).sum()
    cumsum_vol = df['volume'].rolling(window=period).sum()

    vwap = cumsum_tp / cumsum_vol
    return vwap


def calculate_volume_ma(df, period=20):
    """成交量移动平均"""
    return df['volume'].rolling(window=period).mean()


# ==================== 复合指标计算 ====================

def calculate_all_indicators(df, params):
    """
    根据参数计算所有指标

    Args:
        df: 包含OHLCV数据的DataFrame
        params: 参数字典

    Returns:
        带有所有指标的DataFrame
    """
    result = df.copy()

    # 趋势指标
    if 'ma_period' in params:
        result[f"ma_{params['ma_period']}"] = calculate_ma(df, params['ma_period'])

    if 'ema_period' in params:
        result[f"ema_{params['ema_period']}"] = calculate_ema(df, params['ema_period'])

    if 'macd_fast' in params and 'macd_slow' in params and 'macd_signal' in params:
        macd, signal, hist = calculate_macd(
            df,
            params['macd_fast'],
            params['macd_slow'],
            params['macd_signal']
        )
        result['macd'] = macd
        result['macd_signal'] = signal
        result['macd_hist'] = hist

    # 动量指标
    if 'rsi_period' in params:
        result[f"rsi_{params['rsi_period']}"] = calculate_rsi(df, params['rsi_period'])

    if 'stoch_k' in params and 'stoch_d' in params:
        k, d = calculate_stochastic(df, params['stoch_k'], params['stoch_d'])
        result['stoch_k'] = k
        result['stoch_d'] = d

    if 'cci_period' in params:
        result[f"cci_{params['cci_period']}"] = calculate_cci(df, params['cci_period'])

    # 波动指标
    if 'bb_period' in params and 'bb_std' in params:
        upper, middle, lower = calculate_bollinger_bands(
            df,
            params['bb_period'],
            params['bb_std']
        )
        result['bb_upper'] = upper
        result['bb_middle'] = middle
        result['bb_lower'] = lower
        # 计算BB位置
        result['bb_position'] = (df['close'] - lower) / (upper - lower)

    if 'atr_period' in params:
        result[f"atr_{params['atr_period']}"] = calculate_atr(df, params['atr_period'])

    # 成交量指标
    result['obv'] = calculate_obv(df)
    result['volume_ma'] = calculate_volume_ma(df, 20)

    return result


def generate_signals(df, params):
    """
    根据参数生成交易信号

    Args:
        df: 带有技术指标的DataFrame
        params: 参数字典

    Returns:
        signal列: 1=买入, -1=卖出, 0=持有
    """
    signals = pd.Series(0, index=df.index)

    # 确保有足够的数据
    min_period = 50  # 最小预热期
    if len(df) < min_period:
        return signals

    # 使用多个指标组合生成信号
    signal_count = 0

    # 1. MA金叉/死叉信号
    if 'ma_period' in params:
        ma_col = f"ma_{params['ma_period']}"
        if ma_col in df.columns:
            ma = df[ma_col]
            # 价格上穿MA买入
            buy_cond = (df['close'] > ma) & (df['close'].shift(1) <= ma.shift(1))
            # 价格下穿MA卖出
            sell_cond = (df['close'] < ma) & (df['close'].shift(1) >= ma.shift(1))
            signals[buy_cond] += 1
            signals[sell_cond] -= 1
            signal_count += 1

    # 2. EMA金叉/死叉信号
    if 'ema_period' in params:
        ema_col = f"ema_{params['ema_period']}"
        if ema_col in df.columns:
            ema = df[ema_col]
            buy_cond = (df['close'] > ema) & (df['close'].shift(1) <= ema.shift(1))
            sell_cond = (df['close'] < ema) & (df['close'].shift(1) >= ema.shift(1))
            signals[buy_cond] += 1
            signals[sell_cond] -= 1
            signal_count += 1

    # 3. MACD信号
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        # MACD上穿signal线买入
        buy_cond = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        # MACD下穿signal线卖出
        sell_cond = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        signals[buy_cond] += 1
        signals[sell_cond] -= 1
        signal_count += 1

    # 4. RSI信号
    if 'rsi_period' in params:
        rsi_col = f"rsi_{params['rsi_period']}"
        if rsi_col in df.columns:
            rsi = df[rsi_col]
            # RSI低于30超卖买入
            buy_cond = (rsi < 30) & (rsi.shift(1) >= 30)
            # RSI高于70超买卖出
            sell_cond = (rsi > 70) & (rsi.shift(1) <= 70)
            signals[buy_cond] += 1
            signals[sell_cond] -= 1
            signal_count += 1

    # 5. Stochastic信号
    if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
        # %K上穿%D买入
        buy_cond = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
        # %K下穿%D卖出
        sell_cond = (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))
        signals[buy_cond] += 1
        signals[sell_cond] -= 1
        signal_count += 1

    # 6. Bollinger Bands信号
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        # 价格突破下轨买入
        buy_cond = (df['close'] < df['bb_lower']) & (df['close'].shift(1) >= df['bb_lower'].shift(1))
        # 价格突破上轨卖出
        sell_cond = (df['close'] > df['bb_upper']) & (df['close'].shift(1) <= df['bb_upper'].shift(1))
        signals[buy_cond] += 1
        signals[sell_cond] -= 1
        signal_count += 1

    # 7. CCI信号
    if 'cci_period' in params:
        cci_col = f"cci_{params['cci_period']}"
        if cci_col in df.columns:
            cci = df[cci_col]
            buy_cond = (cci < -100) & (cci.shift(1) >= -100)
            sell_cond = (cci > 100) & (cci.shift(1) <= 100)
            signals[buy_cond] += 1
            signals[sell_cond] -= 1
            signal_count += 1

    # 综合信号：多数指标一致时才算信号
    if signal_count > 0:
        # 买入阈值：多数指标看多
        threshold = signal_count / 2
        signals = signals.apply(lambda x: 1 if x >= threshold else (-1 if x <= -threshold else 0))

    return signals


def get_feature_vector(df, params):
    """
    获取用于ML模型的特征向量

    Returns:
        DataFrame with normalized features
    """
    features = pd.DataFrame(index=df.index)

    # 添加所有可用的指标作为特征
    for col in df.columns:
        if col in ['open', 'high', 'low', 'close', 'volume']:
            continue
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            # 归一化
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                features[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                features[col] = 0.5

    # 添加价格变化率
    features['price_change'] = df['close'].pct_change()
    features['volume_change'] = df['volume'].pct_change()

    # 填充NaN
    features = features.fillna(0)

    return features
