#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成Fama-French三因子模型的模拟数据

这个脚本用于生成Fama-French三因子模型的模拟数据，包括：
1. 市场数据：日期、市场指数、牛熊市场状态
2. 股票数据：日期、股票ID、收益率、市场指数、市值、账面市值比、无风险利率
3. 因子数据：日期、市场因子(MKT-RF)、规模因子(SMB)、价值因子(HML)

生成的数据可以直接用于测试Fama-French三因子模型，并进行市场转折点分析。
数据生成过程中模拟了牛熊市场交替的情况，便于测试市场状态检测功能。

参考文献：
Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds.
Journal of Financial Economics, 33(1), 3-56.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns

# 设置随机种子，确保结果可重现
np.random.seed(42)


def generate_dates(start_date, periods):
    """
    生成日期序列
    
    生成连续的工作日（交易日）序列，用于模拟股票市场交易日
    
    参数:
    start_date (str): 开始日期，格式为'YYYY-MM-DD'
    periods (int): 期数（交易日数量）
    
    返回:
    pd.DatetimeIndex: 日期序列
    """
    # 使用pandas的date_range函数生成日期序列
    # freq='B'表示只包含工作日（Business days）
    return pd.date_range(start=start_date, periods=periods, freq='B')


def generate_market_data(dates, bull_bear_change_points=None, volatility=0.01):
    """
    生成市场数据
    
    生成市场指数数据，并模拟牛熊市场交替的情况
    
    参数:
    dates (pd.DatetimeIndex): 日期序列
    bull_bear_change_points (list): 牛熊市场转换点列表，格式为[(index, is_bull), ...]
                                  其中index是日期索引，is_bull是布尔值，True表示牛市，False表示熊市
    volatility (float): 波动率，控制市场指数的波动程度
    
    返回:
    pd.DataFrame: 市场数据，包含日期和市场指数
    """
    n = len(dates)
    
    # 初始化市场指数
    market_index = np.zeros(n)
    market_index[0] = 1000  # 初始指数值
    
    # 如果未指定牛熊市场转换点，则默认为一半牛市一半熊市
    if bull_bear_change_points is None:
        bull_bear_change_points = [(0, True), (n // 2, False)]
    
    # 生成市场指数
    current_state_idx = 0
    is_bull = bull_bear_change_points[current_state_idx][1]
    
    for i in range(1, n):
        # 检查是否需要切换市场状态
        if current_state_idx < len(bull_bear_change_points) - 1 and i >= bull_bear_change_points[current_state_idx + 1][0]:
            current_state_idx += 1
            is_bull = bull_bear_change_points[current_state_idx][1]
        
        # 根据市场状态生成收益率
        if is_bull:
            # 牛市：均值为正（0.08%/天，约合20%/年）
            daily_return = np.random.normal(0.0008, volatility)
        else:
            # 熊市：均值为负（-0.05%/天，约合-12.5%/年），且波动性更大
            daily_return = np.random.normal(-0.0005, volatility * 1.5)
        
        # 更新市场指数
        market_index[i] = market_index[i-1] * (1 + daily_return)
    
    # 创建市场数据框
    market_data = pd.DataFrame({
        'date': dates,
        'market_index': market_index
    })
    
    return market_data


def generate_stock_data(market_data, num_stocks=50):
    """
    生成股票数据
    
    基于市场数据生成多只股票的数据，包括收益率、市值、账面市值比等
    每只股票具有不同的市场敏感度（beta）和因子敏感度
    
    参数:
    market_data (pd.DataFrame): 市场数据，包含日期和市场指数
    num_stocks (int): 股票数量
    
    返回:
    pd.DataFrame: 股票数据，包含多只股票在不同日期的各项指标
    """
    dates = market_data['date']
    n_dates = len(dates)
    
    # 生成股票数据
    stock_data = []
    
    for stock_id in range(1, num_stocks + 1):
        # 生成股票特征
        # 市值在10亿到1000亿之间，对数正态分布
        # exp(24) ≈ 2.6亿，exp(24+1.5) ≈ 1200亿
        market_cap = np.exp(np.random.normal(24, 1.5))  
        
        # 账面市值比在0.1到5之间，对数正态分布
        # exp(0) = 1, exp(0+0.7) ≈ 2, exp(0-0.7) ≈ 0.5
        book_to_market = np.exp(np.random.normal(0, 0.7))  
        
        # 生成股票对市场因子、规模因子和价值因子的敏感度（beta）
        beta_market = np.random.normal(1.0, 0.3)  # 市场beta，均值为1（与市场同步）
        beta_size = np.random.normal(0.0, 0.5)    # 规模beta，均值为0
        beta_value = np.random.normal(0.0, 0.5)   # 价值beta，均值为0
        
        # 生成股票特有的波动（特质风险）
        stock_specific_volatility = np.random.uniform(0.01, 0.03)  # 1%~3%的特质波动
        
        # 初始化股票价格
        stock_price = np.zeros(n_dates)
        stock_price[0] = np.random.uniform(10, 100)  # 初始价格在10到100之间
        
        # 生成股票收益率和价格
        for i in range(1, n_dates):
            # 计算市场收益率
            market_return = market_data['market_index'][i] / market_data['market_index'][i-1] - 1
            
            # 生成规模因子和价值因子的模拟值
            # 这里简化处理，实际上因子值应该是基于所有股票计算得出
            size_factor = np.random.normal(0.0002, 0.005)    # 规模因子，均值略大于0
            value_factor = np.random.normal(0.0003, 0.005)   # 价值因子，均值略大于0
            
            # 计算股票收益率（基于三因子模型）
            # R_i = α_i + β_i,m * R_m + β_i,s * SMB + β_i,h * HML + ε_i
            stock_return = 0.0001 + beta_market * market_return + beta_size * size_factor + beta_value * value_factor
            
            # 添加股票特有的随机波动（特质风险）
            stock_return += np.random.normal(0, stock_specific_volatility)
            
            # 更新股票价格
            stock_price[i] = stock_price[i-1] * (1 + stock_return)
        
        # 计算每日收益率
        stock_returns = np.zeros(n_dates)
        stock_returns[1:] = np.diff(stock_price) / stock_price[:-1]  # 价格变化率
        
        # 添加到股票数据列表
        for i in range(n_dates):
            stock_data.append({
                'date': dates[i],
                'stock_id': f'STOCK_{stock_id:03d}',  # 股票ID格式：STOCK_001, STOCK_002, ...
                'return': stock_returns[i],            # 日收益率
                'market_index': market_data['market_index'][i],  # 市场指数
                # 市值和账面市值比每天略有变化
                'market_cap': market_cap * (1 + np.random.normal(0, 0.01)),  
                'book_to_market': book_to_market * (1 + np.random.normal(0, 0.005)),
                'risk_free_rate': 0.0001  # 假设无风险利率为0.01%/天（约2.5%/年）
            })
    
    # 创建股票数据框
    stock_df = pd.DataFrame(stock_data)
    
    return stock_df


def generate_factor_data(stock_data):
    """
    从股票数据生成因子数据
    
    基于股票数据计算Fama-French三因子：
    1. 市场因子(MKT-RF)：市场超额收益率
    2. 规模因子(SMB)：小市值股票组合收益率 - 大市值股票组合收益率
    3. 价值因子(HML)：高账面市值比股票组合收益率 - 低账面市值比股票组合收益率
    
    参数:
    stock_data (pd.DataFrame): 股票数据
    
    返回:
    pd.DataFrame: 因子数据，包含每个交易日的三个因子值
    """
    # 获取唯一日期列表
    dates = stock_data['date'].unique()
    
    # 初始化因子数据框
    factor_data = []
    
    for date in dates:
        # 获取当前日期的数据
        date_data = stock_data[stock_data['date'] == date].copy()
        
        # 计算市场因子 (MKT-RF)
        # 市场超额收益率 = 市场收益率 - 无风险利率
        market_return = date_data['market_index'].pct_change().mean()
        risk_free_rate = date_data['risk_free_rate'].mean()
        mkt_rf = market_return - risk_free_rate if not np.isnan(market_return) else 0
        
        # 按市值排序，划分大小市值组合
        # 使用百分比排名，将股票分为小市值（<50%）和大市值（>50%）
        date_data['size_rank'] = date_data['market_cap'].rank(pct=True)
        small_stocks = date_data[date_data['size_rank'] <= 0.5]  # 小市值股票
        big_stocks = date_data[date_data['size_rank'] > 0.5]     # 大市值股票
        
        # 计算规模因子 (SMB)
        # SMB = 小市值股票组合收益率 - 大市值股票组合收益率
        small_return = small_stocks['return'].mean()
        big_return = big_stocks['return'].mean()
        smb = small_return - big_return if not (np.isnan(small_return) or np.isnan(big_return)) else 0
        
        # 按账面市值比排序，划分高低价值组合
        # 将股票分为低价值（<30%）、中价值（30%-70%）和高价值（>70%）
        date_data['value_rank'] = date_data['book_to_market'].rank(pct=True)
        high_btm = date_data[date_data['value_rank'] > 0.7]    # 高账面市值比（高价值）
        mid_btm = date_data[(date_data['value_rank'] >= 0.3) & (date_data['value_rank'] <= 0.7)]  # 中价值
        low_btm = date_data[date_data['value_rank'] < 0.3]     # 低账面市值比（低价值）
        
        # 计算价值因子 (HML)
        # HML = 高账面市值比股票组合收益率 - 低账面市值比股票组合收益率
        high_return = high_btm['return'].mean()
        low_return = low_btm['return'].mean()
        hml = high_return - low_return if not (np.isnan(high_return) or np.isnan(low_return)) else 0
        
        # 添加到因子数据框
        factor_data.append({
            'date': date,
            'mkt_rf': mkt_rf,  # 市场因子
            'smb': smb,        # 规模因子
            'hml': hml         # 价值因子
        })
    
    # 创建因子数据框
    factor_df = pd.DataFrame(factor_data)
    
    return factor_df


def visualize_data(market_data, stock_data, factor_data, output_dir='sample_data'):
    """
    可视化生成的数据
    
    生成多种可视化图表，帮助理解数据特征：
    1. 市场指数走势图
    2. 随机选择的股票收益率图
    3. 三因子值的时间序列图
    4. 股票特征（市值和账面市值比）分布图
    
    参数:
    market_data (pd.DataFrame): 市场数据
    stock_data (pd.DataFrame): 股票数据
    factor_data (pd.DataFrame): 因子数据
    output_dir (str): 输出目录，默认为'sample_data'
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置绘图样式
    sns.set_style('whitegrid')
    
    # 1. 绘制市场指数
    plt.figure(figsize=(12, 6))
    plt.plot(market_data['date'], market_data['market_index'])
    plt.title('市场指数')
    plt.xlabel('日期')
    plt.ylabel('指数值')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'market_index.png'))
    plt.close()
    
    # 2. 绘制随机选择的10只股票的收益率
    stock_ids = stock_data['stock_id'].unique()
    # 随机选择最多10只股票
    selected_stocks = np.random.choice(stock_ids, min(10, len(stock_ids)), replace=False)
    
    plt.figure(figsize=(12, 6))
    for stock_id in selected_stocks:
        # 获取该股票的收益率序列
        stock_returns = stock_data[stock_data['stock_id'] == stock_id].set_index('date')['return']
        plt.plot(stock_returns.index, stock_returns.values, label=stock_id)
    
    plt.title('股票收益率')
    plt.xlabel('日期')
    plt.ylabel('收益率')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stock_returns.png'))
    plt.close()
    
    # 3. 绘制因子值
    plt.figure(figsize=(12, 6))
    plt.plot(factor_data['date'], factor_data['mkt_rf'], label='市场因子')
    plt.plot(factor_data['date'], factor_data['smb'], label='规模因子')
    plt.plot(factor_data['date'], factor_data['hml'], label='价值因子')
    plt.title('Fama-French三因子')
    plt.xlabel('日期')
    plt.ylabel('因子值')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'factors.png'))
    plt.close()
    
    # 4. 绘制市值和账面市值比分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 市值分布
    sns.histplot(stock_data['market_cap'], kde=True, ax=axes[0])
    axes[0].set_title('市值分布')
    axes[0].set_xlabel('市值')
    axes[0].set_ylabel('频率')
    
    # 账面市值比分布
    sns.histplot(stock_data['book_to_market'], kde=True, ax=axes[1])
    axes[1].set_title('账面市值比分布')
    axes[1].set_xlabel('账面市值比')
    axes[1].set_ylabel('频率')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stock_features_distribution.png'))
    plt.close()


def main():
    """
    主函数
    
    执行完整的数据生成流程：
    1. 设置参数
    2. 生成日期序列
    3. 生成市场数据（模拟牛熊市场交替）
    4. 生成股票数据
    5. 生成因子数据
    6. 可视化数据
    7. 保存数据到CSV文件
    """
    print("生成Fama-French三因子模型的模拟数据")
    print("="*50)
    
    # 设置参数
    start_date = '2018-01-01'  # 起始日期
    periods = 504              # 约2年的交易日
    num_stocks = 50            # 股票数量
    output_dir = 'sample_data' # 输出目录
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成日期序列
    print("生成日期序列...")
    dates = generate_dates(start_date, periods)
    
    # 设置牛熊市场转换点
    # 格式：[(index, is_bull), ...]
    # 这里设置了三个阶段：开始是牛市，中间是熊市，最后又回到牛市
    bull_bear_change_points = [
        (0, True),              # 开始是牛市
        (periods // 3, False), # 1/3处转为熊市
        (2 * periods // 3, True)  # 2/3处转回牛市
    ]
    
    # 生成市场数据
    print("生成市场数据...")
    market_data = generate_market_data(dates, bull_bear_change_points)
    
    # 生成股票数据
    print(f"生成{num_stocks}只股票的数据...")
    stock_data = generate_stock_data(market_data, num_stocks)
    
    # 生成因子数据
    print("生成因子数据...")
    factor_data = generate_factor_data(stock_data)
    
    # 可视化数据
    print("可视化数据...")
    visualize_data(market_data, stock_data, factor_data, output_dir)
    
    # 保存数据
    print("保存数据...")
    stock_data.to_csv(os.path.join(output_dir, 'stock_data.csv'), index=False)
    factor_data.to_csv(os.path.join(output_dir, 'factor_data.csv'), index=False)
    
    print(f"\n数据生成完成！所有数据已保存到{output_dir}目录。")
    print(f"生成了{len(dates)}个交易日、{num_stocks}只股票的数据。")
    print(f"数据时间范围: {dates[0].strftime('%Y-%m-%d')} 到 {dates[-1].strftime('%Y-%m-%d')}")
    print("\n您可以使用以下命令运行Fama-French三因子模型:")
    print(f"python fama_french_three_factor_model.py")
    print("然后在提示时输入:")
    print(f"股票数据路径: {os.path.join(output_dir, 'stock_data.csv')}")
    print(f"因子数据路径: {os.path.join(output_dir, 'factor_data.csv')}")


if __name__ == "__main__":
    main()