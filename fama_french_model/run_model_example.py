#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行Fama-French三因子模型示例

这个脚本提供了一个完整的Fama-French三因子模型分析流程示例，包括：
1. 生成模拟的股票数据和因子数据
2. 加载和预处理数据
3. 拟合三因子模型
4. 检测市场状态（牛熊市场转折点）
5. 分析因子贡献（使用SHAP值）
6. 分析模型残差
7. 可视化分析结果

这个示例可以作为使用Fama-French三因子模型进行量化分析的参考。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 导入自定义模块
from generate_sample_data import generate_dates, generate_market_data, generate_stock_data, generate_factor_data, visualize_data
from fama_french_three_factor_model import FamaFrenchThreeFactorModel


def generate_sample_dataset(output_dir='sample_data', num_stocks=30, periods=500, seed=42):
    """
    生成示例数据集
    
    生成用于Fama-French三因子模型分析的模拟数据集，包括市场数据、股票数据和因子数据。
    数据生成过程中模拟了多次牛熊市场转换，便于测试市场状态检测功能。
    
    参数:
    output_dir (str): 输出目录，默认为'sample_data'
    num_stocks (int): 股票数量，默认为30只
    periods (int): 交易日数量，默认为500个交易日（约2年）
    seed (int): 随机种子，确保结果可重现，默认为42
    
    返回:
    tuple: (股票数据路径, 因子数据路径) - 保存的数据文件路径
    """
    print("生成示例数据集...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置随机种子，确保结果可重现
    np.random.seed(seed)
    
    # 生成日期序列（从2020年1月1日开始）
    dates = generate_dates(start_date='2020-01-01', periods=periods)
    
    # 设置牛熊市场转换点
    # 这里设置了5个阶段，模拟市场的多次转换
    bull_bear_change_points = [
        (0, True),                  # 开始是牛市
        (periods // 6, False),      # 1/6处转为熊市
        (periods // 3, True),       # 1/3处转为牛市
        (2 * periods // 3, False),  # 2/3处转为熊市
        (5 * periods // 6, True)    # 5/6处转回牛市
    ]
    
    # 生成市场数据
    market_data = generate_market_data(dates, bull_bear_change_points)
    
    # 生成股票数据
    stock_data = generate_stock_data(market_data, num_stocks=num_stocks)
    
    # 生成因子数据
    factor_data = generate_factor_data(stock_data)
    
    # 保存数据到CSV文件
    stock_data_path = os.path.join(output_dir, 'stock_data.csv')
    factor_data_path = os.path.join(output_dir, 'factor_data.csv')
    
    stock_data.to_csv(stock_data_path, index=False)
    factor_data.to_csv(factor_data_path, index=False)
    
    print(f"股票数据已保存到: {stock_data_path}")
    print(f"因子数据已保存到: {factor_data_path}")
    
    # 可视化数据，生成图表
    visualize_data(market_data, stock_data, factor_data, output_dir=output_dir)
    
    return stock_data_path, factor_data_path


def run_model_example():
    """
    运行模型示例
    
    执行完整的Fama-French三因子模型分析流程：
    1. 生成示例数据
    2. 创建模型实例
    3. 加载和预处理数据
    4. 拟合模型
    5. 检测市场状态
    6. 分析因子贡献
    7. 分析残差
    8. 可视化结果
    
    这个函数展示了如何使用FamaFrenchThreeFactorModel类进行完整的分析。
    """
    print("Fama-French三因子模型示例")
    print("=" * 50)
    
    # 生成示例数据
    # 这一步会创建模拟的股票数据和因子数据，并保存到sample_data目录
    stock_data_path, factor_data_path = generate_sample_dataset()
    
    # 创建模型实例
    model = FamaFrenchThreeFactorModel()
    
    # 加载数据
    # 从生成的CSV文件中加载股票数据和因子数据
    model.load_data(stock_data_path, factor_data_path)
    
    # 数据预处理
    # 合并数据、处理缺失值和异常值
    model.preprocess_data()
    
    # 拟合模型
    # 使用OLS拟合三因子模型
    model.fit_model()
    
    # 检测市场状态
    # 识别牛熊市场转折点
    model.detect_market_states()
    
    # 分析因子贡献
    # 使用SHAP值分析各因子的贡献
    model.analyze_factor_contribution()
    
    # 分析残差
    # 检查残差的分布特性和自相关性
    model.analyze_residuals()
    
    # 绘制结果
    # 生成各类分析图表并保存到model_results目录
    model.plot_results(output_dir='model_results')
    
    print("\n示例运行完成！")
    print("模型结果已保存到 'model_results' 目录")
    print("示例数据已保存到 'sample_data' 目录")


if __name__ == "__main__":
    run_model_example()