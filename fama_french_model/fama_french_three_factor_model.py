#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fama-French三因子模型与市场转折点分析

这个脚本实现了Fama-French三因子模型，并结合市场转折点分析，同时支持使用SHAP进行因子贡献分析。
主要功能包括：
1. 数据加载与预处理 - 加载股票和因子数据，处理缺失值和异常值
2. 因子构建 - 市场因子(MKT-RF)、规模因子(SMB)和价值因子(HML)
3. 模型拟合与评估 - 拟合三因子模型并计算R²和RMSE等评估指标
4. 市场状态检测 - 识别牛熊市场转折点及其持续时间
5. 因子贡献分析 - 使用SHAP值分析各因子对股票收益率的贡献
6. 残差分析 - 分析模型残差的分布特性和自相关性
7. 结果可视化 - 生成各类分析图表并保存

参考文献：
Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds.
Journal of Financial Economics, 33(1), 3-56.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import warnings

# 尝试导入SHAP库，如果不可用则提供警告
# SHAP (SHapley Additive exPlanations) 用于解释机器学习模型的预测
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP库未安装，将无法进行因子贡献分析。请使用 'pip install shap' 安装。")

# 忽略警告，避免不必要的警告信息干扰输出
warnings.filterwarnings('ignore')


class FamaFrenchThreeFactorModel:
    """
    Fama-French三因子模型实现
    
    该类实现了完整的Fama-French三因子模型分析流程，包括：
    - 数据加载和预处理
    - 模型拟合（普通拟合和滚动窗口拟合）
    - 市场状态检测（牛熊市场识别）
    - 因子贡献分析（基于SHAP值）
    - 残差分析
    - 结果可视化
    
    三因子模型方程：
    R_i - R_f = α_i + β_i,m(R_m - R_f) + β_i,s·SMB + β_i,h·HML + ε_i
    
    其中：
    - R_i：股票i的收益率
    - R_f：无风险利率
    - R_m：市场收益率
    - SMB：规模因子（小市值股票收益率 - 大市值股票收益率）
    - HML：价值因子（高账面市值比股票收益率 - 低账面市值比股票收益率）
    - α_i, β_i,m, β_i,s, β_i,h：回归系数
    - ε_i：残差项
    """
    
    def __init__(self):
        """
        初始化模型
        
        初始化模型的各个属性，包括数据存储、模型结果和分析结果等
        """
        # 数据存储
        self.stock_data = None    # 股票数据
        self.factor_data = None   # 因子数据
        self.merged_data = None   # 合并后的数据
        
        # 模型和结果
        self.model = None             # 模型对象
        self.results = None           # 模型拟合结果
        self.rolling_results = None   # 滚动窗口模型结果
        
        # 分析结果
        self.market_states = None     # 市场状态（牛熊市场）
        self.shap_values = None       # SHAP值（因子贡献）
    
    def load_data(self, stock_data_path, factor_data_path):
        """
        加载股票数据和因子数据
        
        从指定路径加载股票数据和因子数据，并进行基本的数据检查和预览
        
        参数:
        stock_data_path (str): 股票数据CSV文件路径
        factor_data_path (str): 因子数据CSV文件路径
        """
        print("加载数据...")
        # 读取CSV文件
        self.stock_data = pd.read_csv(stock_data_path)
        self.factor_data = pd.read_csv(factor_data_path)
        
        # 确保日期列是日期类型，便于后续处理
        self.stock_data['date'] = pd.to_datetime(self.stock_data['date'])
        self.factor_data['date'] = pd.to_datetime(self.factor_data['date'])
        
        # 输出数据基本信息
        print(f"加载了{len(self.stock_data)}条股票数据记录")
        print(f"加载了{len(self.factor_data)}条因子数据记录")
        
        # 数据预览，帮助用户了解数据结构
        print("\n股票数据预览:")
        print(self.stock_data.head())
        print("\n因子数据预览:")
        print(self.factor_data.head())
    
    def preprocess_data(self):
        """
        数据预处理
        
        对加载的数据进行预处理，包括：
        1. 合并股票数据和因子数据
        2. 计算超额收益率
        3. 检查并处理缺失值
        4. 检查异常值
        """
        print("\n数据预处理...")
        
        # 合并股票数据和因子数据（基于日期）
        self.merged_data = pd.merge(self.stock_data, self.factor_data, on='date')
        
        # 计算超额收益率（股票收益率减去无风险利率）
        # 这是三因子模型的因变量
        self.merged_data['excess_return'] = self.merged_data['return'] - self.merged_data['risk_free_rate']
        
        # 检查缺失值
        missing_values = self.merged_data.isnull().sum()
        if missing_values.sum() > 0:
            print("发现缺失值:")
            print(missing_values[missing_values > 0])
            print("删除缺失值...")
            # 删除包含缺失值的行
            self.merged_data = self.merged_data.dropna()
        
        # 检查异常值（使用1%和99%分位数作为界限）
        print("检查异常值...")
        for col in ['excess_return', 'mkt_rf', 'smb', 'hml']:
            q1 = self.merged_data[col].quantile(0.01)  # 1%分位数
            q3 = self.merged_data[col].quantile(0.99)  # 99%分位数
            # 找出异常值（小于1%分位数或大于99%分位数的值）
            outliers = self.merged_data[(self.merged_data[col] < q1) | (self.merged_data[col] > q3)]
            print(f"  {col}: 发现{len(outliers)}个异常值 ({len(outliers)/len(self.merged_data):.2%})")
        
        # 输出预处理后的数据形状
        print(f"预处理后的数据形状: {self.merged_data.shape}")
    
    def fit_model(self, stock_id=None):
        """
        拟合三因子模型
        
        使用OLS（普通最小二乘法）拟合Fama-French三因子模型
        可以选择对单只股票或所有股票进行拟合
        
        参数:
        stock_id (str, optional): 股票ID，如果为None则对所有股票进行拟合
        
        返回:
        dict: 包含模型参数、p值、R²和RMSE等结果的字典
        """
        print("\n拟合三因子模型...")
        
        if stock_id is not None:
            # 对单只股票拟合模型
            data = self.merged_data[self.merged_data['stock_id'] == stock_id].copy()
            if len(data) == 0:
                print(f"未找到股票ID: {stock_id}")
                return None
            
            print(f"对股票 {stock_id} 拟合模型")
        else:
            # 对所有股票拟合模型（池化回归）
            data = self.merged_data.copy()
            print("对所有股票拟合模型（池化回归）")
        
        # 准备自变量（三个因子）和因变量（超额收益率）
        X = data[['mkt_rf', 'smb', 'hml']]  # 三因子：市场、规模、价值
        y = data['excess_return']           # 超额收益率
        
        # 添加常数项（截距）
        X = sm.add_constant(X)
        
        # 拟合OLS模型
        self.model = sm.OLS(y, X)
        self.results = self.model.fit()
        
        # 打印模型结果摘要
        print("\n模型结果摘要:")
        print(self.results.summary())
        
        # 计算模型性能指标
        y_pred = self.results.predict(X)  # 模型预测值
        r2 = r2_score(y, y_pred)          # R²（决定系数）
        rmse = np.sqrt(mean_squared_error(y, y_pred))  # RMSE（均方根误差）
        
        print(f"R²: {r2:.4f}")
        print(f"RMSE: {rmse:.6f}")
        
        # 返回模型结果
        return {
            'params': self.results.params,      # 模型参数（系数）
            'pvalues': self.results.pvalues,    # 参数的p值
            'r2': r2,                           # R²
            'rmse': rmse,                       # RMSE
            'results': self.results              # 完整的结果对象
        }
    
    def fit_rolling_model(self, stock_id, window=60):
        """
        拟合滚动窗口三因子模型
        
        使用滚动窗口方法拟合三因子模型，可以观察模型参数随时间的变化
        
        参数:
        stock_id (str): 股票ID
        window (int): 滚动窗口大小（默认为60个交易日，约3个月）
        
        返回:
        pd.DataFrame: 包含滚动窗口模型参数和R²的数据框
        """
        print(f"\n对股票 {stock_id} 拟合滚动窗口三因子模型 (窗口大小: {window})...")
        
        # 筛选特定股票的数据
        data = self.merged_data[self.merged_data['stock_id'] == stock_id].copy()
        if len(data) == 0:
            print(f"未找到股票ID: {stock_id}")
            return None
        
        # 按日期排序，确保时间序列的正确性
        data = data.sort_values('date')
        
        # 准备自变量和因变量
        X = data[['mkt_rf', 'smb', 'hml']]  # 三因子
        y = data['excess_return']           # 超额收益率
        
        # 添加常数项
        X = sm.add_constant(X)
        
        # 拟合滚动窗口模型
        # RollingOLS会在每个窗口内拟合一个OLS模型
        rolling_model = RollingOLS(y, X, window=window)
        self.rolling_results = rolling_model.fit()
        
        # 提取滚动窗口参数
        rolling_params = self.rolling_results.params
        
        # 计算滚动窗口R²
        # 对每个窗口计算模型的R²
        rolling_r2 = []
        for i in range(window, len(data)):
            # 获取当前窗口的数据
            window_X = X.iloc[i-window:i]
            window_y = y.iloc[i-window:i]
            # 使用当前窗口的参数计算预测值
            window_y_pred = np.sum(window_X.values * rolling_params.iloc[i-window].values, axis=1)
            # 计算R²
            r2 = r2_score(window_y, window_y_pred)
            rolling_r2.append(r2)
        
        # 创建结果数据框
        rolling_results_df = pd.DataFrame({
            'date': data['date'].iloc[window:].values,
            'alpha': rolling_params['const'].values,         # 截距（alpha）
            'beta_mkt': rolling_params['mkt_rf'].values,     # 市场因子系数
            'beta_smb': rolling_params['smb'].values,        # 规模因子系数
            'beta_hml': rolling_params['hml'].values,        # 价值因子系数
            'r2': rolling_r2                                # R²
        })
        
        print(f"滚动窗口模型结果形状: {rolling_results_df.shape}")
        print("滚动窗口模型结果预览:")
        print(rolling_results_df.head())
        
        return rolling_results_df
    
    def detect_market_states(self, window=60, threshold=0.1):
        """
        检测市场状态（牛熊市场转折点）
        
        通过分析市场指数的滚动收益率来检测牛熊市场转折点
        
        参数:
        window (int): 滚动窗口大小（默认为60个交易日，约3个月）
        threshold (float): 市场状态转换阈值（默认为0.1，即10%）
        
        返回:
        pd.DataFrame: 包含市场转折点信息的数据框
        """
        print(f"\n检测市场状态 (窗口大小: {window}, 阈值: {threshold})...")
        
        # 获取唯一日期列表并排序
        dates = sorted(self.factor_data['date'].unique())
        
        # 计算市场指数的滚动收益率
        # 首先按日期分组计算平均市场指数
        market_data = self.stock_data.groupby('date')['market_index'].mean().reset_index()
        # 计算市场指数的滞后值
        market_data['market_index_lag'] = market_data['market_index'].shift(1)
        # 计算日收益率
        market_data['market_return'] = market_data['market_index'] / market_data['market_index_lag'] - 1
        market_data = market_data.dropna()
        
        # 计算滚动窗口收益率（用于判断牛熊市场）
        market_data['rolling_return'] = market_data['market_return'].rolling(window=window).mean()
        market_data = market_data.dropna()
        
        # 确定市场状态：滚动收益率为正表示牛市，为负表示熊市
        market_data['is_bull'] = market_data['rolling_return'] > 0
        
        # 检测市场状态转换点
        market_data['prev_is_bull'] = market_data['is_bull'].shift(1)
        # 状态变化：当前状态与前一状态不同
        market_data['state_change'] = market_data['is_bull'] != market_data['prev_is_bull']
        market_data = market_data.dropna()
        
        # 找出转换点（状态发生变化的点）
        turning_points = market_data[market_data['state_change']].copy()
        
        # 计算每个市场状态的持续时间
        if len(turning_points) > 0:
            # 计算下一个转换点的日期
            turning_points['next_date'] = turning_points['date'].shift(-1)
            # 计算持续天数
            turning_points['duration'] = (turning_points['next_date'] - turning_points['date']).dt.days
            turning_points = turning_points.dropna()
        
        # 保存市场状态
        self.market_states = market_data[['date', 'is_bull', 'rolling_return']].copy()
        
        # 输出转换点信息
        print(f"检测到{len(turning_points)}个市场状态转换点")
        if len(turning_points) > 0:
            print("市场状态转换点:")
            for _, row in turning_points.iterrows():
                state = "牛市" if row['is_bull'] else "熊市"
                print(f"  {row['date'].strftime('%Y-%m-%d')}: 转为{state}, 持续{row['duration']:.0f}天")
        
        return turning_points
    
    def analyze_factor_contribution(self, stock_id=None):
        """
        使用SHAP值分析因子贡献
        
        SHAP值可以解释每个因子对模型预测的贡献程度
        
        参数:
        stock_id (str, optional): 股票ID，如果为None则对所有股票进行分析
        
        返回:
        dict: 包含SHAP值和因子重要性的字典
        """
        # 检查SHAP库是否可用
        if not SHAP_AVAILABLE:
            print("\nSHAP库未安装，无法进行因子贡献分析。请使用 'pip install shap' 安装。")
            return None
        
        print("\n使用SHAP值分析因子贡献...")
        
        if stock_id is not None:
            # 对单只股票进行分析
            data = self.merged_data[self.merged_data['stock_id'] == stock_id].copy()
            if len(data) == 0:
                print(f"未找到股票ID: {stock_id}")
                return None
            
            print(f"对股票 {stock_id} 分析因子贡献")
        else:
            # 对所有股票进行分析
            data = self.merged_data.copy()
            print("对所有股票分析因子贡献")
        
        # 准备自变量和因变量
        X = data[['mkt_rf', 'smb', 'hml']]  # 三因子
        y = data['excess_return']           # 超额收益率
        
        # 使用线性回归模型（与OLS等价，但符合SHAP的接口要求）
        model = LinearRegression()
        model.fit(X, y)
        
        # 计算SHAP值
        # LinearExplainer专门用于线性模型的SHAP值计算
        explainer = shap.LinearExplainer(model, X)
        self.shap_values = explainer.shap_values(X)
        
        # 计算每个因子的平均绝对SHAP值（作为因子重要性的度量）
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        factor_importance = pd.DataFrame({
            'factor': ['mkt_rf', 'smb', 'hml'],
            'importance': mean_abs_shap
        })
        # 按重要性降序排序
        factor_importance = factor_importance.sort_values('importance', ascending=False)
        
        # 输出因子重要性
        print("\n因子重要性 (基于SHAP值):")
        for _, row in factor_importance.iterrows():
            print(f"  {row['factor']}: {row['importance']:.6f}")
        
        return {
            'shap_values': self.shap_values,
            'factor_importance': factor_importance
        }
    
    def analyze_residuals(self):
        """
        分析模型残差
        
        对模型残差进行统计分析，包括：
        1. 基本统计量（均值、标准差、偏度、峰度等）
        2. Jarque-Bera正态性检验
        3. Durbin-Watson自相关检验
        
        返回:
        dict: 包含残差分析结果的字典
        """
        # 检查模型是否已拟合
        if self.results is None:
            print("\n请先拟合模型再进行残差分析")
            return None
        
        print("\n分析模型残差...")
        
        # 获取残差（实际值减去预测值）
        X = sm.add_constant(self.merged_data[['mkt_rf', 'smb', 'hml']])
        y = self.merged_data['excess_return']
        y_pred = self.results.predict(X)
        residuals = y - y_pred
        
        # 计算残差的基本统计量
        residual_stats = {
            'mean': residuals.mean(),           # 均值（理想情况下应接近0）
            'std': residuals.std(),             # 标准差
            'min': residuals.min(),             # 最小值
            'max': residuals.max(),             # 最大值
            'skewness': stats.skew(residuals),  # 偏度（衡量分布的对称性）
            'kurtosis': stats.kurtosis(residuals)  # 峰度（衡量分布的尖峭程度）
        }
        
        # 输出残差统计量
        print("残差统计量:")
        for key, value in residual_stats.items():
            print(f"  {key}: {value:.6f}")
        
        # 进行Jarque-Bera正态性检验
        # 该检验用于判断残差是否服从正态分布
        jb_stat, jb_pvalue = stats.jarque_bera(residuals)
        print(f"\nJarque-Bera正态性检验:")
        print(f"  统计量: {jb_stat:.4f}")
        print(f"  p值: {jb_pvalue:.6f}")
        if jb_pvalue < 0.05:
            print("  结论: 残差不服从正态分布 (p < 0.05)")
        else:
            print("  结论: 残差服从正态分布 (p >= 0.05)")
        
        # 进行Durbin-Watson自相关检验
        # 该检验用于判断残差是否存在自相关性
        dw_stat = sm.stats.stattools.durbin_watson(residuals)
        print(f"\nDurbin-Watson自相关检验:")
        print(f"  统计量: {dw_stat:.4f}")
        if dw_stat < 1.5:
            print("  结论: 存在正自相关 (DW < 1.5)")
        elif dw_stat > 2.5:
            print("  结论: 存在负自相关 (DW > 2.5)")
        else:
            print("  结论: 不存在自相关 (1.5 <= DW <= 2.5)")
        
        return {
            'residuals': residuals,
            'stats': residual_stats,
            'jarque_bera': (jb_stat, jb_pvalue),
            'durbin_watson': dw_stat
        }
    
    def plot_results(self, output_dir='results'):
        """
        绘制模型结果图表
        
        生成各种可视化图表，包括：
        1. 因子收益率时间序列
        2. 市场状态（牛熊市场）
        3. 因子重要性（基于SHAP值）
        4. 残差分析图
        
        参数:
        output_dir (str): 输出目录，默认为'results'
        """
        print(f"\n绘制模型结果图表到{output_dir}目录...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置绘图样式
        sns.set_style('whitegrid')
        
        # 1. 绘制因子收益率时间序列
        plt.figure(figsize=(12, 6))
        plt.plot(self.factor_data['date'], self.factor_data['mkt_rf'], label='市场因子')
        plt.plot(self.factor_data['date'], self.factor_data['smb'], label='规模因子')
        plt.plot(self.factor_data['date'], self.factor_data['hml'], label='价值因子')
        plt.title('Fama-French三因子收益率')
        plt.xlabel('日期')
        plt.ylabel('因子收益率')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'factor_returns.png'))
        plt.close()
        
        # 2. 绘制市场状态（如果已检测）
        if self.market_states is not None:
            plt.figure(figsize=(12, 6))
            plt.plot(self.market_states['date'], self.market_states['rolling_return'])
            plt.axhline(y=0, color='r', linestyle='--')  # 添加y=0的水平线
            
            # 标记牛熊市场
            bull_periods = self.market_states[self.market_states['is_bull']]
            bear_periods = self.market_states[~self.market_states['is_bull']]
            
            # 使用散点图标记牛熊市场点
            plt.scatter(bull_periods['date'], bull_periods['rolling_return'], 
                       color='green', alpha=0.5, label='牛市')
            plt.scatter(bear_periods['date'], bear_periods['rolling_return'], 
                       color='red', alpha=0.5, label='熊市')
            
            plt.title('市场状态检测')
            plt.xlabel('日期')
            plt.ylabel('滚动窗口收益率')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'market_states.png'))
            plt.close()
        
        # 3. 绘制因子重要性（如果有SHAP值分析）
        if self.shap_values is not None and SHAP_AVAILABLE:
            # 绘制因子重要性条形图
            plt.figure(figsize=(10, 6))
            shap.summary_plot(self.shap_values, self.merged_data[['mkt_rf', 'smb', 'hml']], 
                             plot_type="bar", show=False)
            plt.title('因子重要性 (基于SHAP值)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'factor_importance.png'))
            plt.close()
            
            # 绘制SHAP值分布图
            plt.figure(figsize=(10, 8))
            shap.summary_plot(self.shap_values, self.merged_data[['mkt_rf', 'smb', 'hml']], show=False)
            plt.title('因子SHAP值分布')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'factor_shap_distribution.png'))
            plt.close()
        
        # 4. 绘制残差分析图（如果已进行残差分析）
        if self.results is not None:
            # 获取残差
            X = sm.add_constant(self.merged_data[['mkt_rf', 'smb', 'hml']])
            y = self.merged_data['excess_return']
            y_pred = self.results.predict(X)
            residuals = y - y_pred
            
            # 残差分布直方图
            plt.figure(figsize=(12, 6))
            sns.histplot(residuals, kde=True)  # 添加核密度估计曲线
            plt.title('残差分布')
            plt.xlabel('残差')
            plt.ylabel('频率')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'residual_distribution.png'))
            plt.close()
            
            # 残差与预测值散点图（用于检查异方差性）
            plt.figure(figsize=(12, 6))
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')  # 添加y=0的水平线
            plt.title('残差与预测值散点图')
            plt.xlabel('预测值')
            plt.ylabel('残差')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'residual_vs_predicted.png'))
            plt.close()
            
            # 残差QQ图（用于检查正态性）
            plt.figure(figsize=(10, 10))
            stats.probplot(residuals, dist="norm", plot=plt)
            plt.title('残差QQ图')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'residual_qq_plot.png'))
            plt.close()
        
        print(f"图表已保存到{output_dir}目录")


def main():
    """
    主函数
    
    提供交互式界面，让用户输入数据路径并运行完整的分析流程
    """
    print("Fama-French三因子模型与市场转折点分析")
    print("=" * 50)
    
    # 创建模型实例
    model = FamaFrenchThreeFactorModel()
    
    # 获取数据路径
    stock_data_path = input("请输入股票数据路径: ")
    factor_data_path = input("请输入因子数据路径: ")
    
    # 加载数据
    model.load_data(stock_data_path, factor_data_path)
    
    # 数据预处理
    model.preprocess_data()
    
    # 拟合模型
    model.fit_model()
    
    # 检测市场状态
    model.detect_market_states()
    
    # 分析因子贡献
    model.analyze_factor_contribution()
    
    # 分析残差
    model.analyze_residuals()
    
    # 绘制结果
    model.plot_results()
    
    print("\n分析完成！")


if __name__ == "__main__":
    main()