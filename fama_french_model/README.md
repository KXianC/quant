# Fama-French三因子模型与市场转折点分析

## 项目概述

本项目实现了完整的Fama-French三因子模型，并结合市场转折点分析和SHAP值解释性分析，为量化投资研究提供了一个全面的工具。项目包含模型实现、数据生成、结果可视化等功能，可用于股票收益率建模、因子重要性分析和市场状态检测。

## 主要功能

1. **完整的Fama-French三因子模型实现**
   - 支持单只股票和全市场股票的模型拟合
   - 提供滚动窗口模型拟合功能
   - 计算模型R²、RMSE等评估指标

2. **市场转折点分析**
   - 自动检测牛熊市场转换点
   - 计算各市场状态持续时间
   - 可视化市场状态变化

3. **因子贡献分析**
   - 使用SHAP值分析各因子对收益率的贡献
   - 计算因子重要性排名
   - 可视化因子SHAP值分布

4. **残差分析**
   - 残差统计分析
   - 正态性检验（Jarque-Bera检验）
   - 自相关性检验（Durbin-Watson检验）
   - 残差分布和QQ图可视化

5. **丰富的可视化功能**
   - 因子收益率时间序列图
   - 市场状态标记图
   - 因子重要性条形图
   - 因子SHAP值分布图
   - 残差分析图表

6. **模拟数据生成**
   - 生成具有真实市场特性的模拟数据
   - 支持设置牛熊市场转换点
   - 生成符合三因子模型的股票收益率

## 文件结构

- `fama_french_three_factor_model.py`: 三因子模型核心实现
- `generate_sample_data.py`: 模拟数据生成工具
- `run_model_example.py`: 模型运行示例
- `README.md`: 项目说明文档

## 使用方法

### 快速开始

运行示例脚本以生成数据并执行完整的模型分析流程：

```bash
python run_model_example.py
```

这将生成模拟数据、拟合模型、分析因子贡献、检测市场状态，并生成各类可视化结果。

### 使用自己的数据

如果要使用自己的数据，需要准备以下两个CSV文件：

1. **股票数据文件**：包含以下列
   - `date`: 日期
   - `stock_id`: 股票ID
   - `return`: 股票收益率
   - `market_cap`: 市值
   - `book_to_market`: 账面市值比

2. **因子数据文件**：包含以下列
   - `date`: 日期
   - `mkt_rf`: 市场风险溢价
   - `smb`: 规模因子
   - `hml`: 价值因子
   - `rf`: 无风险利率

然后可以使用以下代码运行模型：

```python
from fama_french_three_factor_model import FamaFrenchThreeFactorModel

# 创建模型实例
model = FamaFrenchThreeFactorModel()

# 加载数据
model.load_data('your_stock_data.csv', 'your_factor_data.csv')

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
model.plot_results(output_dir='your_results_dir')
```

## 输出结果

运行模型后，将在指定的输出目录生成以下结果：

1. **因子分析图表**
   - `factor_returns.png`: 因子收益率时间序列
   - `factor_importance.png`: 因子重要性排名
   - `factor_shap_distribution.png`: 因子SHAP值分布

2. **市场状态分析**
   - `market_states.png`: 市场状态（牛熊市）标记图

3. **残差分析图表**
   - `residual_distribution.png`: 残差分布直方图
   - `residual_vs_predicted.png`: 残差与预测值散点图
   - `residual_qq_plot.png`: 残差QQ图

## 理论背景

Fama-French三因子模型是由Eugene Fama和Kenneth French提出的资产定价模型，是对传统CAPM模型的扩展。该模型认为股票收益率可以由三个因子解释：

1. **市场风险溢价(MKT-RF)**: 市场组合收益率与无风险利率之差
2. **规模因子(SMB)**: 小市值股票与大市值股票的收益率差异
3. **价值因子(HML)**: 高账面市值比股票与低账面市值比股票的收益率差异

模型的数学表达式为：

```
R_i - R_f = α_i + β_i,MKT(R_M - R_f) + β_i,SMB·SMB + β_i,HML·HML + ε_i
```

其中：
- R_i: 股票i的收益率
- R_f: 无风险利率
- R_M: 市场组合收益率
- SMB: 规模因子
- HML: 价值因子
- α_i: 截距项
- β_i,MKT, β_i,SMB, β_i,HML: 因子敏感度
- ε_i: 残差项

## 依赖库

- pandas
- numpy
- matplotlib
- statsmodels
- scipy
- shap

## 参考文献

1. Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.
2. Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1-22.
3. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.