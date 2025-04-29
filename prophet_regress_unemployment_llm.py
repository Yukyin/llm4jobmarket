import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from prophet import Prophet



employ_data_path='/ossfs/workspace/proj/work_llm/work_data_monthly/UNE_TUNE_SEX_AGE_EDU_NB_M-20241121T1534.csv'


# Step 1: Load data
employment_data = pd.read_csv(employ_data_path)  # 图3数据


# Step 2: Preprocess employment data
employment_data['time'] = pd.to_datetime(employment_data['time'], format='%YM%m')
employment_data = employment_data[employment_data['classif1.label'].str.contains('Total', na=False)]  # 筛选包含 "Total" 的行
employment_data = employment_data[employment_data['sex.label'].str.contains('Total', na=False)]  # 筛选包含 "Total" 的行


employment_data = employment_data.groupby('time')['obs_value'].mean().reset_index()


employment_data = employment_data.reset_index(drop=True).drop_duplicates(subset='time').set_index('time')  # 确保时间索引唯一

# Step 3: Test stationarity of time series (ADF Test)
adf_test = adfuller(employment_data['obs_value'])
print(f"ADF Test Statistic: {adf_test[0]}, p-value: {adf_test[1]}")

if adf_test[1] > 0.05:
    print("Time series is not stationary. Differencing the data...")
    employment_data['obs_value_diff'] = employment_data['obs_value'].diff().dropna()
else:
    employment_data['obs_value_diff'] = employment_data['obs_value']






# Step 4: 数据准备
prophet_data = employment_data.reset_index()[['time', 'obs_value']]  # 将索引列转换为普通列
prophet_data.columns = ['ds', 'y']  # Prophet 需要列名为 'ds' 和 'y'

# Step 5: 初始化 Prophet 模型
prophet_model = Prophet(
    yearly_seasonality=True,  # 包含年度季节性
    weekly_seasonality=False,  # 不包含周季节性（按需调整）
    daily_seasonality=False,  # 不包含日季节性
    changepoint_prior_scale=0.1  # 控制变化点的灵敏度
)

# Step 6: 拟合模型
prophet_model.fit(prophet_data)

# Step 7: 创建未来的时间数据
future = prophet_model.make_future_dataframe(periods=24, freq='M')  # 预测未来 24 个月
forecast = prophet_model.predict(future)

# Step 8: 可视化预测结果
fig = prophet_model.plot(forecast)
fig.set_size_inches(12, 10)  # 设置图形大小
plt.title("Prophet Model - Unemployment Forecast")
plt.xlabel("Date")
plt.ylabel("Unemployment")
plt.savefig("/ossfs/workspace/proj/work_llm/result_unemployment/Prophet_Unemployment_Forecast.png")
plt.show()

# Step 10: 组件分析（趋势、季节性等）
fig_components = prophet_model.plot_components(forecast)
plt.savefig("/ossfs/workspace/proj/work_llm/result_unemployment/Prophet_Unemployment_Components.png")
plt.show()



# 以上是prophet分析
###########################################
# 以下是回归因果分析



import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.vector_ar.var_model import VAR
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Step 1: Load data
# 加载大模型分数数据
model_scores = pd.read_csv('/ossfs/workspace/proj/work_llm/llm_data/total_month.csv')
model_scores['time'] = pd.to_datetime(model_scores['time'], format='%Y-%m')

# 加载就业数据
employment_data = pd.read_csv(employ_data_path)
employment_data['time'] = pd.to_datetime(employment_data['time'], format='%YM%m')
employment_data = employment_data[employment_data['classif1.label'].str.contains('Total', na=False)]
employment_data = employment_data[employment_data['sex.label'].str.contains('Total', na=False)]  # 筛选包含 "Total" 的行


employment_data = employment_data.groupby('time')['obs_value'].mean().reset_index()

employment_data = employment_data.drop_duplicates(subset='time').set_index('time')

# 按时间从早到晚排序
employment_data = employment_data.sort_index()


# Step 2: Merge datasets
# 将大模型能力分数与就业数据合并
merged_data = pd.merge(
    employment_data.reset_index(), 
    model_scores.rename(columns={'time': 'Date', 'score': 'Score'}), 
    left_on='time', 
    right_on='Date', 
    how='left'
)
merged_data['Score'] = merged_data['Score'].fillna(0)  # 模型发布前分数为 0
merged_data['Post_Model'] = (merged_data['Date'] >= '2023-09-01').astype(int)  # 模型发布后标志
merged_data.set_index('time', inplace=True)



# Step 3: Create interaction term for DID analysis
merged_data['Interaction'] = merged_data['Post_Model'] * merged_data['Score']




# 多变量回归分析（DID）
X = sm.add_constant(merged_data[['Post_Model', 'Score', 'Interaction']])
y = merged_data['obs_value']
did_model = sm.OLS(y, X).fit()
print(did_model.summary())

# 可视化回归结果
params = did_model.params.copy()
params_scaled = params.copy()
params_scaled[['Score', 'Interaction']] *= 100  # 放大 100 倍

plt.figure(figsize=(8, 6))
plt.bar(params_scaled.index, params_scaled.values, color=['blue', 'orange', 'green', 'red'], alpha=0.7)
plt.title('Impact of LLM Scores on Unemployment (DID Regression)')
plt.ylabel('Coefficient Value (Scaled)')
plt.tight_layout()
plt.savefig("/ossfs/workspace/proj/work_llm/result_unemployment/DID_Impact_LLM_Scores.png")
plt.show()



# Step 4: 工具变量回归 (Instrumental Variables - IV)

# 模拟工具变量
merged_data['IV'] = np.random.uniform(0.5, 1.5, size=len(merged_data)) * merged_data['Score']

# 第一阶段：工具变量对Score的回归
X_iv = sm.add_constant(merged_data[['Post_Model', 'IV']])
y_iv = merged_data['Score']
first_stage = sm.OLS(y_iv, X_iv).fit()
merged_data['Score_hat'] = first_stage.fittedvalues

# 第二阶段：使用预测的Score_hat进行回归
X_iv2 = sm.add_constant(merged_data[['Post_Model', 'Score_hat', 'Interaction']])
y_iv2 = merged_data['obs_value']
iv_model = sm.OLS(y_iv2, X_iv2).fit()

# 输出结果
print(iv_model.summary())

# 可视化IV回归结果
params_iv = iv_model.params.copy()
params_iv_scaled = params_iv.copy()
params_iv_scaled[['Score_hat', 'Interaction']] *= 100

plt.figure(figsize=(8, 6))
plt.bar(params_iv_scaled.index, params_iv_scaled.values, color=['blue', 'orange', 'green', 'red'], alpha=0.7)
plt.title('Impact of LLM Scores on Unemployment (IV Regression)')
plt.ylabel('Coefficient Value (Scaled)')
plt.tight_layout()
plt.savefig("/ossfs/workspace/proj/work_llm/result_unemployment/IV_Impact_LLM_Scores.png")
plt.show()






# Step 5: VAR Model for dynamic analysis
var_data = merged_data[['obs_value', 'Score']].dropna()
var_model = VAR(var_data)
var_results = var_model.fit(maxlags=5)
print(var_results.summary())

# 绘制脉冲响应函数（IRF）
irf = var_results.irf(10)
fig = irf.plot(orth=False)
plt.title('Dynamic Impact of LLM Scores on Unemploy with Lags 5')
plt.savefig("/ossfs/workspace/proj/work_llm/result_unemployment/Dynamic_Impact_LLM_Scores_5.png")
plt.show()




# 增加滞后阶数
var_lag_data = merged_data[['obs_value', 'Score']].dropna()

# 构建VAR模型
var_model_lag = VAR(var_lag_data)
var_results_lag = var_model_lag.fit(maxlags=10)  # 可增加滞后阶数
print(var_results_lag.summary())

# 绘制滞后脉冲响应函数
irf_lag = var_results_lag.irf(15)  # 分析15期的影响
fig_lag = irf_lag.plot(orth=False)
plt.title('Dynamic Impact of LLM Scores on Unemploy with Lags 10')
plt.savefig("/ossfs/workspace/proj/work_llm/result_unemployment/Dynamic_Impact_LLM_Scores_10.png")
plt.show()






# Step 6: Visualization of trends

plt.figure(figsize=(12, 6))

# 绘制就业数据和模型分数
plt.plot(merged_data.index, merged_data['obs_value'], label='Observed Unemployment')
plt.plot(merged_data.index, merged_data['Score'], label='LLM Scores (Scaled)', linestyle='--')

# 将'2023-09-01'转换为datetime格式
model_release_date = datetime.strptime('2023-09-01', '%Y-%m-%d')

# 添加垂直线
plt.axvline(x=model_release_date, color='red', linestyle='--', label='Model Releases Start')

# 图表标题和标签
plt.title('Observed Unemployment and LLM Scores Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment')
plt.legend()

# 保存图表
plt.tight_layout()
plt.savefig("/ossfs/workspace/proj/work_llm/result_unemployment/Observed_Unemployment_and_LLM_Scores.png")
plt.show()


# Step 7: 情景模拟：未来模型分数持续增长
future_scores = pd.DataFrame({
    'Date': pd.date_range(start='2024-11-01', periods=60, freq='M'),  # 预测未来 60 个月
    'Score': np.linspace(95, 150, 60)  # 假设模型分数线性增长到 150
})

# 合并未来数据
future_data = pd.concat([merged_data.reset_index(), future_scores], ignore_index=True, sort=False)
future_data['Post_Model'] = (future_data['Date'] >= '2023-09-01').astype(int)
future_data['Interaction'] = future_data['Post_Model'] * future_data['Score']

# 使用已有回归模型预测未来就业
X_future = sm.add_constant(future_data[['Post_Model', 'Score', 'Interaction']])
future_data['Predicted_Unemployment'] = did_model.predict(X_future)

# 可视化更长时间的预测结果
plt.figure(figsize=(12, 6))
plt.plot(future_data['Date'], future_data['Predicted_Unemployment'], label='Predicted Unemployment (Long-term)', linestyle='--', color='orange')
plt.plot(merged_data.index, merged_data['obs_value'], label='Observed Unemployment', color='blue')
plt.title("Long-term Unemployment Prediction with LLM Score Growth")
plt.xlabel("Date")
plt.ylabel("Unemployment")
plt.legend()
plt.tight_layout()
plt.savefig("/ossfs/workspace/proj/work_llm/result_unemployment/Long_Term_Unemployment_Prediction.png")
plt.show()




