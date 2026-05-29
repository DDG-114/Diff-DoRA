# Diff-DoRA 数据集清单与分类

本文档整理当前项目目录中的主要数据集、缓存和预测结果，方便后续写报告、设计实验和避免混淆不同数据源。

## 1. 已接入 Diff-DoRA 的 EV 充电需求数据集

这类数据已经符合当前项目的核心接口：

```text
occupancy:   time x node
timestamps:  time index
node_meta:   node metadata
adj:         node x node adjacency
weather:     optional exogenous features
price:       optional price features
```

### 1.1 ST-EVCDP

定位：原始论文/复现实验中的主要 EV 充电需求数据集。

路径：

```text
data/raw/st_evcdp/
data/processed/st_evcdp.pkl
data/processed/st_evcdp_trainnorm_h6.pkl
```

规模：

```text
occupancy shape: (8640, 247)
时间范围: 2022-06-19 00:00:00 到 2022-07-18 23:55:00
节点数: 247
adj shape: (247, 247)
weather shape: (8640, 5)
price shape: (8640, 247)
```

用途：

```text
主复现实验
MoE 专家训练
RAG 检索缓存构建
few-shot / zero-shot 协议
论文对齐实验
```

相关缓存：

```text
data/retrieval_cache/st_evcdp_h6.pkl
data/retrieval_cache/st_evcdp_h6_step6.pkl
data/sample_cache/train_experts_st_evcdp_h6_hist12_nbr7.pkl
data/manifests/st_evcdp/
```

说明：

ST-EVCDP 是真实多充电节点数据，`CBD / Residential` 路由标签来自站点元数据或占用率代理划分，适合解释为空间节点上的充电需求预测。

### 1.2 UrbanEV

定位：另一个已接入的 EV 充电需求基准数据集。

路径：

```text
data/raw/urbanev/
data/processed/urbanev.pkl
```

规模：

```text
occupancy shape: (4344, 275)
时间范围: 2022-09-01 00:00:00 到 2023-02-28 23:00:00
节点数: 275
adj shape: (275, 275)
weather shape: (4344, 18)
price shape: (4344, 275)
```

用途：

```text
跨数据集泛化实验
EV 充电需求预测对比
UrbanEV 相关评估脚本
```

说明：

UrbanEV 同样是多站点 EV 充电需求数据，节点语义接近 ST-EVCDP，适合作为第二个 EV 数据集验证模型泛化能力。

## 2. 沃太 EV / 综合能源数据集

### 2.1 原始沃太数据

定位：用户提供的新数据源，来自沃太项目，包含负荷、光伏、储能、电表、天气和已有 PV/Load 子任务结果。

原始压缩包：

```text
39e50eff-5119-4a3f-9833-405697551abe.zip
```

解压路径：

```text
data/raw/wotai_source/
```

主要内容：

```text
data/raw/wotai_source/Raw Data/
data/raw/wotai_source/PV Prediction/
data/raw/wotai_source/Load Prediction/
```

代表文件：

```text
Load Prediction/Preprocessed_Load_Prediction.csv
PV Prediction/Preprocessed_PV_Prediction.csv
Raw Data/光伏运行数据.csv
Raw Data/储能20240701电表倍率4000.csv
Raw Data/关口表20240701.csv
Raw Data/weather_history_202508041807.csv
```

说明：

沃太原始数据不是 ST-EVCDP 那种多充电站空间节点数据，而是单项目综合能源系统数据。它更适合按功率/能流通道构造功能节点。

### 2.2 已适配的 Wotai-EVCDP 格式

定位：为了接入当前 Diff-DoRA 框架，将沃太数据整理成 `time x node` 结构后的版本。

路径：

```text
data/raw/wotai_evcdp/
data/processed/wotai_evcdp.pkl
```

生成脚本：

```text
scripts/prepare_wotai_evcdp.py
src/data/load_wotai_evcdp.py
```

规模：

```text
occupancy shape: (38208, 4)
时间范围: 2024-07-02 00:00:00 到 2025-08-03 23:45:00
节点数: 4
adj shape: (4, 4)
weather shape: (38208, 15)
price shape: (0, 0)
```

功能节点：

```text
actual_load        实际负荷/需求，主预测目标
pv_total_power     光伏总功率
storage_ac_power   储能交流功率
grid_active_power  关口有功功率
```

相关缓存：

```text
data/retrieval_cache/wotai_evcdp_h6_step6.pkl
data/sample_cache/train_experts_wotai_evcdp_h6_hist12_nbr7_step6.pkl
```

相关预测输出：

```text
outputs/wotai_evcdp_existing_weights_smoke.json
outputs/wotai_scenario_forecast_overnight/
outputs/wotai_rolling_forecast_20260505T165645Z/
```

说明：

这里的节点是功能节点，不是地理站点节点。当前 `CBD / Residential` 标签主要是为了复用双专家 MoE 路由，不应在报告中解释为真实城市区域划分。

## 3. 新增可再生能源发电特征分析数据集

定位：独立放入项目的新数据集，主题是 renewable energy generation input feature variables analysis。它不是 EV 充电需求数据，而是可再生能源发电预测/特征分析数据。

路径：

```text
data/raw/renewable_generation/source/
```

目录结构：

```text
data/raw/renewable_generation/source/data_original/
data/raw/renewable_generation/source/data_processed/
data/raw/renewable_generation/source/variables_correlation_analysis.ipynb
data/raw/renewable_generation/source/variables_correlation_analysis_processe_data.ipynb
```

### 3.1 太阳能电站数据

路径：

```text
data/raw/renewable_generation/source/data_original/solar_stations/
data/raw/renewable_generation/source/data_processed/solar_stations/
```

规模：

```text
原始太阳能站点 Excel: 8 个
处理后太阳能站点 Excel: 8 个
多数站点约 70177 行
```

典型字段：

```text
Time(year-month-day h:m:s)
Total solar irradiance (W/m2)
Direct normal irradiance (W/m2)
Global horizontal irradiance (W/m2)
Air temperature (°C)
Atmosphere (hpa)
Relative humidity (%)
Power (MW)
```

容量示例：

```text
Solar station site 1: 50MW
Solar station site 2: 130MW
Solar station site 3: 30MW
Solar station site 4: 130MW
Solar station site 5: 110MW
Solar station site 6: 35MW
Solar station site 7: 30MW
Solar station site 8: 30MW
```

### 3.2 风电场数据

路径：

```text
data/raw/renewable_generation/source/data_original/wind_farms/
data/raw/renewable_generation/source/data_processed/wind_farms/
```

规模：

```text
原始风电场 Excel: 6 个
处理后风电场 Excel: 6 个
多数站点约 70177 行
```

典型字段：

```text
Time(year-month-day h:m:s)
Wind speed at height of 10 meters (m/s)
Wind direction at height of 10 meters
Wind speed at height of 30 meters (m/s)
Wind direction at height of 30 meters
Wind speed at height of 50 meters (m/s)
Wind direction at height of 50 meters
Wind speed at the height of wheel hub
Air temperature (°C)
Atmosphere (hpa)
Relative humidity (%)
Power (MW)
```

容量示例：

```text
Wind farm site 1: 99MW
Wind farm site 2: 200MW
Wind farm site 3: 99MW
Wind farm site 4: 66MW
Wind farm site 5: 36MW
Wind farm site 6: 96MW
```

### 3.3 已适配的发电预测数据集

该数据集已经被拆分并接入为两个独立任务：

```text
renewable_solar
renewable_wind
```

对应脚本与 loader：

```text
scripts/prepare_renewable_generation.py
src/data/load_renewable_generation.py
```

#### renewable_solar

路径：

```text
data/raw/renewable_solar/
data/processed/renewable_solar.pkl
```

规模：

```text
occupancy shape: (70175, 3)
weather shape: (70175, 6)
adj shape: (3, 3)
```

保留站点：

```text
solar_site_01
solar_site_02
solar_site_04
```

剔除站点：

```text
solar_site_03
solar_site_05
solar_site_06
solar_site_07
solar_site_08
```

典型外生变量：

```text
total_solar_irradiance
direct_normal_irradiance
global_horizontal_irradiance
temperature
pressure
humidity
```

#### renewable_wind

路径：

```text
data/raw/renewable_wind/
data/processed/renewable_wind.pkl
```

规模：

```text
occupancy shape: (70176, 2)
weather shape: (70176, 11)
adj shape: (2, 2)
```

保留站点：

```text
wind_site_01
wind_site_02
```

剔除站点：

```text
wind_site_03
wind_site_04
wind_site_05
wind_site_06
```

典型外生变量：

```text
wind_speed_10m
wind_direction_10m
wind_speed_30m
wind_direction_30m
wind_speed_50m
wind_direction_50m
wind_speed_hub
wind_direction_hub
temperature
pressure
humidity
```

说明：

这里采用的是保守适配策略：只保留时间覆盖充分、可对齐到共同时间轴的站点。这样可以立即进入 Diff-DoRA 标准流程，但站点数会少于原始 Excel 总数。

注意：

该数据集更适合作为“可再生能源发电预测”任务，不建议直接归入“EV 充电需求预测”类别。若用于 Diff-DoRA 扩展，应在报告中说明任务已从需求侧预测扩展到发电侧预测。

## 4. 缓存与实验结果，不属于原始数据集

这些文件是训练或评估过程中生成的中间结果，不应和原始数据混在一起解释。

### 4.1 Retrieval cache

路径：

```text
data/retrieval_cache/
```

用途：

```text
保存 KNN-RAG 的历史窗口向量和样本池
```

示例：

```text
data/retrieval_cache/st_evcdp_h6_step6.pkl
data/retrieval_cache/wotai_evcdp_h6_step6.pkl
```

### 4.2 Sample cache

路径：

```text
data/sample_cache/
```

用途：

```text
保存滑动窗口样本，避免重复构造训练样本
```

示例：

```text
data/sample_cache/train_experts_st_evcdp_h6_hist12_nbr7.pkl
data/sample_cache/train_experts_wotai_evcdp_h6_hist12_nbr7_step6.pkl
```

### 4.3 输出结果

路径：

```text
outputs/
```

用途：

```text
模型权重
metrics.json
预测结果 JSON
滚动预测结果
zero-shot / few-shot 实验结果
```

示例：

```text
outputs/full_repro_st_evcdp_h6_bs48/
outputs/zeroshot_moe_st_evcdp_h6_20260426T211800Z/
outputs/wotai_rolling_forecast_20260505T165645Z/
```

## 5. 推荐分类方式

后续报告和实验记录中建议按以下类别描述：

```text
EV charging demand benchmark:
  ST-EVCDP
  UrbanEV

Wotai integrated energy / EV demand adaptation:
  wotai_source
  wotai_evcdp

Renewable energy generation:
  data/raw/renewable_generation/source
  renewable_solar
  renewable_wind
  solar stations
  wind farms

Generated caches:
  retrieval_cache
  sample_cache
  manifests

Experiment outputs:
  outputs/full_repro_*
  outputs/zeroshot_*
  outputs/wotai_*
```

## 6. 当前可直接用于项目脚本的数据集

已经接入 `src.data.loaders.DATASET_LOADERS` 的数据集：

```text
st_evcdp
urbanev
wotai_evcdp
renewable_solar
renewable_wind
```

尚未接入、需要后续适配的数据集：

```text
data/raw/renewable_generation/source
```
