---
layout: post
title: 推荐系统简介
categories: [RS]
---

在**信息过载**的背景下，当用户**无明确需求**时，推荐系统利用**相关数据****主动**将**相关信息**推送给用户。
  
## 推荐系统介绍 
### what
在推荐系统之前，有像Hao12/Yahoo等的分类目录的门户网站，其覆盖少量热门网站。有像Google/Baidu等的搜索引擎，其通过搜索关键词获取相关信息。而推荐系统不需要用户提供明确的需求，其通过分析用户的历史行为主动给用户推送能满足其兴趣和需求的信息。

搜索场景与推荐场景在不同维度的对比如下：

|场景|搜素|推荐|
|:--|:--:|:--:|
|行为方式|主动|被动|
|意图|明确|模糊|
|个性化|弱|强|
|流量分布|马太效应|长尾效应|
|目标|快速满足|持续服务|
|评估指标|简明|复杂|

- 马太效应：热点信息曝光的机会更大，可理解为富者愈富，贫者愈贫
- 长尾效应：挖掘长尾分布(不流行的物品)，让热点信息和小众信息尽可能得到相同的曝光机会(流量)

搜索和推荐也在不断地相结合来产生新的产品形态。

### why
存在前提：
- 信息过载
- 用户需求不明确

目标
- 连接用户和物品，发现长尾物品
- 留住用户(消费者)和内容提供方(生产者)，实现商业目标

### where
头条、快手、淘宝等

## 推荐系统评估
### 指标
- **准确性**
- 满意度
- **覆盖率(长尾分布)**
- **多样性**
- **新颖性**
- **惊喜度**
- 信任度
- 实时性
- 鲁棒性 
- 可扩展性
- 商业目标
- 用户留存

trade off
- exploitation(开发/利用)：选择**现在**可能的最佳方案 
- exploration(探索)：选择现在不确定的一些方案，但**未来**可能会有高收益的方案，防止用户兴趣收敛、单一

**探索(EE)** 实践
- 兴趣扩展：相似话题、搭配推荐
- 人群算法：userCF、用户聚类
- bandit算法
- graph walking
- 平衡个性化推荐和热门推荐的比例
- 随机丢弃用户行为历史
- 随机扰动模型参数

### 方法
- 问卷调查：成本高
- 离线评估：
    - 只能在用户看到过的候选集上做评估，且跟线上真实效果**存在偏差**
    - 只能评估少数指标
    - 速度快，不损害用户体验
- 在线评估：A/B testing(单层实验/多层重叠)
 
实践：离线评估和在线评估相结合，定期做问卷调查

## 推荐系统实战

### 冷启动问题
- bandit

### 工业界推荐系统架构

- Netflix : offline/nearline/online

- Taobao : architecture(Hbase/Tair/ODPS) ---> match(RT/CF/DT/content-based/user-based) ---> rank(RTP/Olive) ---> re-rank(business goal/diversity/freshness) ---> application

- YouTube : corpus --millions--> candidate --hundreds--> ranking --dozens--> result to show

### 推荐系统发展阶段
- 1.0：关联规则、热门推荐等统计方法
- 2.0：矩阵分解(MF)、协同过滤(CF)等ML方法，离线计算推荐列表
- 3.0：**召回 + learning to rank 重排序** (offline + nearline)
- 4.0：召回、排序实时化 (online)
- 5.0：端到端的深度学习，一切皆embedding
- 6.0：智能化推荐系统(强化学习/图谱/多任务)

趋势

|之前|未来|
|:--:|:--:|
|单一模块|多模块解耦|
|单一目标|多目标|
|单个场景|多场景(平台化)|
|离线计算|实时计算|
|人工规则|人工智能|
|浅度模型|深度模型|


### 学术 vs 工业

||学术|工业|
|:--:|:--:|:--:|
|数据量|百万|百亿|
|数据分布|稳定|变化|
|研究问题|定义清晰|复杂，不可形式化|
|关注点|精度的极致|性价比|
|评估指标|单一|多个|
|评估方法|离线|在线+问卷调查|

## 结构与资料
### 结构
候选--million-->召回(模型/特征/推荐)--thousand-->预估(模型/特征/推荐)--hundred-->排序--ten-->展示---->日志---->实时/离线学习---->模型/特征/推荐

### 资料

- book
    - 项亮:推荐系统实战
    - recommender systems handbook

- paper
    - Item-Based Collaborative Filtering Recommendation Algorithms
    - **Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model**
    - Matrix factorization techniques for recommender systems
    
- link
    - Facebook实践： recommending items to more than a billion people
    - Quora是如何做推荐的?
    - Real-time Personalization using Embeddings for Search Ranking at Airbnb
    - Deep Neural Networks for YouTube Recommendations 
    - Wide & Deep learning for Recommender Systems
    - Ad Click Prediction: a View from the Trenches
    




 
