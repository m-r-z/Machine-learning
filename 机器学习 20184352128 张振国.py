#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pandas_profiling as ppf

from sklearn.cluster import MiniBatchKMeans,KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

import seaborn as sns
sns.set()
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置
from mpl_toolkits.mplot3d import Axes3D


# In[30]:


data = pd.read_csv('/home/aistudio/data/data96250/dataset.csv')
submit = pd.read_csv('/home/aistudio/data/data96250/submission.csv')

data.columns = ['风机编号','时间戳','风速','功率','风轮转速']
# data['label'] = 0 # 1为异常数据
print('Data shape:{}'.format(data.shape))
print('Submit shape:{}'.format(submit.shape))
print('训练集与测试集主键是否相同:',(data['时间戳'] == submit['Time']).all())
data.head()


# In[31]:


def speed(v):
    return 0 if v>7 else 1
data['风轮转速_01'] = data['风轮转速'].apply(speed)


# In[32]:


data1 = data.loc[data['风机编号']==1]
data2 = data.loc[data['风机编号']==2]
data3 = data.loc[data['风机编号']==3]
data4 = data.loc[data['风机编号']==4]
data5 = data.loc[data['风机编号']==5]
data6 = data.loc[data['风机编号']==6]
data7 = data.loc[data['风机编号']==7]
data8 = data.loc[data['风机编号']==8]
data9 = data.loc[data['风机编号']==9]
data10 = data.loc[data['风机编号']==10]
data11 = data.loc[data['风机编号']==11]
data12 = data.loc[data['风机编号']==12]


# In[33]:


# 官方给的每个风机的风轮直径、额定功率、切入风速、切出风速、风轮转速范围等
def wheel_diameter(id_):
    id_ = int(id_)
    if id_==5:return 100.5
    elif id_==11:return 115
    elif id_==12:return 104.8
    else:return 99 
def rated_p(id_):
    return 2000
def cutin_wind(id_):
    id_ = int(id_)
    if id_ == 11:return 2.5
    else:        return 3 
def cutout_wind(id_):
    id_ = int(id_)
    if id_==5 or id_==12:return 22
    elif id_==11:        return 19
    else:                return 25
def wheel_speed(id_):
    id_ = int(id_)
    if id_ == 5:return [5.5,19]
    elif id_ == 11:return [5,14]
    elif id_ == 12:return [5.5,17]
    return [8.33,16.8]
    
    
data['风轮直径'] = data['风机编号'].apply(wheel_diameter)
data['额定功率'] = data['风机编号'].apply(rated_p)
data['切入风速'] = data['风机编号'].apply(cutin_wind)
data['切出风速'] = data['风机编号'].apply(cutout_wind)
data['风轮转速范围'] = data['风机编号'].apply(wheel_speed)
data.head()


# In[34]:


data['时间戳'] = pd.to_datetime(data['时间戳'])
data = data.sort_values('时间戳')
data.head()


# In[35]:


sns.set()
print(data['风速'].loc[data['风机编号']==1].describe())
data['风速'].loc[data['风机编号']==1].hist()


# In[36]:


mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
data.loc[data['风机编号']==1].plot(x='时间戳',y='风速',figsize=(12,6))


# In[37]:


mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
data.loc[data['风机编号']==12].plot(x='时间戳',y='风速',figsize=(12,6))


# In[38]:


a1 = data.loc[data['风机编号']==1,'风速']
a2 = data.loc[data['风机编号']==2,'风速']
a3 = data.loc[data['风机编号']==3,'风速']
a4 = data.loc[data['风机编号']==4,'风速']
a5 = data.loc[data['风机编号']==5,'风速']
a6 = data.loc[data['风机编号']==6,'风速']
a7 = data.loc[data['风机编号']==7,'风速']
a8 = data.loc[data['风机编号']==8,'风速']
a9 = data.loc[data['风机编号']==9,'风速']
a10 = data.loc[data['风机编号']==10,'风速']
a11 = data.loc[data['风机编号']==11,'风速']
a12 = data.loc[data['风机编号']==12,'风速']
plt.figure(figsize=(15,10))
plt.hist(a1,bins=50 , alpha=0.5,label='风机编号1')
plt.hist(a2,bins=50 , alpha=0.5,label='风机编号2')
plt.hist(a3,bins=50 , alpha=0.5,label='风机编号3')
plt.hist(a4,bins=50 , alpha=0.5,label='风机编号4')
plt.hist(a5,bins=50 , alpha=0.5,label='风机编号5')
plt.hist(a6,bins=50 , alpha=0.5,label='风机编号6')
plt.hist(a7,bins=50 , alpha=0.5,label='风机编号7')
plt.hist(a8,bins=50 , alpha=0.5,label='风机编号8')
plt.hist(a9,bins=50 , alpha=0.5,label='风机编号9')
plt.hist(a10,bins=50 , alpha=0.5,label='风机编号10')
plt.hist(a11,bins=50 , alpha=0.5,label='风机编号11')
plt.hist(a12,bins=50 , alpha=0.5,label='风机编号12')
plt.legend(loc='upper right')
plt.xlabel('风速')
plt.ylabel('统计')
plt.show()


# In[39]:


data_num = data[['风速','功率','风轮转速']]
n_cluster = range(1, 20)
kmeans = [KMeans(n_clusters=i).fit(data_num) for i in n_cluster]
scores = [kmeans[i].score(data_num) for i in range(len(kmeans))]

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(n_cluster, scores)
plt.xlabel('N_cluster')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()


# In[40]:


X = data_num
X = X.reset_index(drop=True)
km = KMeans(n_clusters=10)
km.fit(X)
km.predict(X)
labels = km.labels_
 
fig = plt.figure(1, figsize=(15,15))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(X.iloc[:,0], X.iloc[:,1], X.iloc[:,2],
          c=labels.astype(np.float), edgecolor="k")
# '风速','功率','风轮转速'
ax.set_xlabel("风速")
ax.set_ylabel("功率")
ax.set_zlabel("风轮转速")
plt.title("K Means", fontsize=14)


# In[41]:


X = data[['风速','功率','风轮转速']].loc[data['风机编号']==1]
X = X.reset_index(drop=True)
km = KMeans(n_clusters=2)
km.fit(X)
km.predict(X)
labels = km.labels_
 
fig = plt.figure(1, figsize=(10,10))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(X.iloc[:,0], X.iloc[:,1], X.iloc[:,2],
          c=labels.astype(np.float), edgecolor="k")
# '风速','功率','风轮转速'
ax.set_xlabel("风速")
ax.set_ylabel("功率")
ax.set_zlabel("风轮转速")
plt.title("风机编号1 K Means", fontsize=14)


# In[42]:


X = data_num.values

# 标准化 均值为0 标准差为1
X_std    = StandardScaler().fit_transform(X)
mean_vec = np.mean(X_std,axis=0)
# 协方差,协方差矩阵反应了特征变量之间的相关性
# 如果两个特征变量之间的协方差为正则说明它们之间是正相关关系
# 如果为负则说明它们之间是负相关关系
cov_mat = np.cov(X_std.T)

# 特征值和特征向量
eig_vals,eig_vecs = np.linalg.eig(cov_mat)
# 特征值对应的特征向量
eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort(key = lambda x: x[0], reverse= True)
# 特征之求和
eig_sum = sum(eig_vals)


# 解释方差
var_exp = [(i/eig_sum)*100 for i in sorted(eig_vals, reverse=True)]
# 累计的解释方差
cum_var_exp = np.cumsum(var_exp)

plt.figure(figsize=(10,5))
plt.bar(range(len(var_exp)), var_exp, alpha=0.3, align='center', label='独立的解释方差', color = 'g')
plt.step(range(len(cum_var_exp)), cum_var_exp, where='mid',label='累积解释方差')
plt.ylabel('解释方差率')
plt.xlabel('主成分')
plt.legend(loc='best')
plt.show()


# In[43]:


# 标准化处理,均值为0,标准差为1
X_std = StandardScaler().fit_transform(data_num.values)
data_std = pd.DataFrame(X_std)
 
#将特征维度降到1
pca = PCA(n_components=1)
data_std = pca.fit_transform(data_std)
# 降维后将1个新特征进行标准化处理
scaler = StandardScaler()
np_scaled = scaler.fit_transform(data_std)
data_std = pd.DataFrame(np_scaled)
 
kmeans = [KMeans(n_clusters=i).fit(data_std) for i in n_cluster]
data['cluster'] = kmeans[9].predict(data_std) # 刚才Elbow曲线10类基本收敛了，故还是选择10类
data.index = data_std.index
data['principal_feature1'] = data_std[0]
data.head()


# In[44]:


# 计算每个数据点到其聚类中心的距离
def getDistanceByPoint(data, model):
    distance = pd.Series(dtype='float64')
    for i in range(0,len(data)):
        Xa = np.array(data.loc[i])
        Xb = model.cluster_centers_[model.labels_[i]]
        distance.at[i] = np.linalg.norm(Xa-Xb)
    return distance
 
#设置异常值比3566例
outliers_fraction = 0.4
 
# 得到每个点到取聚类中心的距离，我们设置了10个聚类中心，kmeans[9]表示有10个聚类中心的模型
distance = getDistanceByPoint(data_std, kmeans[9])
 
#根据异常值比例outliers_fraction计算异常值的数量
number_of_outliers = int(outliers_fraction*len(distance))
 
#设定异常值的阈值
threshold = distance.nlargest(number_of_outliers).min()
 
#根据阈值来判断是否为异常值
data['anomaly1'] = (distance >= threshold).astype(int)


# In[45]:


#数据可视化
fig, ax = plt.subplots(figsize=(10,6))
colors = {0:'blue', 1:'red'}
ax.scatter(data['principal_feature1'],data['风速'],c=data["anomaly1"].apply(lambda x: colors[x]))
plt.xlabel('principal feature1')
plt.ylabel('风速')
plt.show()


# In[46]:


data['anomaly1'].value_counts()


# In[47]:


fig, ax = plt.subplots(figsize=(12,6))
 
a = data.loc[data['anomaly1'] == 1, ['时间戳', '风轮转速']] #anomaly
 
ax.plot(data['时间戳'], data['风轮转速'], color='blue', label='正常值')
ax.scatter(a['时间戳'],a['风轮转速'], color='red', label='异常值')
plt.xlabel('时间戳')
plt.ylabel('风轮转速')
plt.legend()
plt.show()


# In[48]:


# 训练孤立森林模型
model =  IsolationForest(contamination=outliers_fraction)
model.fit(data_std)
 
#返回1表示正常值，-1表示异常值
data['anomaly2'] = pd.Series(model.predict(data_std)) 
 
fig, ax = plt.subplots(figsize=(10,6))
a = data.loc[data['anomaly2'] == -1, ['时间戳', '风轮转速']] #异常值
ax.plot(data['时间戳'], data['风轮转速'], color='blue', label = '正常值')
ax.scatter(a['时间戳'],a['风轮转速'], color='red', label = '异常值')
plt.legend()
plt.show()


# In[49]:


# 训练 oneclassSVM 模型

print(0)

model = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
model.fit(data_std)
 
print(1)

data['anomaly3'] = pd.Series(model.predict(data_std))
fig, ax = plt.subplots(figsize=(10,6))

print(2)

a = data.loc[data['anomaly3'] == -1, ['时间戳', '风轮转速']] #异常值

print(3)

ax.plot(data['时间戳'], data['风轮转速'], color='blue', label = '正常值')
ax.scatter(a['时间戳'],a['风轮转速'], color='red', label = '异常值')
plt.legend()
plt.show()


# In[50]:


print(1)

data['anomaly3'] = pd.Series(model.predict(data_std))
fig, ax = plt.subplots(figsize=(10,6))



# In[51]:


print(2)

a = data.loc[data['anomaly3'] == -1, ['时间戳', '风轮转速']] #异常值



# In[52]:


print(3)

ax.plot(data['时间戳'], data['风轮转速'], color='blue', label = '正常值')
ax.scatter(a['时间戳'],a['风轮转速'], color='red', label = '异常值')
plt.legend()
plt.show()


# In[53]:


# 基于高斯概分布的异常检测
df_class0 = data.loc[data['风轮转速'] >7, '风轮转速']
df_class1 = data.loc[data['风轮转速'] <=7, '风轮转速']
 
envelope =  EllipticEnvelope(contamination = outliers_fraction) 
X_train = df_class0.values.reshape(-1,1)
envelope.fit(X_train)
df_class0 = pd.DataFrame(df_class0)
df_class0['deviation'] = envelope.decision_function(X_train)
df_class0['anomaly'] = envelope.predict(X_train)
 
envelope =  EllipticEnvelope(contamination = outliers_fraction) 
X_train = df_class1.values.reshape(-1,1)
envelope.fit(X_train)
df_class1 = pd.DataFrame(df_class1)
df_class1['deviation'] = envelope.decision_function(X_train)
df_class1['anomaly'] = envelope.predict(X_train)
 
df_class = pd.concat([df_class0, df_class1])

data['anomaly4'] = df_class['anomaly']
fig, ax = plt.subplots(figsize=(10, 6))
a = data.loc[data['anomaly4'] == -1, ('时间戳', '风轮转速')] 
ax.plot(data['时间戳'], data['风轮转速'], color='blue', label = '正常值')
ax.scatter(a['时间戳'],a['风轮转速'], color='red', label = '异常值')
plt.show()


# In[54]:


# anomaly1 == 1 异常值
# anomaly2 == -1 异常值
# anomaly3 == -1 异常值
# anomaly4 == -1 异常值
data.head()


# In[55]:


def anomaly_process(df):
    df = df.copy()
    df.loc[:, 'anomaly1'] = df.loc[:, 'anomaly1'].map({ 0:0, 1:1})
    df.loc[:, 'anomaly2'] = df.loc[:, 'anomaly2'].map({ 1:0, -1:1})
    df.loc[:, 'anomaly3'] = df.loc[:, 'anomaly3'].map({ 1:0, -1:1})
    df.loc[:, 'anomaly4'] = df.loc[:, 'anomaly4'].map({ 1:0, -1:1})
    return df
data = anomaly_process(data)
data.head()


# In[56]:


data['sum'] = data['风轮转速_01'] + data['anomaly1'] + data['anomaly2'] + data['anomaly3'] + data['anomaly4']
submit.loc[:,'label'] = data.loc[:,'sum'].map({0:0,1:1,2:1,3:1,4:1,5:1})


# In[62]:


submit.to_csv('baseline.csv',index=False)
submit.head()

