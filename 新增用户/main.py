# %% [markdown]
# **赛题名称**：用户新增预测挑战赛
# **赛题类型**：数据挖掘、二分类
# **赛题链接**👇：
# https://challenge.xfyun.cn/topic/info?type=subscriber-addition-prediction&ch=ymfk4uU
# 
# 其中uuid为样本唯一标识，eid为访问行为ID，udmap为行为属性，其中的key1到key9表示不同的行为属性，如项目名、项目id等相关字段，common_ts为应用访问记录发生时间（毫秒时间戳），其余字段x1至x8为用户相关的属性，为匿名处理字段。target字段为预测目标，即是否为新增用户。
# 
# ## 评价指标
# 本次竞赛的评价标准采用f1_score，分数越高，效果越好。
# 

# %%
import pandas as pd
import numpy as np

import matplotlib.pylab as plt
import seaborn as sns
# 导入模型
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import KNeighborsClassifier

# 导入交叉验证和评价指标
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
#%%

train_data = pd.read_csv('用户新增预测挑战赛公开数据/train.csv')
test_data = pd.read_csv('用户新增预测挑战赛公开数据/test.csv')

# 转换为毫秒
train_data['common_ts'] = pd.to_datetime(train_data['common_ts'], unit='ms')
test_data['common_ts'] = pd.to_datetime(test_data['common_ts'], unit='ms')

# %%
# 文本特征提取函数
def udmap_onethot(d):
    v = np.zeros(9)
    if d == 'unknown':
        return v
    d = eval(d)
    for i in range(1, 10):
        if 'key' + str(i) in d:
            v[i-1] = d['key' + str(i)]       
    return v
def gen_udmap(d):
    if d == 'unknown':
        return 'unknown'
    d = eval(d)
    v = ''
    for i in range(1, 10):
        if 'key' + str(i) in d:
            v = v + 'key' + str(i)
    return v
#%%
train_udmap_df = pd.DataFrame(np.vstack(train_data['udmap'].apply(udmap_onethot)))
test_udmap_df = pd.DataFrame(np.vstack(test_data['udmap'].apply(udmap_onethot)))
train_data['udmap_key'] = pd.DataFrame(np.vstack(train_data['udmap'].apply(gen_udmap)))
test_data['udmap_key'] = pd.DataFrame(np.vstack(test_data['udmap'].apply(gen_udmap)))

train_udmap_df.columns = ['key' + str(i) for i in range(1, 10)]
test_udmap_df.columns = ['key' + str(i) for i in range(1, 10)]

train_data = pd.concat([train_data, train_udmap_df], axis=1)
test_data = pd.concat([test_data, test_udmap_df], axis=1)

#%%
train_data['udmap_key'] = pd.factorize(train_data['udmap_key'])[0].astype(np.uint16)
test_data['udmap_key'] = pd.factorize(test_data['udmap_key'])[0].astype(np.uint16)
#%%
# print(train_data['udmap_key'].value_counts())
# %%
# 特征生成
train_data['eid_freq'] = train_data['eid'].map(train_data['eid'].value_counts())
test_data['eid_freq'] = test_data['eid'].map(train_data['eid'].value_counts())

train_data['eid_mean'] = train_data['eid'].map(train_data.groupby('eid')['target'].mean())
test_data['eid_mean'] = test_data['eid'].map(train_data.groupby('eid')['target'].mean())

train_data['udmap_isunknown'] = (train_data['udmap'] == 'unknown').astype(int)
test_data['udmap_isunknown'] = (test_data['udmap'] == 'unknown').astype(int)

train_data['common_ts_day'] = train_data['common_ts'].dt.day
test_data['common_ts_day'] = test_data['common_ts'].dt.day
train_data['common_ts_hour'] = train_data['common_ts'].dt.hour
test_data['common_ts_hour'] = test_data['common_ts'].dt.hour
train_data['common_ts_min'] = train_data['common_ts'].dt.minute
test_data['common_ts_min'] = test_data['common_ts'].dt.minute

#%%
# train -> training dataframe
# test -> test dataframe
# cat_cols -> categorical columns
one_hot_col = test_data.columns.tolist()
print(one_hot_col)
#%%
one_hot_col = ['x1', 'x2', 
                'x6', 'x7', 'udmap_key']
train_data_one_hot = pd.get_dummies(train_data, columns = one_hot_col)
# print(train_data_one_hot.columns)
#%%
test_data_one_hot = pd.get_dummies(test_data, columns = one_hot_col)
#%%
# print(train_data_one_hot['udmap_key_1'].value_counts())
#%%
# # Task 2.1
'''
#提取x1 - x8
x_lable = [x for x in train_data.columns if 'x' in x]
print(x_lable)
train_data_x = train_data[x_lable]
train_data_x.info()

观察数据分布情况，判断是否为类别数据或是数值数据
for i in range(1,9):
    train_data_x['x'+ str(i)].value_counts().plot(kind='bar')
    plt.title('x'+ str(i))
    plt.show()
**类别类型数据:x1 x2 x6 x7 x8**

**数值类型数据:x3 x4 x5**

#观察类别数据中的标签情况
for i in [1, 2, 6, 7, 8]:
    plt.figure()
    sns.barplot(x='x'+ str(i), y = 'target', data = train_data)

# 从common_ts中提取小时，绘制每小时下标签分布的变化
sns.barplot(x = 'common_ts_hour', y = 'target', data = train_data)

# 对udmap进行onehot，统计每个key对应的标签均值，绘制直方图
print(train_udmap_df.mean())
for i in range(7, 10):
    # print(train_data['key'+ str(i)].value_counts())
    # plt.title('key'+ str(i))
    plt.figure()
    plt.title('key'+str(i))
    sns.barplot(x = 'key' + str(i), y = 'target', data=train_data)

'''
# %%
# Task 2.2
# 导入模型

#%%
# x_train = train_data.drop(['udmap', 'common_ts', 'uuid', 'target'], axis=1)
y_train = train_data['target']
#%%
'''
model_list =[SGDClassifier(), 
             DecisionTreeClassifier(),
             MultinomialNB(),
             RandomForestClassifier(),
             GradientBoostingClassifier(),
             SGDOneClassSVM(),
             KNeighborsClassifier(2)]
model_name = ['SGD', 'dtree', 'Naivebayes', 'RF', 'GBTree', 'SVM', 'KNN']
models = zip(model_name, model_list)
for name, model in models:
    if name == 'KNN' or 'SVM':
        print('jump' + name)
        continue
    pred = cross_val_predict(model, 
                         x_train, 
                         y_train,
                         cv = 5)
    print('the f1-score for '+name)
    
    print(classification_report(y_train, pred, digits=3))
'''
#%% Evaluate the importance
'''
# clf_dc = DecisionTreeClassifier()
# clf_dc.fit(x_train, 
#            y_train)
# print('clf_dc1:')
# print(cross_val_score(clf_dc, x_train, y_train, scoring='accuracy', cv=5).mean())
# try:
#     x_train = x_train.drop(['eid_mean'], axis=1)
#     print(x_train.columns)
# except:
#     print('no means found')
#     print(x_train.columns)
# clf_dc.fit(x_train,
#            y_train)
# print('clf_dc2')
# print(cross_val_score(clf_dc, x_train, y_train, scoring='accuracy', cv=5).mean())

# importances_dc = clf_dc.feature_importances_
# train_data_dc = x_train
# print(importances_dc)
# # indices = np.argsort(importances_dc)[::-1] #reverse
# index_sorted = np.flipud(np.argsort(importances_dc))
# print(index_sorted)
# names = train_data_dc.columns
# print(names)
# plt.figure()
# plt.bar(names[index_sorted],importances_dc[index_sorted])
# plt.xticks(rotation=90, fontsize=14)
# plt.title('Decision Tree')

# #%%

# #%%
# clf_rf = RandomForestClassifier()
# clf_rf.fit(x_train, 
#            y_train)
# print('clf_rf1:')
# print(cross_val_score(clf_rf, x_train, y_train, scoring='accuracy', cv=5).mean())





# importances_rf = clf_rf.feature_importances_
# index_sorted = np.flipud(np.argsort(importances_rf))
# print(index_sorted)
# names = train_data_dc.columns
# print(names[index_sorted])
# plt.figure()
# plt.bar(names[index_sorted],importances_rf[index_sorted])
# plt.xticks(rotation=90, fontsize=14)
# plt.title('Random forest')
'''
#%%
print(train_data.columns)
#%%Task 2.3
# 特征构造
import traceback
clf_dc = DecisionTreeClassifier()


# deep copy!
train_data_ori = train_data.copy(deep=True)
train_data_new = pd.DataFrame()  
test_data_new = pd.DataFrame()



from betatarget import BetaEncoder
N_min = 1000
feature_cols = []  
for c in ['x1', 'x2', 'x5', 'x6',
       'x7',  'key4',
       'key5']:

    # fit encoder
    be = BetaEncoder(c)
    be.fit(train_data, 'target')

    # mean
    feature_name = f'{c}_mean'
    train_data_new[feature_name] = be.transform(train_data, 'mean', N_min)
    test_data_new[feature_name]  = be.transform(test_data,  'mean', N_min)
    feature_cols.append(feature_name)


    # # mode
    # feature_name = f'{c}_mode'
    # train_data_new[feature_name] = be.transform(train_data, 'mode', N_min)
    # test_data_new[feature_name]  = be.transform(test_data,  'mode', N_min)
    # feature_cols.append(feature_name)
    # # median
    # feature_name = f'{c}_median'
    # train_data_new[feature_name] = be.transform(train_data, 'median', N_min)
    # test_data_new[feature_name]  = be.transform(test_data,  'median', N_min)
    # feature_cols.append(feature_name) 

    # var
    feature_name = f'{c}_var'
    train_data_new[feature_name] = be.transform(train_data, 'var', N_min)
    test_data_new[feature_name]  = be.transform(test_data,  'var', N_min)
    feature_cols.append(feature_name)     
    
    # skewness
    feature_name = f'{c}_skewness'
    train_data_new[feature_name] = be.transform(train_data, 'skewness', N_min)
    test_data_new[feature_name]  = be.transform(test_data,  'skewness', N_min)
    feature_cols.append(feature_name)
    
    # kurtosis
    feature_name = f'{c}_kurtosis'
    train_data_new[feature_name] = be.transform(train_data, 'kurtosis', N_min)
    test_data_new[feature_name]  = be.transform(test_data,  'kurtosis', N_min)
    feature_cols.append(feature_name)  

train_data_new.fillna(0, inplace=True)
test_data_new.fillna(0, inplace=True)
train_data_new.replace(np.inf, 0, inplace=True)
test_data_new.replace(np.inf, 0, inplace=True)

#%%




# # 对x的频率和均值
# for i in range(1, 10):
#     if i == 3 or i == 8:
#         continue
#     try:
#         train_data_new['x'+str(i)+'_freq'] = train_data['x'+str(i)].map(train_data['x'+ str(i)].value_counts())
#         test_data_new['x'+str(i)+'_freq'] = test_data['x'+str(i)].map(train_data['x'+str(i)].value_counts())
#         train_data_new['x'+str(i)+'_mean'] = train_data['x'+str(i)].map(train_data.groupby('x'+str(i))['target'].mean())
#         test_data_new['x'+str(i)+'_mean'] = test_data['x'+str(i)].map(train_data.groupby('x'+str(i))['target'].mean())

#     except:
#         print(i)
#         print(traceback.format_exc())
#     train_data_new.fillna(0, inplace=True)
#     test_data_new.fillna(0, inplace=True)
#     train_data_new.replace(np.inf, 0, inplace=True)
#     test_data_new.replace(np.inf, 0, inplace=True)
    
# # 对key的频率和均值
# for i in range(1, 10):
#     try:
#         train_data_key['key'+str(i)+'_freq'] = train_data['key'+str(i)].map(train_data['key' + str(i)].value_counts())
#         test_data_key['key'+str(i)+'_freq'] = test_data['key'+str(i)].map(train_data['key'+str(i)].value_counts())
#         train_data_key['key'+str(i)+'_mean'] = train_data['key'+str(i)].map(train_data.groupby('key'+str(i))['target'].mean())
#         test_data_key['key'+str(i)+'_mean'] = test_data['key'+str(i)].map(train_data.groupby('key'+str(i))['target'].mean())
#     except:
#         print(i)
#         print(traceback.format_exc())
    # train_data_new.fillna(0, inplace=True)
    # test_data_new.fillna(0, inplace=True)
    # train_data_new.replace(np.inf, 0, inplace=True)
    # test_data_new.replace(np.inf, 0, inplace=True)
    

#%%
train_data_cross = pd.DataFrame()
test_data_cross = pd.DataFrame()
for i in ['x1', 'x2', 'x4', 'x5', 'x6',
       'x7', 'eid' ]:
    train_data_cross[i+ '_cnt'] = train_data[i].map(train_data[i].value_counts())
    test_data_cross[i+ '_cnt'] = test_data[i].map(test_data[i].value_counts())
    train_data_cross[i+ '_ratio'] = train_data_cross[i+ '_cnt'] / train_data.shape[0]
    test_data_cross[i+ '_ratio'] = test_data_cross[i+ '_cnt'] / test_data.shape[0]
train_data_cross.fillna(0, inplace=True)
test_data_cross.fillna(0, inplace=True)
train_data_cross.replace(np.inf, 0, inplace=True)
test_data_cross.replace(np.inf, 0, inplace=True)
#%%

for i in ['x1', 'x2', 'x4', 'x5', 'x6',
       'x7', 'eid' ]:
    for j in ['x1', 'x2', 'x4', 'x5', 'x6',
       'x7', 'eid' ]:
        if i == j :
            continue
        train_data_cross[i+ '_' + j + '_cnt'] = train_data_cross[i+ '_cnt'] * train_data_cross[j+ '_cnt']/ train_data.shape[0]
        train_data_cross[i+ '_' + j + '_cnt'] = test_data_cross[i+ '_cnt'] * test_data_cross[j+ '_cnt']/ test_data.shape[0]
        train_data_cross[i+ '_' + j + '_ratio'] = train_data_cross[i+ '_ratio'] * train_data_cross[j+ '_ratio']
        test_data_cross[i+ '_' + j + '_ratio'] = test_data_cross[i+ '_ratio'] * test_data_cross[j+ '_ratio']
        # train_data_cross[i+ '_' + j + '_cnt'] = train_data_cross[i+ '_cnt'] * train_data_cross[j+ '_cnt']/ train_data.shape[0]
        # train_data_cross[i+ '_' + j + '_cnt'] = test_data_cross[i+ '_cnt'] * test_data_cross[j+ '_cnt']/ test_data.shape[0]

train_data_cross.fillna(0, inplace=True)
test_data_cross.fillna(0, inplace=True)
train_data_cross.replace(np.inf, 0, inplace=True)
test_data_cross.replace(np.inf, 0, inplace=True)
print(train_data_cross.info())
    
#%%
# for i in range(1, 10):
#     print(train_data['key' + str(i)].value_counts())
#%%
train_data_split = train_data.copy(deep=True)
# print(train_data_split['key3'].value_counts())
train_data_split.loc[train_data_split['key3'].between(0, 52841.0, 'right'), 'key3'] = 1.0
train_data_split.loc[train_data_split['key3'].between(52841.0, 67804.0, 'right'), 'key3'] = 2.0
train_data_split.loc[train_data_split['key3'] > 67804.0, 'key3'] = 3.0
# print(train_data.info)
# print(train_data_split['key3'].value_counts())
#%%
train_data_split['key1'] = pd.cut(train_data_split['key1'], 3, labels=[0.0, 1.0, 2.0])
train_data_split.loc[train_data_split['key2'].between(0, 484.0, 'right'), 'key2'] = 1.0
train_data_split.loc[train_data_split['key2'].between(484.0, 10000.0, 'right'), 'key2'] = 2.0
train_data_split.loc[train_data_split['key6']  != 0.0, 'key6'] = 1.0
train_data_split = train_data_split[['key1', 'key2', 'key3', 'key6']]


test_data_split = test_data.copy(deep=True)

test_data_split.loc[test_data_split['key3'].between(0, 52841.0, 'right'), 'key3'] = 1.0
test_data_split.loc[test_data_split['key3'].between(52841.0, 67804.0, 'right'), 'key3'] = 2.0
test_data_split.loc[test_data_split['key3'] > 67804.0, 'key3'] = 3.0
# print(train_data.info)

test_data_split['key1'] = pd.cut(test_data_split['key1'], 3, labels=[0.0, 1.0, 2.0])
test_data_split.loc[test_data_split['key2'].between(0, 484.0, 'right'), 'key2'] = 1.0
test_data_split.loc[test_data_split['key2'].between(484.0, 10000.0, 'right'), 'key2'] = 2.0
test_data_split.loc[test_data_split['key6']  != 0.0, 'key6'] = 1.0

test_data_split = test_data_split[['key1', 'key2', 'key3', 'key6']]
#%%

# for i in [1, 2, 3]:
#     try:
#         train_data_split['key'+str(i)+'_freq'] = train_data_split['key'+str(i)].map(train_data_split['key' + str(i)].value_counts())
#         train_data_split['key'+str(i)+'_mean'] = train_data_split['key'+str(i)].map(train_data_split.groupby('key'+str(i))['target'].mean())
#         test_data_split['key'+str(i)+'_freq'] = test_data_split['key'+str(i)].map(train_data_split['key' + str(i)].value_counts())
#         test_data_split['key'+str(i)+'_mean'] = test_data_split['key'+str(i)].map(train_data_split.groupby('key'+str(i))['target'].mean())
#     except:
#         print(i)
#         print(traceback.format_exc())
    # train_data_split.fillna(0, inplace=True)
    # train_data_split.replace(np.inf, 0, inplace=True)


#%%

try:
    train_data.drop(['udmap', 'common_ts', 'uuid', 'target'], axis=1, inplace=True)    
except:
    print(traceback.format_exc())
# train_data.info()
# train_total = train_data_new.merge(train_data_new,how='left')
train_data_new['target'] = y_train
# train_data_key['target'] = y_train
# print(train_data_new.columns)
# print(train_data_key.columns)
#%%
    
# from itertools import combinations

# def combine(listshow):
#     '''
#     根据n获得列表中的所有可能组合（n个元素为一组）
#     '''
#     endlist = []
#     for i in range(len(listshow)):
#         if i not in [0,1,2]:
#             continue
#         temp_list2 = []
#         for c in combinations(listshow, i):
#             temp_list2.append(list(c))
#         endlist.extend(temp_list2)
#     return endlist
#%%
'''
新变量组合预测key
new_list = ['key1_freq', 'key1_mean', 'key2_freq', 'key2_mean', 'key3_freq',
       'key3_mean', 'key4_freq', 'key4_mean', 'key5_freq', 'key5_mean',
       'key6_freq', 'key6_mean', 'key7_freq', 'key7_mean', 'key8_freq',
       'key8_mean', 'key9_freq', 'key9_mean']
out_list = combine(new_list)


for i in range(len(out_list)):
    try:
        predict = cross_val_predict(clf_dc, 
                                    pd.concat([train_data, train_data_key[out_list[i]]], axis=1),
                                    y_train,
                                    cv=5)
        print('the f1 score for decision tree '+ str(out_list[i]))
        print(classification_report(y_train, predict, digits=3))
    except:
        print(i)
        print(traceback.format_exc())
        print('data wrong')
'''

# 新变量组合预测x
'''
new_list = ['x1_freq', 'x1_mean', 'x2_freq', 'x2_mean', 'x3_mean',
        'x4_mean', 'x5_mean', 'x6_freq', 'x6_mean',
       'x7_freq', 'x7_mean', 'x8_freq', 'x8_mean']
out_list = combine(new_list)


for i in range(len(out_list)):
    try:
        predict = cross_val_predict(clf_dc, 
                                    pd.concat([train_data, train_data_new[out_list[i]]], axis=1),
                                    y_train,
                                    cv=5)
        print('the f1 score for decision tree '+ str(out_list[i]))
        print(classification_report(y_train, predict, digits=3))
    except:
        print(i)
        print(traceback.format_exc())
        print('data wrong')
'''

# %%
print(train_data_one_hot.info())
#%%
# Train the DTree
temp_df = pd.concat([train_data_one_hot.drop(['key1', 'key2', 'key3', 'key6', 'udmap', 'common_ts', 'uuid', 'target', 'udmap_key_11'], axis=1), 
                    train_data_split,
                     train_data_new.drop(['target'], axis = 1)], axis=1)
score = cross_val_score(clf_dc,
                            temp_df.drop([ 'x3', 'x8', 'key7', 'key8', 'key9'],axis=1),
                            y_train,
                            cv=5,
                            scoring='f1')
print('the f1 score for decision tree ' +' is :')
print(score, score.mean())
#%%
# for i in train_data_cross.columns[14:]:
#     print(i)
#     temp_df = pd.concat([temp_df, train_data_cross[i]], axis=1)
#     score = cross_val_score(clf_dc,
#                             temp_df.drop([ 'x3', 'x8', 'key7', 'key8', 'key9'],axis=1),
#                             y_train,
#                             cv=5,
#                             scoring='f1')
#     print('the f1 score for decision tree ' + i +' is :')
#     print(score, score.mean())
# print(np.isinf(train_data).any()[np.isinf(train_data).any() == True])
#%%
# from sklearn.metrics import mutual_info_score
# # print(train_data.().all)
# mul = list(range(0, temp_df.columns.size))
# for i in range(0, temp_df.columns.size):
#     print('the mi for target and ' + str(temp_df.columns[i]) + ' is')
#     print(mutual_info_score(y_train, temp_df[temp_df.columns[i]]))
    
    
#%%
# Fit the model
clf_dc.fit(temp_df.drop(['x3', 'x8', 'key7', 'key8', 'key9'], axis=1),
           y_train)
# clf_dc.fit(temp_df,
#            y_train)
print(temp_df.columns)

from sklearn.model_selection import RandomizedSearchCV
clf_rf = RandomForestClassifier()
clf_GB = GradientBoostingClassifier()

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 3000, num = 15)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10, 15]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               }
random_grid_forRF = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap':bootstrap
               }


# score = cross_val_score(clf_rf,
#                         fit_train,
#                         y_train,
#                         cv=5,
#                         scoring='f1')
# print('the f1 score for RF with appending  :')
# print(score, score.mean())
# clfrf_opt = RandomizedSearchCV(estimator=clf_rf, param_distributions=random_grid_forRF, n_iter=30, cv=5, n_jobs = -1)
clf_rf.fit(temp_df.drop(['x3', 'x8', 'key7', 'key8', 'key9'], axis=1), y_train)
score = cross_val_score(clf_rf,
                            temp_df.drop([ 'x3', 'x8', 'key7', 'key8', 'key9'],axis=1),
                            y_train,
                            cv=5,
                            scoring='f1')

#%%
print(score.mean())


#%%
#%%
# # train_data.info()
# for i in range(1, 9):
#     try:
#         data = train_data_ori.insert(0, 'x'+str(i)+'_mean', train_data[ 'x'+str(i)+'_mean'].values)
#         data = train_data_ori.insert(0, 'x'+str(i)+'_freq', train_data[ 'x'+str(i)+'_freq'].values)
#     except:
#         print(traceback.format_exc())
#     try:
#         predict = cross_val_predict(clf_dc, 
#                                     data,
#                                     train_data['target'],
#                                     cv=5)
#         print('the f1 score for decision tree '+ str(i))
#         print(classification_report(train_data['target'], predict, digits=3))
#     except:
#         print(i)
#         print(traceback.format_exc())
#         print('data wrong')


#%%
# test_data_ori = test_data.copy(deep=True)

    
#%%



#%%
# print(test_data.info())
# print(test_data_new.info())
# try:
#     test_data_ori.drop(['udmap', 'common_ts', 'uuid'], axis=1, inplace=True)
# except:
#     print(traceback.format_exc())
# test_data_temp = pd.concat([test_data_ori, test_data_new[['x1_mean','x1_freq', 'x2_mean', 'x4_mean', 'x7_mean']]], axis=1)
test_data_temp = pd.concat([test_data_one_hot.drop(['key1', 'key2', 'key3', 'key6', 'udmap', 'common_ts', 'uuid'], axis=1), 
                            test_data_split, 
                            test_data_new], axis=1)

# print(test_data_temp.info())
# %%
pd.DataFrame({
    'uuid': test_data['uuid'],
    'target': clf_dc.predict(test_data_temp.drop([ 'x3', 'x8',  'key7', 'key8', 'key9'], axis=1))
}).to_csv('1001.csv', index=None)
pd.DataFrame({
    'uuid': test_data['uuid'],
    'target': clf_rf.predict(test_data_temp.drop([ 'x3', 'x8',  'key7', 'key8', 'key9'], axis=1))
}).to_csv('1201.csv', index=None)

#%%
# print(test_data_ori.info())
# .drop(['key5', 'key6', 'key7', 'key8', 'key9', 'x3'], axis=1)
# %%
