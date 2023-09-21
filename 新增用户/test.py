#%%
import pandas as pd
from betatarget import BetaEncoder


N_min = 100
feature_cols = []    
#%%
train = pd.DataFrame({'deal_probability': [1, 0, 0], 'a':[1, 2, 1]})
test = pd.DataFrame({'deal_probability': [1, 0, 0], 'a':[1, 2, 2]})
print(train)
#%%
# encode variables
for c in ['a']:

    # fit encoder
    be = BetaEncoder(c)
    be.fit(train, 'deal_probability')

    # mean
    feature_name = f'{c}_mean'
    train[feature_name] = be.transform(train, 'mean', N_min)
    test[feature_name]  = be.transform(test,  'mean', N_min)
    train['mean2'] = train['a'].map(train.groupby('a')['deal_probability'].mean())
    test['mean2'] = test['a'].map(train.groupby('a')['deal_probability'].mean())

    feature_cols.append(feature_name)
    print(train)
    print(test)

    # # mode
    # feature_name = f'{c}_mode'
    # train[feature_name] = be.transform(train, 'mode', N_min)
    # test[feature_name]  = be.transform(test,  'mode', N_min)
    # feature_cols.append(feature_name)
    
    # # median
    # feature_name = f'{c}_median'
    # train[feature_name] = be.transform(train, 'median', N_min)
    # test[feature_name]  = be.transform(test,  'median', N_min)
    # feature_cols.append(feature_name)    

    # # var
    # feature_name = f'{c}_var'
    # train[feature_name] = be.transform(train, 'var', N_min)
    # test[feature_name]  = be.transform(test,  'var', N_min)
    # feature_cols.append(feature_name)        
    
    # # skewness
    # feature_name = f'{c}_skewness'
    # train[feature_name] = be.transform(train, 'skewness', N_min)
    # test[feature_name]  = be.transform(test,  'skewness', N_min)
    # feature_cols.append(feature_name)    
    
    # # kurtosis
    # feature_name = f'{c}_kurtosis'
    # train[feature_name] = be.transform(train, 'kurtosis', N_min)
    # test[feature_name]  = be.transform(test,  'kurtosis', N_min)
    # feature_cols.append(feature_name)  