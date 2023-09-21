
#     # mode
#     feature_name = f'{c}_mode'
#     train[feature_name] = be.transform(train, 'mode', N_min)
#     test[feature_name]  = be.transform(test,  'mode', N_min)
#     feature_cols.append(feature_name)
    
#     # median
#     feature_name = f'{c}_median'
#     train[feature_name] = be.transform(train, 'median', N_min)
#     test[feature_name]  = be.transform(test,  'median', N_min)
#     feature_cols.append(feature_name)    

#     # var
#     feature_name = f'{c}_var'
#     train[feature_name] = be.transform(train, 'var', N_min)
#     test[feature_name]  = be.transform(test,  'var', N_min)
#     feature_cols.append(feature_name)        
    
#     # skewness
#     feature_name = f'{c}_skewness'
#     train[feature_name] = be.transform(train, 'skewness', N_min)
#     test[feature_name]  = be.transform(test,  'skewness', N_min)
#     feature_cols.append(feature_name)    
    
#     # kurtosis
#     feature_name = f'{c}_kurtosis'
#     train[feature_name] = be.transform(train, 'kurtosis', N_min)
#     test[feature_name]  = be.transform(test,  'kurtosis', N_min)
#     feature_cols.append(feature_name)  