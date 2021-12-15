# import catboost


# import numpy as np

# np.expm1()

# import hypergbm

# # 样本多就stack （自己写可以，不用sklearn的stack）
# # 样本少就average。或者先做一版average？  stack太耗时间了主要是，样本多cv少一点还好。
# # 模型这里可以自己搜参（自己划一个验证集出来搜），也可以和整个pipeline一起搜。
# # 如果整个pipeline的参数比较少，那就一起搜吧。如果pipeline参数也多，那就各搜各的。

# # 算了，每个模型使用一个hold out，搜出自己最好的参数，然后三个average  （n_estimator不用搜吧，直接全部训完，节约资源啊）。
# # 大pipeline如果要搜，那就一个大hold out，然后搜。


# # 12.15更新！

# # 每个模型自己一个hold out找到最优参数

# # 然后三个模型/4个模型一起 去stack，stack可以用n_jobs，就sklearn的stack就醒了
# # 虽然stack要cv，但是在搜参的时候不cv，就无所谓。
# refer:
# https://xgboost.readthedocs.io/en/stable/python/python_api.html
# https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost/

# 是否可以选择正交还是不正交

# 超参的搜索就全部交给hyperopt了

# 整个hyperopt外面画一个验证集

# 然后XGB、LGB都不划出验证集，训练一个max_iter(比如2000)。

# 然后调的参数是predict的时候的num_rounds。

# from sklearn.ensemble import GradientBoostingRegressor


#%%

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn import metrics

model = XGBClassifier(learning_rate=0.01,
                      n_estimators=10,           # 树的个数-10棵树建立xgboost
                      max_depth=4,               # 树的深度
                      min_child_weight = 1,      # 叶子节点最小权重
                      gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                      subsample=1,               # 所有样本建立决策树
                      scale_pos_weight=1,        # 解决样本个数不平衡的问题
                      random_state=27,           # 随机数

                      )
# model.fit(X_train,
#           y_train)
# %%
import numpy as np
import xgboost

data = np.random.rand(5000,10) # 5 entities, each contains 10 features
label = np.random.randint(2, size=5000) # binary target
#dtrain = xgboost.DMatrix( data, label=label)
# %%

model.fit(data,label)
# %%

# test regression dataset
from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
# define dataset
x_train, y_train = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
x_eval, y_eval = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# summarize the dataset
# test classification dataset

#%%
x_train, y_train = make_classification(n_samples=1000000, n_features=10, n_informative=5, random_state=1,n_redundant=5)
x_eval, y_eval = make_classification(n_samples=10000, n_features=10, n_informative=5, random_state=1,n_redundant=5)
# define dataset

# summarize the dataset
#print(X.shape, y.shape)

# %%
model = XGBClassifier(
                      learning_rate=0.1,   # 0.05 , 0.1,  0.3
                      n_estimators=3000,          # 树的个数-10棵树建立xgboost
                      max_depth=4,               # 树的深度,[3,4,5]

                      
                      gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                      subsample=1,               # 所有样本建立决策树[0.7,0.9,1]
                      colsample_bytree=0.7,      # [0.7,0.9]
                      scale_pos_weight=1,        # 解决样本个数不平衡的问题,[True,False]
                      random_state=4396,           # 随机数
    # 'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
    # 'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
                      )

model.fit(x_train,y_train,
        eval_set=[(x_eval,y_eval)],
        eval_metric='logloss',
        early_stopping_rounds = 50
         )
# %%


model.best_score

# %%
model.best_iteration
# %%


from xgboost import XGBRegressor


XGBRegressor()


from lightgbm import LGBMClassifier,LGBMRegressor




model = LGBMClassifier(
                      learning_rate=0.1,   # 0.05 , 0.1,  0.3
                      n_estimators=3000,          # 树的个数-10棵树建立xgboost
                      max_depth=4,               # 树的深度,[3,4,5]

                      
                      gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                      subsample=1,               # 所有样本建立决策树[0.7,0.9,1]
                      colsample_bytree=0.7,      # [0.7,0.9]
                      scale_pos_weight=1,        # 解决样本个数不平衡的问题,[True,False]
                      random_state=4396,           # 随机数
    # 'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
    # 'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
                      )

model.fit(
        x_train,y_train,
        eval_set=[(x_eval,y_eval)],
        eval_metric='logloss',
        early_stopping_rounds = 50,

        categorical_feature=!!  # 这个怎么搞。迷茫。xgboost的这个直接忽略掉？

         )