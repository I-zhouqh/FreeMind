from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer,LabelEncoder
from sklearn.base import BaseEstimator,TransformerMixin
import sklearn.cluster as cluster
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
import math

from hyperopt import fmin,tpe,hp,partial,Trials,space_eval

from lightgbm import LGBMClassifier,LGBMRegressor
from xgboost import XGBClassifier,XGBRegressor
from catboost import CatBoostClassifier,CatBoostRegressor

"""
todo：

多分类和二分类没有去区别，我相信可以自动区别，吧？
没有均衡采样，没有管
category_data怎么加还没有搞！接口怎么加也还没有来得及搞。不过接口简单吧

把这三个模型并行一下ok


is it enough, to encode cols as category in pandas.,
or should I specify the cols in .fit ? 
"""






from sklearn.ensemble import StackingClassifier,StackingRegressor,VotingClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.model_selection import train_test_split



class Modeler(BaseEstimator,TransformerMixin):
    """对清洗好的数据进行建模

    先搞一个holdout set，在这上面分别把xgboost/lightgbm/catboost的参数调好【使用hyperopt工具进行搜参】
    然后使用最优的参数，对这三个模型stack，作为最终的模型。


    """
    def __init__(self,cv=5,use_raw_data = True,stack_jobs = 6,reverse_y=True,
                xgboost_serach_times=15,lgbm_search_times=25, catboost_search_times = 2, class_balance=True):
        """[summary]

        Parameters
        ----------
        cv : int, optional
            [description], by default 5
        use_raw_data : bool, optional
            [description], by default True
        stack_jobs : int, optional
            [description], by default 6
        reverse_y : bool, optional
            [description], by default True
        xgboost_serach_times : int, optional
            [description], by default 5
        lgbm_search_times : int, optional
            [description], by default 10
        catboost_search_times : int, optional
            [description], by default 2
        """
        self.cv = cv
        self.use_raw_data = use_raw_data
        self.stack_jobs = stack_jobs
        self.reverse_y = reverse_y 

        self.xgboost_serach_times = xgboost_serach_times
        self.lgbm_search_times=lgbm_search_times
        self.catboost_search_times = catboost_search_times

        self.class_balance = class_balance

    def fit(self,datawrapper,y):
        
        self.clf = self.get_super_estimator(datawrapper)

        return self

    def predict(self,datawrapper):
        # boom!
        # all over!

        x,category_columns = self.prepare_data(datawrapper)
        predict_value = self.clf.predict(x)

        if self.reverse_y:

            if self.problem == 'classification':
                return datawrapper.y_encoder.inverse_transform(predict_value)
            elif self.problem == 'regression':
                tmp = datawrapper.y_encoder.inverse_transform(predict_value.reshape(-1,1))
                return tmp.reshape(-1)
        else:
            return predict_value            

    def predict_proba(self,datawrapper):

        x,category_columns = self.prepare_data(datawrapper)
        return self.clf.predict_proba(x)

    def transform(self,datawrapper):
        # boom!
        # all over!
        x,category_columns = self.prepare_data(datawrapper)
        predict_value = self.clf.predict(x)

        if self.reverse_y:
            return datawrapper.y_encoder.inverse_transform(predict_value)
        else:
            return predict_value    


    def prepare_data(self,datawrapper):
        
        # numeric
        numeric = []
        if self.use_raw_data:
            numeric.append( datawrapper.x_numeric_rep  )
        else:
            numeric.append( datawrapper.x_numeric  )
        
        if hasattr(datawrapper,'x_numeric_crossed'):
            numeric.append( datawrapper.x_numeric_crossed  )
        if hasattr(datawrapper,'x_gp'):
            numeric.append( datawrapper.x_gp  )

        numeric = np.hstack(numeric)

        # category
        category = []
        if self.use_raw_data:
            category.append( datawrapper.x_category_rep  )
        else:
            category.append( datawrapper.x_category  )

        if hasattr(datawrapper,'x_category_crossed'):
            category.append( datawrapper.x_category_crossed  )         
        
        category = np.hstack(category)

        #data = np.hstack([numeric,category])
        
        df = pd.DataFrame(np.hstack([numeric,category]))

        for col in range( numeric.shape[1] , numeric.shape[1] + category.shape[1] ) :
            # change dtype to category, for lgbm and catboost
            df.iloc[:,col] = df.iloc[:,col].astype(int).astype('category')
        
        category_columns = list( range( numeric.shape[1] , numeric.shape[1] + category.shape[1] )  )

        return df,category_columns

    def get_super_estimator(self,datawrapper):

        x,category_columns = self.prepare_data(datawrapper)
        
        y = datawrapper.y
        self.problem = datawrapper.problem
        self.category_columns = category_columns
        # hold out的比例和stack的时候cv的比例相同，保证在hold out上找到的参数在stack的时候具有一致性
        x_train, x_eval, y_train, y_eval = train_test_split(x,y,test_size = 1/self.cv, random_state= 4396)

        xgboost_best= self.get_optimize_xgboost(x_train,y_train,x_eval,y_eval)
        lgbm_best = self.get_optimize_lgbm(x_train,y_train,x_eval,y_eval)
        #catboost_best = self.get_optimize_catboost(x_train,y_train,x_eval,y_eval)

        estimators = [
            ('xgboost', xgboost_best ),
            ('lgbm', lgbm_best ),
            #('catboost',catboost_best)
        ]

        if self.problem == 'regression':

            clf = StackingRegressor(
                estimators=estimators, 
                final_estimator=LinearRegression(),
                cv=self.cv,
                n_jobs = 2
            )
        else:

            class_weight = 'balanced' if self.class_balance else None
            clf = StackingClassifier(
                estimators=estimators, 
                final_estimator=LogisticRegression(class_weight=class_weight),
                cv=self.cv,
                n_jobs = 2
            )   

            # clf = VotingClassifier(
            #         estimators= estimators,
            #         voting='soft', 
            #         #weights=[2,1,1],
            #         n_jobs = 2
            # )

        # 使用全部数据 (x_train + x_eval) 进行预测，而不是x_train
        clf.fit(x, y,
                #category_feature = category_columns 
                )
        return clf


    def get_optimize_catboost(self, x_train,y_train,x_eval,y_eval):
        # in catboost, we first convert x to pandas and mark the category columns as 'category' dtype
        # and then sepcify categorical_feature in catboost.__init__. Therefore, no need to specity it in .fit

        if self.problem == 'regression':
            model_class = CatBoostRegressor
        elif self.problem == 'classification':
            model_class = CatBoostClassifier

        space_dtree = {
            'learning_rate':hp.uniform('learning_rate', 0.01,0.1  ),
            'depth':hp.choice('depth', [5,7,9,12]),
            'subsample':hp.uniform('subsample',0.5,0.8),
            'colsample_bylevel':hp.uniform('colsample_bytree',0.6,1.0),
            'l2_leaf_reg':hp.randint('l2_leaf_reg',3,50),
            'max_ctr_complexity':hp.choice('max_ctr_complexity',[2])  # 特征交叉数量
        }


        def hyperopt_model_score_catboost(params):
            model = model_class(**params,
                                    n_estimators = 3000,
                                    cat_features=self.category_columns  
                                    )
            
            model.fit(
                x_train,y_train,
                eval_set = [(x_eval,y_eval)],
                early_stopping_rounds= 50,
                #cat_features = self.category_columns,
            )

            # The best score is loss, so the smaller the better! not minor it !!
            if 'Logloss' in model.best_score_['validation']:
                return model.best_score_['validation']['Logloss']
            elif 'RMSE' in model.best_score_['validation']:
                return model.best_score_['validation']['RMSE']
            else:
                raise ValueError



        trials = Trials()
        best = fmin(
            fn=hyperopt_model_score_catboost, 
            space=space_dtree, 
            algo=tpe.suggest, 
            max_evals=self.catboost_search_times, 
            trials=trials
        )

        # hyperopt.choice return index instead of value, we need to convert it
        best = space_eval(space_dtree, best)

        # get best_iteration
        model = model_class(**best,
                                n_estimators = 3000,
                                cat_features=self.category_columns 
                                )
        
        model.fit(
            x_train,y_train,
            eval_set = [(x_eval,y_eval)],
            early_stopping_rounds= 50,
            #cat_features = self.category_columns,
        )

        # 这个best_iteration_作为之后使用的n_estimators
        best['n_estimators'] = model.best_iteration_

        print("best param for catboost:",best)

        return  model_class(**best,cat_features=self.category_columns )



    def get_optimize_xgboost(self,x_train,y_train,x_eval,y_eval):

        space_dtree = {
            'learning_rate':hp.uniform('learning_rate', 0.02,0.3  ),
            'max_depth':hp.choice('max_depth', [5,7,9]),
            'subsample':hp.uniform('subsample',0.5,0.8),
            'colsample_bytree':hp.uniform('colsample_bytree',0.6,1.0),

            'reg_alpha':hp.uniform('reg_alpha',0.0,1.0),
            'reg_lambda':hp.uniform('reg_lamda',0.0,1.0),
            'use_label_encoder':hp.choice('use_label_encoder',[False]),
        }

        if self.problem == 'regression':
            model_class = XGBRegressor
        elif self.problem == 'classification':
            model_class = XGBClassifier
            if self.class_balance:
                scale_pos_weight = np.sum(y_train==0) / np.sum(y_train==1)
                print("scale weight",scale_pos_weight)
                #exit()
                space_dtree['scale_pos_weight'] = hp.choice('scale_pos_weight',[scale_pos_weight])   



        def hyperopt_model_score_xgboost(params):
            model = model_class(**params,
                                    n_estimators = 3000,
                                    )
            
            model.fit(
                x_train,y_train,
                eval_set = [(x_eval,y_eval)],
                early_stopping_rounds= 50,
            )

            # The best score is loss, so the smaller the better! not minor it !!
            return model.best_score


        trials = Trials()
        best = fmin(
            fn=hyperopt_model_score_xgboost, 
            space=space_dtree, 
            algo=tpe.suggest, 
            max_evals=self.xgboost_serach_times, 
            trials=trials
        )

        # hyperopt.choice return index instead of value, we need to convert it
        best = space_eval(space_dtree, best)

        # get best_iteration
        model = model_class(**best, n_estimators = 3000,)

        model.fit(
            x_train,y_train,
            eval_set = [(x_eval,y_eval)],
            early_stopping_rounds= 50,
        )

        # 这个best_iteration_作为之后使用的n_estimators
        best['n_estimators'] = model.best_iteration

        print("best param for xgb:",best)

        return model_class(**best)



    def get_optimize_lgbm(self,x_train,y_train,x_eval,y_eval):

        # the x has specify the category variable as 'category' dtype
        # so we do not need to design categorical_feature

        space_dtree = {
            'learning_rate':hp.uniform('learning_rate', 0.02,0.3  ),
            'num_leaves':hp.choice('num_leaves', [32,64,128]),
            'subsample':hp.uniform('subsample',0.5,0.8),
            'colsample_bytree':hp.uniform('colsample_bytree',0.6,1.0),

            'reg_alpha':hp.uniform('reg_alpha',0.0,1.0),
            'reg_lambda':hp.uniform('reg_lamda',0.0,1.0),

            # constant params
            'subsample_freq':hp.choice('subsample_freq', [1]),
            'max_depth':hp.choice('max_depth', [15]),

        }

        if self.problem == 'regression':
            model_class = LGBMRegressor
        elif self.problem == 'classification':
            model_class = LGBMClassifier
            if self.class_balance:
                scale_pos_weight = np.sum(y_train==0) / np.sum(y_train==1)
                space_dtree['scale_pos_weight'] = hp.choice('scale_pos_weight',[scale_pos_weight])   




        def hyperopt_model_score_lgbm(params):
            model = model_class(**params, n_estimators = 3000, )
            
            model.fit(
                x_train,y_train,
                eval_set = [(x_eval,y_eval)],
                early_stopping_rounds= 50,
                #categorical_feature = self.category_columns,
            )

            # The best score is loss, so the smaller the better! not minor it !!
            # distinct regression and classfication

            if 'binary_logloss' in model.best_score_['valid_0']:
                return model.best_score_['valid_0']['binary_logloss']
            elif 'l2' in model.best_score_['valid_0']:
                return model.best_score_['valid_0']['l2']
            else:
                raise ValueError


        trials = Trials()
        best = fmin(
            fn=hyperopt_model_score_lgbm, 
            space=space_dtree, 
            algo=tpe.suggest, 
            max_evals=self.lgbm_search_times, 
            trials=trials
        )

        # hyperopt.choice return index instead of value, we need to convert it
        best = space_eval(space_dtree, best)

        # get best_iteration
        model = model_class(**best,n_estimators = 3000)
        
        model.fit(
            x_train,y_train,
            eval_set = [(x_eval,y_eval)],
            early_stopping_rounds= 50,
            #categorical_feature = self.category_columns,
        )

        # 这个best_iteration_作为之后使用的n_estimators
        best['n_estimators'] = model.best_iteration_

        print("best param for lgbm:",best)

        return  model_class(**best)