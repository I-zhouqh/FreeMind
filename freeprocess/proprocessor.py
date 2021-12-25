from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import BayesianRidge
from sklearn.impute import IterativeImputer
import pandas as pd
import numpy as np
from sklearn import pipeline
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin
import copy



class PreProcessor_simple(BaseEstimator,TransformerMixin):

    """数据预处理（主要是缺失值）
        
        连续变量的预处理。对于缺失比例小于0.05的变量，直接均值插补，对于缺失比例大于0.05小于
        0.3的变量，此处暂不处理，等待后续循环插补，对于缺失比例大于0.3的变量，创建indicaor变量并删除
        （在rep中仍有留存）。如果某个变量为常数（方差过小），则直接删除。

        分类变量的预处理。删除一些分类变量，如果最大水平的占比是dominate大于90%，或者如果变量水平太多，都会去掉。
        注意分类变量不用处理缺失值，因为缺失值在前面都当成一个新的类encode了。

        剩下的缺失值（连续缺失值）在下一个pipeline中进行迭代插补
    
    """

    def __init__(self):
        pass

    def remove_missing_row(self,datawrapper):
        # 缺失比例大于0.3的行，丢掉
        missing_num = np.isnan(datawrapper.x_numeric).sum(axis=1) + np.isnan(datawrapper.x_category).sum(axis=1)
        
        mask =  missing_num > 0.4 * ( datawrapper.x_numeric.shape[1] +  datawrapper.x_category.shape[1] )
        print(f"rows removed: {sum(mask)} ")

        datawrapper.drop_rows(mask) 


    
    def deal_category(self,datawrapper):    
        '''
        description:分类变量的预处理。删除一些分类变量，如果最大水平的占比是dominate大于90%，或者如果变量水平太多，都会去掉。
        注意分类变量不用处理缺失值，因为缺失值在前面都当成一个新的类encode了。

        param {*}
        return {*} rm_cols:删除列的编号
        '''        

        rm_cols = []
        x_category = datawrapper.x_category.copy()

        for i in range(x_category.shape[1]):
            series = x_category[:,i]

            unique_value, bin_counts = np.unique( series , return_counts=True)

            if max(bin_counts)/len(series) > 0.95 or  len(unique_value) > 0.1 * len(series):
                # 如果最大水平的占比是dominate大于90%
                # 如果变量水平太多，都会去掉
                rm_cols.append(i)

        return rm_cols

    def deal_numeric(self,datawrapper):
        '''
        description: 连续变量的预处理。对于缺失比例小于0.05的变量，直接均值插补，对于缺失比例大于0.05小于
        0.3的变量，此处暂不处理，等待后续循环插补，对于缺失比例大于0.3的变量，创建indicaor变量并删除
        （在rep中仍有留存）。如果某个变量为常数（方差过小），则直接删除。

        注意这里只是记录，在transform部分才变换


        return {*} y:asd
        '''        
        replace_dict = {}
        cols_create_indicator = []
        rm_cols = []

        x_numeric = datawrapper.x_numeric.copy()

        # 缺失值的处理
        for i in range( x_numeric.shape[1]  ):
            series = x_numeric[:,i]
            missing_rate = np.mean( np.isnan( series ) )

            if missing_rate<0.05:
                replace_dict[i] = np.nanmean(series)
            elif missing_rate<0.3:
                # wait for iterative impute
                pass
            else:
                cols_create_indicator.append(i)
                rm_cols.append(i)


        # 删除方差太小的变量
        for i in range( x_numeric.shape[1]  ):
            series = x_numeric[:,i]

            if np.var(series) < 1e-7:
                if i not in rm_cols:
                    rm_cols.append(i)


        return rm_cols, replace_dict, cols_create_indicator


    def iterative_impute(self,datawrapper):
        self.iterative_imputer = IterativeImputer(estimator=BayesianRidge(), n_nearest_features=None, imputation_order='ascending')
        self.iterative_imputer.fit(datawrapper.x_numeric)


    def fit(self,datawrapper):
        datawrapper = copy.deepcopy(datawrapper)
        # 去掉缺失值太多的行，这一步only in fit
        self.remove_missing_row(datawrapper)

        self.category_rm_cols = self.deal_category(datawrapper)
        self.numeric_rm_cols, self.numeric_replace_dict, self.numeric_nan_indicator = self.deal_numeric(datawrapper)

        return self


    def transform(self, datawrapper,**fit_params):
        datawrapper = copy.deepcopy(datawrapper)
        # 备份  (没有处理缺失值），为了建模的时候使用
        if datawrapper.create_rep:
            datawrapper.create_rep()

        # 创造indicator
        datawrapper.create_nan_indicator(self.numeric_nan_indicator,type='numeric' )
        # replace，针对的是常数插补
        datawrapper.replace_values(self.numeric_replace_dict,type='numeric' )
        # 删除列，数值型
        datawrapper.drop_cols(self.numeric_rm_cols,type='numeric')
        # 删除列，分类型
        datawrapper.drop_cols(self.category_rm_cols,type='category')
        
        return datawrapper


class PreProcessor_iterative(BaseEstimator,TransformerMixin):


    def fit(self,datawrapper):
        '''
        description: 迭代插补
        param {*} self
        param {*} datawrapper
        return {*}
        '''    
        self.iterative_imputer = IterativeImputer(estimator=BayesianRidge(), n_nearest_features=None, imputation_order='ascending')
        self.iterative_imputer.fit(datawrapper.x_numeric)

        return self
    
    def transform(self,datawrapper):

        datawrapper.x_numeric =  self.iterative_imputer.transform( datawrapper.x_numeric  )

        return datawrapper


preprocess_pipeline=Pipeline([
    ('PreProcessor_simple',PreProcessor_simple()),
    ('PreProcessor_iterative',PreProcessor_iterative())
])
 