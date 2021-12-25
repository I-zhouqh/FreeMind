from os import replace
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,LabelEncoder
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import PowerTransformer,LabelEncoder


# todo:要能够删除一些列？比如ID。同样要能够删除constant列  if train_df[col].std() == < 1e-7:  
# 消除重复列？which cost much

class toDataWrapper:


    def __init__(self,create_rep = True,supple_category_cols=[],drop_cols=[]):
        '''
        description: 负责把数据打包成后续需要的模样
        包括：
        1. 确定问题的种类，是分类还是回归
        2. 转换y，是分类labelencoder，是回归则尽量正态，同时保留encoder为了后续inverse_transform
        3. 根据x的dtypes以及用户自己补充的supple_category_feature确定出分类变量，剩下的为连续变量。对于分类变量使用ordinal_encoder(缺失也作为一类)
        4. 是否创建备份（备份主要用于建模）

        param {*} create_rep: 是否创建备份
        param {*} supple_category_feature: 补充的分类变量。模型自动只会把object识别为分类变量，如果还有其他数值为int，但实际为分类变量的，需要用这个指出。
        return {*}
        '''        

        self.create_rep = create_rep
        self.supple_category_cols = supple_category_cols
        self.drop_cols = drop_cols

    def convert(self,x,y = None):
        '''
        description: 转换成datawrapper，如果有y则是训练的模式，如果没有y则是测试的模式
        param {*}
        return {*}
        '''        
        if y is not None:
            return self.convert_train_mode(x.copy(),y.copy())
        else:
            return self.convert_eval_mode(x.copy())

    def convert_train_mode(self,x, y):


        ## 处理columns
        x = x.drop(columns=self.drop_cols)
        self.columns = x.columns
        self.category_columns = list( set(  list( x.select_dtypes('object').columns ) + self.supple_category_cols )  )
        self.numeric_columns =  list( set(self.columns) - set(self.category_columns) )


        ## 处理x
        #categories_columns =  list(range(len(self.category_columns)))
        #The OrdinalEncoder does not have handle_unknown, so we have to handle it manually in transform
        self.ordinal_encoder= OrdinalEncoder()

        #kkk = x[self.category_columns].fillna(value='MissingValue')
        x_category = self.ordinal_encoder.fit_transform( x[self.category_columns].fillna(value='MissingValue')   )

        x_numeric = x[self.numeric_columns].values


        ## 处理y，仅对训练数据有
        if y.isna().sum() > 0:
            raise ValueError
        
        y = y.values
        print(y.dtype)
        if y.dtype in [object] or len(np.unique(y))<10 :
            self.problem = 'classification'
        else:
            self.problem = 'regression'
        # else:
        #    raise ValueError

        if self.problem == 'classification':
            self.y_encoder =  LabelEncoder()
            y = self.y_encoder.fit_transform(y)
        
        elif self.problem == 'regression':
            self.y_encoder =  PowerTransformer()
            y = self.y_encoder.fit_transform(y.reshape(-1,1))
            y = y.reshape(-1)

        return DataWrapper(x_numeric,x_category,y,self.problem,self.y_encoder,self.create_rep)


    def convert_eval_mode(self,x):
        
        ## 处理x
        x = x.drop(columns=self.drop_cols)
        x = x[self.columns]  #让列的顺序一致。同时如果列不一致，也会报错

        from .utils import handle_unknown
        x_category = handle_unknown( x[self.category_columns].fillna(value='MissingValue') ,self.ordinal_encoder  )
        x_category = self.ordinal_encoder.transform(x_category)

        #x_category = self.ordinal_encoder.transform( x[self.category_columns].fillna(value='MissingValue')  )
        x_numeric = x[self.numeric_columns].values

        return DataWrapper(x_numeric,x_category,None,self.problem,self.y_encoder,self.create_rep)  


class DataWrapper:

    def __init__(self,x_numeric,x_category,y,problem,y_encoder,is_create_rep):
        '''
        description: 打包数据，方便后续处理
        param {*} x_numeric:数值型变量，numpy
        param {*} x_category:分类型变量，numpy
        param {*} y:target
        param {*} problem:字符串，分类还是回归
        param {*} y_encoder:给y encode


        return {*}
        '''        
        self.x_numeric = x_numeric
        self.x_category = x_category.astype(int)
        self.y = y
        self.problem = problem
        self.y_encoder = y_encoder  # y的编码器，可以逆解码
        self.is_create_rep = is_create_rep
        # columns of dataframe
        # self.colnames = x.columns
        # # 以后拿到新数据集要transform的时候一定要check这个columns对不对劲！

        # self.x_numeric =  x.select_dtypes(include='number').values
        # self.x_category = x.select_dtypes(include='object').values
        # self.y = y
    
    def drop_rows(self,mask):
        
        self.x_numeric = self.x_numeric[~mask]
        self.x_category = self.x_category[~mask]
        self.y = self.y[~mask]
    
    def __len__(self):
        return len(self.y)

    def create_rep(self):

        self.x_numeric_rep = self.x_numeric.copy()
        self.x_category_rep = self.x_category.copy()

    def create_nan_indicator(self,cols,type):
        '''
        description: 
        param {*}
        return {*}
        '''        

        if type=='numeric':
            
            indicators = []
            for col in cols:
                indicators.append( np.isnan(self.x_numeric[:,col]).astype(int) )
            
            if len(indicators)>0:
                self.x_numeric = np.hstack(  [  self.x_numeric,  np.stack(indicators).T  ] )

        elif type=='category':

            indicators = []
            for col in cols:
                indicators.append( np.isnan(self.x_category[:,col]).astype(int) )
            
            if len(indicators)>0:
                self.x_category = np.hstack(  [  self.x_category,  np.stack(indicators).T  ] )

        else:
            raise ValueError
    
    def replace_values(self,replace_dict,type):

        if type=='numeric':

            for col,fill_value in replace_dict.items():
                self.x_numeric[:,col] = np.nan_to_num(self.x_numeric[:,col],nan=fill_value)

        elif type=='category':

            for col,fill_value in replace_dict.items():
                self.x_category[:,col] = np.nan_to_num(self.x_category[:,col],nan=fill_value)

        else:
            raise ValueError

    def drop_cols(self,rm_cols,type='numeric'):
        
        if type=='numeric':

            self.x_numeric = np.delete(self.x_numeric,rm_cols,axis=1)

        elif type=='category':
            self.x_category = np.delete(self.x_category,rm_cols,axis=1)

        else:
            raise ValueError     
    
