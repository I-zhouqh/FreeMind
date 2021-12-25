'''
Author: your name
Date: 2021-12-12 17:55:36
LastEditTime: 2021-12-18 22:29:56
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \ML_final\freeprocess\gpgenerate.py
'''
from scipy.sparse import data
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer,LabelEncoder
from sklearn.base import BaseEstimator,TransformerMixin
import sklearn.cluster as cluster
from gplearn.genetic import SymbolicTransformer

def transform_numpy_with_replace_dict(data:np.ndarray, replace_dict):
    # numpy没有方便的函数做这个事情，只能靠pandas
    return pd.DataFrame(data).replace(replace_dict).values


class GpGenerate(BaseEstimator,TransformerMixin):

    def __init__(self,n_jobs=6,enable=True) -> None:
        
        self.n_jobs = n_jobs
        self.enable = enable

    def fit(self,datawrapper):
        if not self.enable:
            return self
        # Can I use oob to select model just here? or 把这个作为大参数和整个model一起搜参，which is cost
        # 所以能在这里搜参最好。

        """
        Hi @pGit1 , thanks for the kind words. It is difficult to have a one size fits all answer to this question unfortunately. But in general, population size, tournament size and parsimony are going to have a great effect on the size and fitness of the programs. Getting the balance right takes a lot of trial and error. I would generally start with fewer generations and see how the fitness is evolving while trying different tournament sizes and parsimony coeffs. Then maybe reign in the parsimony a bit and let it grow slower over more generations to get a better fit.

        I   f your research is academic, please consider a citation should a publication arise 👍
        """

        # 这样的话，即使没有上一步，我也可以做。就可以比较了
        if hasattr(datawrapper,'x_numeric_crossed'): 
            x_data = np.hstack([datawrapper.x_numeric,datawrapper.x_numeric_crossed])
        else:
            x_data = datawrapper.x_numeric.copy() 

        p = x_data.shape[1]

        function_set = ['add', 'sub', 'mul', 'div', 'log', 'sqrt', 'abs', 'neg', 'max', 'min']
        metrics_name = 'MI_r' if datawrapper.problem == 'regression' else 'MI_c'
        self.GPmodel = SymbolicTransformer(
                metric=metrics_name,

                generations=5,
                n_components = 10,
                population_size=50,
                hall_of_fame=25,
                parsimony_coefficient=0.001,
                function_set = function_set,
                stopping_criteria = 1e5,

                random_state=4396,
                n_jobs = self.n_jobs,
            )


        y = datawrapper.y.copy()
        self.GPmodel.fit(x_data, y )

        return self
    
    def transform(self,datawrapper):
        if not self.enable:
            return datawrapper


        if hasattr(datawrapper,'x_numeric_crossed'): 
            x_data = np.hstack([datawrapper.x_numeric,datawrapper.x_numeric_crossed])
        else:
            x_data = datawrapper.x_numeric.copy() 
        
        x_gp = self.GPmodel.transform(x_data)

        datawrapper.x_gp = x_gp


        return datawrapper
        
        # 联合x_GP,x_numeric_rep,x_category_rep,可以建模了？
        # how about x_numeric, x_category, x_cross不要了？
        # 也可以用验证来说吧。感觉超参真的很多了。