from scipy.sparse import data
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer,LabelEncoder
from sklearn.base import BaseEstimator,TransformerMixin
import sklearn.cluster as cluster


def transform_numpy_with_replace_dict(data:np.ndarray, replace_dict):
    # numpy没有方便的函数做这个事情，只能靠pandas
    return pd.DataFrame(data).replace(replace_dict).values


class DataTransformer(BaseEstimator,TransformerMixin):
    """数据变换

    连续变量:robust scalar（没有正态化）
    分类变量:变量水平聚类。对于水平数较多的分类变量，使用他的响应概率进行k-means聚类，将水平数降到max_levels
    
    max_levels:一个变量最多允许的水平数
    
    Parameters
    ----------
    BaseEstimator : [type]
        [description]
    TransformerMixin : [type]
        [description]
    """

    def __init__(self,max_levels=50) -> None:
        self.max_levels = max_levels


    def fit(self,datawrapper):

        self.robust_scalar = self.deal_numeric(datawrapper)
        self.category_replace_table = self.deal_category(datawrapper)
        return self


    def transform(self, datawrapper,**fit_params):

        datawrapper.x_numeric = self.robust_scalar.transform( datawrapper.x_numeric)         
        datawrapper.x_category = transform_numpy_with_replace_dict(datawrapper.x_category, self.category_replace_table)

        return datawrapper

    def deal_numeric(self,datawrapper):
        # robust scalar
        # 但是异常值依然是异常值。

        robust_scalar = RobustScaler()
        robust_scalar.fit(datawrapper.x_numeric)
        return robust_scalar

    def deal_category(self,datawrapper):
        # 变量水平聚类 , kmeans走起来

        category_data = datawrapper.x_category.copy()

        max_levels = min(self.max_levels, len(datawrapper)*0.1 )  # 一个变量最多容许的水平

        result = {}
        for i in range(category_data.shape[1]):
            series = category_data[:,i]

            n_levels = len(np.unique( series))
            if  n_levels > max_levels:
  
            # 映射表，把哪些水平映射到哪些水平
                replace_dict = ReduceCategoryLevel( category_data[:,i], datawrapper.y, datawrapper.problem, max_levels )

                result[i] = replace_dict

                # 这个result长这样的：
                # {1: {1:0,2:3,4:5},
                #  2: {1:0,2:5,6:3},
                #  3: {1:9,2:0,3:3}   }
                # 于是每列挨着去replace就行了。

        return result






def ReduceCategoryLevel(x_series, y, problem,levels):
    if problem == 'regression':
        mm = pd.DataFrame(np.vstack([x_series,y]).T ).groupby(0).mean()[1]
        level_name = mm.index
        clustering_matirx = mm.value.reshape(-1,1)


    elif problem == 'classification':
        
        minv = y.min()
        maxv = y.max()

        def my_deal(df):
            series = df[1]
            #print(series)
            result = []
            for i in range(minv,maxv+1):

                result.append( sum(series==i) / len(series) )
            return np.array(result)

        mm = pd.DataFrame(np.vstack([x_series,y]).T ).groupby(0).apply(my_deal)

        level_name = mm.index
        clustering_matirx = np.stack(mm.values)

    kmeans = cluster.KMeans(n_clusters = levels).fit(clustering_matirx)

    # level_name: 0,1,2,3,4,5 . 检查一下数据类型，我总觉得int会变成float

    # 这时候的cluster_map_table就是 {k:v},k是原来的水平，v是水平聚类之后的簇
    cluster_map_table = {k:v for k,v in zip(level_name.tolist(),kmeans.labels_.tolist())}

    return cluster_map_table