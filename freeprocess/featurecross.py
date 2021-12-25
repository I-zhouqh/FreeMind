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

def transform_numpy_with_replace_dict(data:np.ndarray, replace_dict):
    # numpy没有方便的函数做这个事情，只能靠pandas
    return pd.Series(data).replace(replace_dict).values



def np_groupby(a,b,func):
    """
    
    a: group by number
    b: number to summary
    func:
    """
    map_dict={}
    unique_value = np.unique(a)
    for value in unique_value:
        mask = a==value
        map_dict[value] = func(b[mask])
    return map_dict

def get_numeric_mutual_info_columns(datawrapper,keep_ratio ):
    x = datawrapper.x_numeric
    n,p = x.shape

    y = datawrapper.y

    # numeric, but the values it pick are discrete，比如年龄
    discrete_features = []

    for i in range( p ):
        if len( np.unique( x[:,i] ) ) <= max(256, 0.03 * n):
            discrete_features.append(i)

    # 特意识别出discrete的数值特征是为了互信息算的更准确，见https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html
    if datawrapper.problem == 'regression':
        mi_scores = mutual_info_regression(x, y, discrete_features=discrete_features )

    elif datawrapper.problem == 'classification':
        mi_scores = mutual_info_classif(x, y, discrete_features=discrete_features )

    select_num = min(12, math.ceil( p * keep_ratio ) )

    # 前select_num个值的序号
    return np.argsort(mi_scores)[::-1][:select_num]

def get_category_mutual_info_columns(datawrapper,keep_ratio):
    if hasattr(datawrapper,'x_category_crossed'):
        x = np.hstack( [datawrapper.x_category,datawrapper.x_category_crossed]   )
    else:
        x = datawrapper.x_category

    if x.shape[1]<=0:
        return []

    n,p = x.shape

    y = datawrapper.y

    # 特意识别出discrete的数值特征是为了互信息算的更准确，见https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html
    if datawrapper.problem == 'regression':
        mi_scores = mutual_info_regression(x, y, discrete_features=True )

    elif datawrapper.problem == 'classification':
        mi_scores = mutual_info_classif(x, y, discrete_features=True )

    select_num = min(12, math.ceil(p * keep_ratio ) )

    # 前select_num个值的序号
    return np.argsort(mi_scores)[::-1][:select_num]




# 不可能所有值都去交叉，根据互信息选出一定数量的特征（根据featuresize来定），他们才可以交叉。
# 先离散X离散： 笛卡尔积，互信息比较高的特征，且水平数不能太多。
# 离散X连续： groupby算统计量（现在用的是mean,std)，然后以这个值给numeric encode，加入x_numeric
# 然后和原始的x_numeric combine在一起就可以送进gp。









class CategoryFeatureCross(BaseEstimator,TransformerMixin):
    """离散特征交叉

    选出互信息比较高（由keep_ratio决定前百分之几）的离散特征（且分类水平数比较小），做笛卡尔积，生成新的离散特征

    新产生的特征存为datawrapper.x_category_crossed

    """

    def __init__(self,keep_ratio=0.5):
        self.keep_ratio = keep_ratio

    def fit(self,datawrapper):

        self.select_features = get_category_mutual_info_columns(datawrapper,self.keep_ratio)

        if len(self.select_features) <= 1:
            return self

        select_feature_df = pd.DataFrame( datawrapper.x_category[:,self.select_features] )

        cross_result_df = pd.DataFrame()

        for i_index in range(len(self.select_features)):
            for j_index in range(i_index+1,len(self.select_features)):
                series_i =  select_feature_df.iloc[:,i_index]
                series_j =  select_feature_df.iloc[:,j_index]

                

                if series_i.nunique() * series_j.nunique() > 255:
                    # 类别数太多就算了，handle不了。
                    continue
                
                print(series_i.value_counts())
                print(series_j.value_counts())

                new_col_name = str(i_index) + '*' + str(j_index)
                cross_result_df[new_col_name] = series_i.astype('str') + '*' + series_j.astype('str')

        self.ordinal_encoder= OrdinalEncoder()
        self.ordinal_encoder.fit( cross_result_df  )

        return self


    def transform(self,datawrapper):
        if len(self.select_features) <= 1:
            return datawrapper
        
        # 选出目标特征
        select_feature_df = pd.DataFrame( datawrapper.x_category[:,self.select_features] )

        # transform成笛卡尔积
        cross_result_df = pd.DataFrame()
        for i_index in range(len(self.select_features)):
            for j_index in range(i_index+1,len(self.select_features)):
                series_i =  select_feature_df.iloc[:,i_index]
                series_j =  select_feature_df.iloc[:,j_index]

                if series_i.nunique() * series_j.nunique() > 255:
                    # 类别数太多就算了，handle不了。
                    continue

                new_col_name = str(i_index) + '*' + str(j_index)
                cross_result_df[new_col_name] = series_i.astype('str') + '*' + series_j.astype('str')


        from .utils import handle_unknown
        x_category_crossed = handle_unknown( cross_result_df ,self.ordinal_encoder  )
        x_category_crossed = self.ordinal_encoder.transform( cross_result_df )
        #x_category = self.ordinal_encoder.transform(x_category)



        #x_category_crossed = self.ordinal_encoder.transform( cross_result_df )

        datawrapper.x_category_crossed =  x_category_crossed 
        #datawrapper.x_category = np.hstack( [datawrapper.x_category , x_cross_category  ] )


        return datawrapper


class NCFeatureCross(BaseEstimator,TransformerMixin):
    """离散特征与连续特征交叉

    选出互信息前30%的离散特征、互信息前30%的连续特征，做交叉

    交叉的方法是
    
    for i in 离散变量
        for j in 连续变量
            group by列i，算列j的std与mean，作为新的特征
    
    交叉处的特征存为 datawrapper.x_numeric_crossed

    """

    def __init__(self,keep_ratio=0.3):
        self.keep_ratio = keep_ratio


    def fit(self,datawrapper):
        
        numeric_select_cols =  get_numeric_mutual_info_columns(datawrapper,self.keep_ratio)
        category_select_cols = get_category_mutual_info_columns(datawrapper,self.keep_ratio)


        replace_material = []
        """
        replace_meterial:

        [
            (1, {3:1.5,4:3.8,5:4.8}          ),
            (1, {3:1.6,1:3.8,12:4.8}          ),
            (2, {3:1.7,2:3.8,16:4.8}          ),
            (2, {3:1.8,1:3.8,17:4.8}          )
        ]
        # 可以理解为每个category_column的映射值，映射函数不止一个，比如分组均值、分组标准差，所以可能有很多个
        """

        if hasattr(datawrapper,'x_category_crossed'):
            category_data = np.hstack( [datawrapper.x_category,datawrapper.x_category_crossed]   )
        else:
            category_data = datawrapper.x_category.copy()

        for category_col in category_select_cols:
            category_series = category_data[:,category_col]

            for numeric_col in numeric_select_cols:
                numeric_series = datawrapper.x_numeric[:,numeric_col]

                map_dict_mean = np_groupby(category_series,numeric_series,np.mean)
                map_dict_std = np_groupby(category_series,numeric_series,np.std)

                replace_material.append( (category_col,map_dict_mean)   )
                replace_material.append( (category_col,map_dict_std)   )
        
        self.replace_material = replace_material

        return self


    def transform(self,datawrapper):

        if hasattr(datawrapper,'x_category_crossed'):
            x_category = np.hstack( [datawrapper.x_category,datawrapper.x_category_crossed]   )
        else:
            x_category = datawrapper.x_category.copy()

        x_nc_cross = []
        for col, map_dict in self.replace_material:
            data = x_category[:,col]
            x_nc_cross.append(
                     transform_numpy_with_replace_dict( data, map_dict    )
            )

        if len(x_nc_cross)==0:
            return datawrapper
        else:
            x_numeric_crossed = np.stack( x_nc_cross).T
            datawrapper.x_numeric_crossed = x_numeric_crossed

            return datawrapper

class ScaleCrossFeature(BaseEstimator,TransformerMixin):
    def fit(self,datawrapper):
        if not hasattr(datawrapper,'x_numeric_crossed'):
            return self
        self.robust_scalar = RobustScaler()
        self.robust_scalar.fit(datawrapper.x_numeric_crossed)
        return self

    def transform(self,datawrapper):        
        if not hasattr(datawrapper,'x_numeric_crossed'):
            return datawrapper      
        datawrapper.x_numeric_crossed = self.robust_scalar.transform(datawrapper.x_numeric_crossed)
        return datawrapper

class NullPipeline(BaseEstimator,TransformerMixin):
    # just a place holder
    def fit(self,datawrapper):
        return self

    def transform(self,datawrapper):        
        return datawrapper

featurecross=Pipeline([
    ('PreProcessor_simple',CategoryFeatureCross()),
    ('PreProcessor_iterative',NCFeatureCross()),
    ('ScaleCrossFeature',ScaleCrossFeature())
])
 