

#%%

# y也需要transform！这样算metrics才更好吧！
# 但是transform之后怎么回来呢
# 我可以跟踪记录变量名吗！应该可以的吧。现在是完全丢掉的。有必要吗，想一下吧


# 尤其注意输入的时候的变量类型，case by case去看。可能会有很多坑！
# 检查一个回归数据集！
# 展示的时候增量的来看效果

# todo：遗传规划，晚上做掉。他会有效果吗。疑问之。

# 先看怎么批量化的prepare数据集吧。

import pandas as pd
import numpy as np
from get_pipeline import create_pipeline
from freeprocess.todatawrapper import *
from get_data import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def mape(y,y_pred):
    return np.mean( np.abs( y_pred - y ) / np.abs(y) )



import pandas as pd
import numpy as np
from get_pipeline import create_pipeline,init_logger
from freeprocess.todatawrapper import *
from get_data import *
from sklearn.model_selection import train_test_split
from sklearn import metrics


# 参数：是否cross，是否gp，是否class weight

logger = init_logger(filename="diary_reg.log")


def mape(y,y_pred):
    return np.mean( np.abs( y_pred - y ) / np.abs(y) )

def regression_metrics(y,x_wrapper,model):

    y_pred = model.predict(x_wrapper) # 原始target，没有transform；y也是原始target


    logger.info(f"mape: {mape(y,y_pred)} " )
    logger.info(f"mse: {metrics.mean_squared_error(y,y_pred)} " )
    
    logger.info(f"log mape: {mape(np.log( y ), np.log(y_pred))} " )
    logger.info(f"log mse: {metrics.mean_squared_error(np.log(y),np.log(y_pred))} " )
    
    # y = x_wrapper.y_encoder.transform(y)
    # y_test_pred_label = x_wrapper.y_encoder.transform(y_test_pred_label)

    # logger.info(f"accuracy: {metrics.accuracy_score(y,y_test_pred_label)} " )
    # logger.info(f"F1 score:{metrics.f1_score(y,y_test_pred_label)}")
    # logger.info(f"Recall score:{metrics.recall_score(y,y_test_pred_label)}")
    # logger.info(f"Precision score:{metrics.precision_score(y,y_test_pred_label)}")
    # logger.info(f"auc值为:{metrics.roc_auc_score(y,y_test_pred_prob) }")



def run_horse(get_data_function):

    for enable_cross in [False,True]:
        for enable_gp in [False,True]:
                
            logger.info(f"\n")
            logger.info(f"********************************************")
            logger.info( f"cross{enable_cross};gp{enable_gp}" )


            x_train,y_train,x_test,y_test = get_data_function()
            todatawrapper = toDataWrapper()
            train_wrapper = todatawrapper.convert(x_train,y_train) 
            test_wrapper = todatawrapper.convert(x_test) 

            freemindpipeline = create_pipeline(enable_cross=enable_cross,
                                                enable_gp=enable_gp,
                                                )

            freemindpipeline.fit(train_wrapper)
            #y_train_predict = freemindpipeline.predict(train_wrapper)
            logger.info("train stat")
            regression_metrics(y_train,train_wrapper,freemindpipeline)
            logger.info("test stat")
            regression_metrics(y_test,test_wrapper,freemindpipeline)
get_data_funcions =[
  #  get_adult_data,
  #  get_bankchurn_data,
  #  get_crime_data,
    get_house_data
 #   get_rain_australia_data,
]

for get_data_function in get_data_funcions:
    logger.info(str(get_data_function))
    run_horse(get_data_function=get_data_function)
