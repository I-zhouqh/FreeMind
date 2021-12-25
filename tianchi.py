import pandas as pd
import numpy as np
from get_pipeline import create_pipeline,init_logger
from freeprocess.todatawrapper import *
from get_data import *
from sklearn.model_selection import train_test_split
from sklearn import metrics


# 参数：是否cross，是否gp，是否class weight

logger = init_logger(filename="diary_classi.log")


def mape(y,y_pred):
    return np.mean( np.abs( y_pred - y ) / np.abs(y) )


x_train,y_train,x_test,y_test = get_large_bankchurn_data()
todatawrapper = toDataWrapper()
train_wrapper = todatawrapper.convert(x_train,y_train) 
test_wrapper = todatawrapper.convert(x_test) 

freemindpipeline = create_pipeline(enable_cross=True,
                                    enable_gp=True,
                                    class_balance=False
                                    )

freemindpipeline.fit(train_wrapper)
y_test_predict = freemindpipeline.predict_proba(test_wrapper)

pd.DataFrame(y_test_predict).to_csv("tianchi_predict.csv",index=False)