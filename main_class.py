

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
from get_pipeline import create_pipeline,init_logger
from freeprocess.todatawrapper import *
from get_data import *
from sklearn.model_selection import train_test_split
from sklearn import metrics


# 参数：是否cross，是否gp，是否class weight

logger = init_logger(filename="diary_classi.log")


def mape(y,y_pred):
    return np.mean( np.abs( y_pred - y ) / np.abs(y) )

def classication_metrics(y,x_wrapper,model):

    y_test_pred_label = model.predict(x_wrapper) # 原始label
    y_test_pred_prob = model.predict_proba(x_wrapper)[:,1]

    # 两个都是原始label，要transform一下
    y = x_wrapper.y_encoder.transform(y)
    y_test_pred_label = x_wrapper.y_encoder.transform(y_test_pred_label)

    acc = metrics.accuracy_score(y,y_test_pred_label)
    f_score = metrics.f1_score(y,y_test_pred_label)
    recall = metrics.recall_score(y,y_test_pred_label)
    precision = metrics.precision_score(y,y_test_pred_label)
    auc = metrics.roc_auc_score(y,y_test_pred_prob) 

    logger.info(f"accuracy: {acc} " )
    logger.info(f"F1 score:{f_score}")
    logger.info(f"Recall score:{recall}")
    logger.info(f"Precision score:{precision}")
    logger.info(f"auc值为:{auc}")

    return acc,f_score,recall,precision,auc


def run_horse(get_data_function):

    stats = []
    for enable_cross in [False,True]:
        for enable_gp in [False,True]:
            for class_balance in [False,True]:
                
                logger.info(f"\n")
                logger.info(f"********************************************")
                logger.info( f"cross{enable_cross};gp{enable_gp};class_balance{class_balance}" )


                x_train,y_train,x_test,y_test = get_data_function()
                todatawrapper = toDataWrapper()
                train_wrapper = todatawrapper.convert(x_train,y_train) 
                test_wrapper = todatawrapper.convert(x_test) 

                freemindpipeline = create_pipeline(enable_cross=enable_cross,
                                                    enable_gp=enable_gp,
                                                    class_balance=class_balance
                                                    )

                freemindpipeline.fit(train_wrapper)
                #y_train_predict = freemindpipeline.predict(train_wrapper)
                logger.info("train stat")
                train_stats = classication_metrics(y_train,train_wrapper,freemindpipeline)

                stats.append(
                    {
                        'mode':'train',
                        'enable_cross':enable_cross,
                        'enable_gp':enable_gp,
                        'class_balance':class_balance,
                        'stats':str(train_stats)
                    }
                )


                logger.info("test stat")
                test_stats = classication_metrics(y_test,test_wrapper,freemindpipeline)


                stats.append(
                    {
                        'mode':'test',
                        'enable_cross':enable_cross,
                        'enable_gp':enable_gp,
                        'class_balance':class_balance,
                        'stats':str(test_stats)
                    }
                )

    return stats

get_data_funcions =[
   # ('adult', get_adult_data),
   # ('bankchurn', get_bankchurn_data),
    ('spam', get_spam_data),
   # ('rain_au', get_rain_australia_data),
]


for func_name, get_data_function in get_data_funcions:
    logger.info(func_name)
    stats = run_horse(get_data_function=get_data_function)

    pd.DataFrame(stats).to_csv(func_name+".csv",index=False)
