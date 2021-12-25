#%%
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


#%%


def get_adult_data():
    # classfication
    train = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        header=None
    )

    test = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        header=None,
        skiprows=1
    )
    test.columns = list(train.columns)

    y_train = train.pop(14)
    y_test = test.pop(14)
    y_test = y_test.str.replace('.','',regex=False)  # y_test有. but y_train没有，使其一致。

    return train,y_train,test,y_test


def get_rain_australia_data():
    # classfication
    df = pd.read_csv("datasets/rain_australia/weatherAUS.csv")
    df = df.dropna(subset=['RainTomorrow'])

    train, test = train_test_split(df,test_size=0.2,random_state=4396)

    y_train = train.pop('RainTomorrow')
    y_test = test.pop('RainTomorrow')

    return train,y_train,test,y_test


def get_kicked_data():
    # classfication
    train = pd.read_csv("datasets/DontGetKicked/training.csv")

    y_train = train.pop('IsBadBuy')

    test = pd.read_csv("datasets/DontGetKicked/test.csv")

    return train,y_train,test,None
# %%

def get_crime_data():
    # regression
    train = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data",header=None)
    
    # 删去等于1的
    mask = train.iloc[:,0]==1
    train = train[~mask]

    # ? 是代表缺失，手动识别了。识别之后依然是object对象，放在pipeline会被当做分类变量，所以再手动转换成float
    train = train.replace('?',np.nan)
    cols= train.select_dtypes('object').columns

    for col in cols:
        try:
            train[col] = train[col].astype('float')
            #print(i,"ok")
            #print(kk.head(1))
        except:
            print(col,"not ok")

    train, test = train_test_split(train,test_size=0.2,random_state=4396)
    
    y_train = train.pop(0)
    y_test = test.pop(0)

    return train,y_train,test,y_test

def get_house_data():
    # regression
    train = pd.read_csv("datasets/house_price/train.csv")
    y_train = train['SalePrice']

    test = pd.read_csv("datasets/house_price/test.csv",header=None)
    test.columns =train.columns
    y_test = test['SalePrice']

    train = train.drop(columns=['SalePrice','Id'])
    test = test.drop(columns=['SalePrice','Id'])

    return train,y_train,test,y_test

def get_bankchurn_data():
    # classfication
    train = pd.read_excel("datasets/bankchurn2/BankChurners2.xlsx",sheet_name='training')
    test = pd.read_excel("datasets/bankchurn2/BankChurners2.xlsx",sheet_name='testing')
    y_train = train['Attrition_Flag']
    y_test = test['Attrition_Flag']
    x_train = train.drop(columns=['id','Attrition_Flag'])
    x_test = test.drop(columns=['id','Attrition_Flag'])

    return x_train,y_train,x_test,y_test

def get_spam_data():
    # classfication
    train = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",header=None)
    train = train.dropna(subset=[57])
    y = train.pop(57)
    x_train, x_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=42)
    return x_train,y_train,x_test,y_test

def get_large_bankchurn_data():

    train = pd.read_csv("datasets/bankchurn_large/train.csv")
    train.pop("id")
    y_train = train.pop("isDefault")

    test = pd.read_csv("datasets/bankchurn_large/testA.csv")
    test.pop("id")

    return train,y_train,test,None
