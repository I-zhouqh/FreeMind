import random

def handle_unknown(category_data, category_encoder, fill_value = 'MissingValue' ):
    """把这份category_data中没有被category_encoder见过的全部替换成fill_value
    主要是sklearn的ordinal_encoder没有支持这个功能，只能自己写，无语

    Parameters
    ----------
    category_data : 想要转换的类别数据
        [description]
    category_encoder : OrdinalEncoder
        [description]
    fill_value : str, optional
        [description], by default 'MissingValue'
    """

    exist_levels = category_encoder.categories_.copy()

    assert category_data.shape[1] == len(exist_levels)

    for i in range(len(exist_levels)):

        mask =  category_data.iloc[:,i].isin( category_encoder.categories_[i] )

        series = category_data.iloc[:,i].copy()

        # 有些时候categories_里面连'MissingValue'都没有，如果改成MissingValue会继续报错
        # 这个时候就随机取一个值吧
        if fill_value in category_encoder.categories_[i]:
            series[~mask] = fill_value
        else:
            series[~mask] = random.choice( category_encoder.categories_[i] )

        category_data.iloc[:,i]  = series

    return category_data