import math

def get_mse(records_real, records_predict):
    """
    均方误差
    """
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None
def get_rmse(records_real, records_predict):
    """
    均方根误差
    """
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None