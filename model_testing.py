from sklearn.metrics import r2_score  # коэффициент детерминации  от Scikit-learn
from sklearn.metrics import mean_squared_error as mse  # метрика MSE от Scikit-learn
import warnings
import os.path
import fnmatch
from random import randint

from model_preparation import preparation

import pandas as pd  # Библиотека Pandas для работы с табличными данными

warnings.filterwarnings('ignore')

train_df_path = "train/df_train_" + str(randint(0, len(fnmatch.filter(os.listdir('test/'), '*.*')) - 1)) + ".csv"
test_df_path = "test/df_test_" + str(randint(0, len(fnmatch.filter(os.listdir('test/'), '*.*')) - 1)) + ".csv"

print("<< Start model testing >>")


def read_file(file_path):
    try:
        df = pd.read_csv(file_path)
        df['day_mean_temp'] = pd.to_numeric(df['day_mean_temp'])
        df = df.fillna(0)
        print("File " + file_path + " readed successfully.")
        return df
    except IOError:
        print("Error uccured while readed file '" + file_path + "'.")


def calculate_metric(model_pipe, x, y, metric=r2_score, **kwargs):
    """Расчет метрики.
    Параметры:
    ===========
    model_pipe: модель или pipeline
    X: признаки
    y: истинные значения
    metric: метрика (r2 - по умолчанию)
    """
    y_model = model_pipe.predict(x)
    return metric(y, y_model, **kwargs)


def main(train_df_path, test_df_path):
    df = read_file(train_df_path)

    # разбиваем на тестовую и валидационную
    dft = read_file(test_df_path)
    x_test = dft.drop(['month'], axis=1)
    y_test = dft['month']

    model = preparation(df)

    print("\nModel results:")
    print(f"r2 of model on test data: {calculate_metric(model, x_test, y_test):.4f}")
    print(f"mse of model on test data: {calculate_metric(model, x_test, y_test, mse):.4f}")
    print(f"rmse of model on test data: {calculate_metric(model, x_test, y_test, mse, squared=False):.4f}")


main(train_df_path, test_df_path)
print("<< Finish model testing >>")
