from sklearn.metrics import r2_score  # коэффициент детерминации  от Scikit-learn
from sklearn.metrics import mean_squared_error as mse  # метрика MSE от Scikit-learn
import warnings

from model_preparation import preparation

import pandas as pd  # Библиотека Pandas для работы с табличными данными
from sys import argv

warnings.filterwarnings('ignore')

# a, file_path = argv
file_path = "train/df_train_2.csv"


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


def main(file_path):
    df = read_file(file_path)

    # разбиваем на тестовую и валидационную
    dft = read_file("test/df_test_2.csv")
    x_test = dft.drop(['month'], axis=1)
    y_test = dft['month']

    model = preparation(df)

    print(f"r2 модели на тестовой выборке: {calculate_metric(model, x_test, y_test):.4f}")
    print(f"mse модели на тестовой выборке: {calculate_metric(model, x_test, y_test, mse):.4f}")
    print(f"rmse модели на тестовой выборке: {calculate_metric(model, x_test, y_test, mse, squared=False):.4f}")


main(file_path)