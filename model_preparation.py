from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler  # Импортируем стандартизацию от scikit-learn
from sklearn.model_selection import train_test_split  # функция разбиения на тренировочную и тестовую выборку
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor  # Линейная регрессия с градиентным спуском от scikit-learn
import warnings

import pandas as pd  # Библиотека Pandas для работы с табличными данными

warnings.filterwarnings('ignore')


def read_file(file_path):
    try:
        df = pd.read_csv(file_path)
        df['day_mean_temp'] = pd.to_numeric(df['day_mean_temp'])
        df = df.fillna(0)
        print("File " + file_path + " readed successfully.")
        return df
    except IOError:
        print("Error uccured while readed file '" + file_path + "'.")


def preparation(df):
    num_pipe_day_mean_temp = Pipeline([('scaler', StandardScaler())])
    num_day_mean_temp = ['day_mean_temp']

    preprocessor = ColumnTransformer(transformers=[('num_day_mean_temp', num_pipe_day_mean_temp, num_day_mean_temp)])

    # df = read_file(file_path)
    # не забываем удалить целевую переменную цену из признаков
    x_train = df.drop(['month'], axis=1)
    y_train = df['month']

    # Сначала обучаем на тренировочных данных
    x_train_prep = preprocessor.fit_transform(x_train)
    model = SGDRegressor(random_state=42)
    model.fit(x_train_prep, y_train)

    return model
