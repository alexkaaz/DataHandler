import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

file = "house_prices_dataset.csv"


# Позволяет выполнять базовые взаимодействия с датасетом
class BasicDataInteraction():

    def __init__(self, df):
        self.df = df
        self.forest = RandomForestClassifier(random_state=32)
        self.lime = LinearRegression()

    # Считает метрики классификатора
    def __classification_metric__(self, test, pred):
        print(f'Accuracy: {metrics.accuracy_score(test, pred)}')
        print(f'Precision: {metrics.precision_score(test, pred, average='weighted')}')
        fpr, tpr, treshhold = metrics.roc_curve(test, pred, pos_label=1)
        print(fpr, tpr, treshhold)
        print(f'AUC: {metrics.auc(fpr, tpr)}')
        
    def __regression_metric__(self, test, pred):
        print(f'MAE: {metrics.mean_absolute_error(pred, test)}')
        print(f'MAPE: {metrics.mean_absolute_percentage_error(pred, test)}')
        print(f'R2: {metrics.r2_score(test, pred)}')

    # Выводит основуню информацию по датасету 
    def short_cut(self):
        print('head:\n', self.df.head())
        print("info:\n", self.df.info())
        print("descibe:\n", self.df.describe())
        print("missing data:\n", self.df.isnull().sum())

    # Возвращает название не числовых столбцов 
    def not_num_columns(self):
        no_num_cols = []
        for col in self.df.columns:
            column = self.df.loc[:, col]
            if column.dtype != np.float64 and column.dtype != np.int64:
                no_num_cols.append(column.name)
        return no_num_cols
    
    # Рисует график между двумя конкретными столбцами
    def plot_cor(self, y_column, x_column):
        plt.figure(figsize=(12, 8))
        plt.scatter(self.df[x_column], self.df[y_column])
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()

    # Рисует график попарной коррелиции между иследуемым параметором и остальными столбцами
    def plot_cor_mul(self, y_column, n_col):  
        # Делит датасет на искомый столбец и факторы     
        Y = self.df[[y_column]]
        x = self.df.drop(y_column, axis=1)
        # Определяет кол-во строк 
        n_row = round(len(x.columns)/n_col)
        if n_row < 2:
            n_row = 2
        # создает n_col строк по n_row графиков в каждой 
        fig, ax = plt.subplots(n_row, n_col)
        j = 0
        i = 0
        for t in range(0, len(x.columns)):
            # достает t-ый столбец из датасета 
            x_col = x.iloc[:, t]
            # строит графики 
            print(j, i)
            ax[j, i].scatter(x_col, Y); ax[j, i].set_title(x_col.name)
            ax[j, i].set_title(x_col.name)
            i += 1
            if i == n_col:
                i = 0
                j += 1 

        plt.show()
    
    # базовый набор для чистки датасета от остуствуюищих элементов. Позволяет выбрать способ заполнения
    def data_clean(self, method='dn'):
        method.lower()
        # Удаляет все строки с NaN элементами
        if method == 'dropnone':
            self.df = self.df.dropna()
        # Заполняет все нолями
        if method == 'fillzero':
            self.df = self.df.fillna(0)
        # Заполняет все средним значением
        if method == 'fillmean':
            self.df = self.df.fillna(self.df.mean(), inplace=True)
        # Заполняет все медианой
        if method == 'fillmediana':
            self.df = self.df.fillna(self.df.median(), inplace=True)
        # Заполняет все модой``
        if method == 'fillmoda':
            self.df = self.df.fillna(self.df.mode()[0], inplace=True)
        
    # Обучает модель, делает предсказание и выводит метрики 
    def RandomForest(self, X_train, X_test, y_train, y_test):
        self.forest.fit(X_train, y_train)
        y_pred = self.forest.predict(X_test)
        self.__classification_metric__(y_test, y_pred)

    def Linearregression(self, X_train, X_test, y_train, y_test):
        print(self.df.corr())
        self.lime.fit(X_train, y_train)
        y_pred = self.lime.predict(X_test)
        plt.scatter(X_train, y_train)
        plt.plot(X_test, y_pred, 'r')
        plt.xlabel(X_train.iloc[:, 0].name)
        plt.ylabel(y_train.iloc[:, 0].name)
        plt.show()
        self.__regression_metric__(y_test, y_pred)



df = pd.read_csv(file)

df = BasicDataInteraction(df)
# df.short_cut()
df.plot_cor_mul('price', n_col=4)




# y_col = df[['price']]
# x_col = df[['square_feet']]
# X_train, X_test, y_train, y_test = train_test_split(x_col, y_col, test_size=.3, random_state=42)
# df.Linearregression(X_train, X_test, y_train, y_test)

        

  
