import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
import random
import statsmodels.api as sm
from scipy.stats import shapiro
import collections

os.chdir("C:/Users/Ökans/Desktop/Bootcamp")

class information():

    # dosyanın okunması
    def __init__(self, dosya):
        self.data = pd.read_csv(dosya)

    # data ile ilgili bilgi
    def infos(self):
        print("--> DATA INFO <--")
        print("\n")
        print(self.data.info())
        print("\n")
        print("--> DATA DESCRIBE <--")
        print("\n")
        print(self.data.describe())
        print("\n")
        print("--> DATA CORRELATION <--")
        print("\n")
        print(self.data.corr())

def listToString(s):
    str1 = ""
    for ele in s:
        str1 += ele
    return str1

class dataPreprocessing():

    def __init__(self, data):
        if(listToString(data[-3:])=="csv"):
            self.data = pd.read_csv(data)

        if (listToString(data[-4:]) == "xlsx"):
            self.data = pd.read_excel(data)


    # datanın NaN değerlerden temizlenmesi
    def cleaner(self, how):
        self.data = self.data.dropna(how=how)

    def dropColumn(self, column):
        self.data = self.data.drop(column, axis=1)

    def toDataFrame(self):
        return pd.DataFrame(self.data)

    def numeric_variables(self):
        numeric_variables = []
        for i in self.data.columns:
            if self.data[i].dtypes != np.object:
                numeric_variables.append(i)
        return numeric_variables

    def categoric_variables(self):
        categoric_variables = []
        for i in self.data.columns:
            if self.data[i].dtypes == np.object:
                categoric_variables.append(i)
        return categoric_variables

    # tekrar eden gereksiz değerlerin temizlenmesi
    def dropDuplicates(self):
        self.data = self.data.drop_duplicates(keep=False, inplace=True)

    # datanın kolon bazlı birleştirilmesi
    # how:{‘left’, ‘right’, ‘outer’, ‘inner’}
    def concatenation(self, merge_data, how, left_on, right_on):
        return self.data.merge(merge_data, how=how, left_on=left_on, right_on=right_on)

    def dummy(self, categoric_variables):
        return pd.get_dummies(self.data[categoric_variables])

    def preparingDataToPCA(self, dms, target, drop_variables, dummy_variables):
        y = self.data[target]
        X_ = self.data.drop(drop_variables, axis=1).astype("float64")
        X = pd.concat([X_, dms[dummy_variables]], axis=1)
        return X, y

    # crosstab tablosunun index ve column olarak sunulması ve index isimlerin atanması
    def crosstabing(index, columns, normalize, *args):
        index_name = []
        tab = pd.crosstab(index=index, columns=columns, normalize=normalize)
        for arg in args:
            index_name.append(arg)
        tab.index = index_name
        return tab


class Test():

    def __init__(self, data):
        self.data = data

    def shapiro(self, column, alpha):
        stat, p = shapiro(self.data[column])
        if p > alpha:
            print('Örneklem Normal (Gaussian) Dağılımdan gelmektedir (Fail to Reject H0)')
        else:
            print('Örneklem Normal (Gaussian) Dağılımdan gelmemektedir (reject H0)')


class visualization():

    def __init__(self):
        print(self)

    # datanın factorplot olarak görselleştrilmesi
    def factorPlot(column1, column2, hue, data):
        sns.factorplot(column1, column2, hue=hue, data=data)
        plt.show()

    # datanın barplot olarak görselleştirilmesi
    def barPlot(x, y, hue, data):
        sns.barplot(x=x, y=y, hue=hue, data=data)
        plt.show()

# randomly select columns from numeric_variables and use it in OLS analysis.
def random_select(variables, n):
    # n:how many columns in analysis
    j = len(variables) - 1
    column_list = []
    random_list = []
    for i in range(n):
        c = random.randint(0, j)
        if random_list.count(c) > 0:  # duplicate columns dropping in here
            continue
        else:
            random_list.append(c)
            column_list.append(variables[c])
    return column_list



# data : dataframe that are used in analysis
# variables : to use column in OLS test
# columns_range : number of Columns in OLS test
# target : target variable in data

# randomly select columns from variables and use it OLS analysis and keep best solutions in list with variables
# name. then this variables calculate with counter and most common variables select.
# this best variables about OLS analysis are used in another OLS analysis until found best solution.
# best solution param is rsquared_adj score in here.
def OLS_analysis(data, variables, columns_range, target):
    columnsToAnalysis = []
    length_columns = len(variables)
    for i in range(5000):
        for j in range(2, columns_range):
            column_list = random_select(variables, j)
            X = data[column_list]
            stat, p = shapiro(data[column_list])
            if p < 0.05:
                X = sm.add_constant(X)

                y = data[target]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

                ols_model = sm.OLS(y_train, X_train)
                model = ols_model.fit()
                if model.rsquared_adj > 0.005:
                    print(column_list)
                    print(model.params)
                    print("r2: " + "%.5f" % model.rsquared_adj)
                    print("------------------")

                    df = pd.DataFrame([model.params.index, model.pvalues]).T
                    for indis, value in df.itertuples(index=False):
                        if value < 0.05:
                            columnsToAnalysis.append(indis)
    return columnsToAnalysis

#returns most common colums in OLS analysis
def most_common_columns(column_analysis_list, nth):
    counter = collections.Counter(column_analysis_list)
    good_anlaysis = []
    for i in range(nth):
        good_anlaysis.append(counter.most_common(nth)[i][0])
    return good_anlaysis


''' WORKİNG STUFF '''

# dataPreprocessing = dataPreprocessing("hitters.csv")
# dataPreprocessing.cleaner('any')
# numeric_variables = dataPreprocessing.numeric_variables()
# categoric_variables = dataPreprocessing.categoric_variables()
# dummy_data = dataPreprocessing.dummy(categoric_variables)
# print(categoric_variables)
# print("\n")
# print(dummy_data)
# X, y = dataPreprocessing.preparingDataToPCA(dummy_data,"Salary", ["Salary", "League", "Division", "NewLeague"],
#                                         ["League_N", "Division_W", "NewLeague_N"])
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# print("X_train",X_train.shape)
# print("y_train",y_train.shape)
# print("X_test",X_test.shape)
# print("y_test",y_test.shape)
# training = dataPreprocessing.data.copy()
# print("training", training.shape)

datapreprocessing = dataPreprocessing("dataset.xlsx")
datapreprocessing.cleaner('any')
# numeric_variables = datapreprocessing.numeric_variables()
# categoric_variables = datapreprocessing.categoric_variables()
#
# print(categoric_variables)
# print("-------")
# print(numeric_variables)
#
# data = dataPreprocessing.toDataFrame()

# col = OLS_analysis(datapreprocessing.toDataFrame(), numeric_variables, 5, "90_target")

# import pandas_summary as ps
# dfs = ps.DataFrameSummary(datapreprocessing.toDataFrame())
# print(dfs.columns_stats)
