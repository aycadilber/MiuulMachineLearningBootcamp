###############################################################
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
#############################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")

cat_cols = [col for col in df.columns if str(df[col].dtype) in ['object', 'category', "bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
cat_cols = cat_cols + num_but_cat


cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[["age", "fare"]].describe().T

# veri seti içinden nümerik değişkenleri nasıl seçeriz?
num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]

#num_cols içinde olup cat_cols'ta olmayan değişkenleri seç
num_cols = [col for col in num_cols if col not in cat_cols]

# fonksiyon

def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

num_summary(df, "age")

for col in num_cols:
    num_summary(df, col)

# fonksiyona grafik özelliği ekleyelim.

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

num_summary(df, "age", plot=True)


###########################################################
# Değişkenlerin Yakalanması ve İşlemlerin Genelleştirilmesi
###########################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.info()

# kategorik değişkenleri ve nümerik değişkenleri, kategorik ama kardinal olanları bize ayrı ayrı getirsin.
#bir kategorik değişken eğer eşsiz değer sayısı 20'den büyükse kardinal diyoruz. car_th=20
#bir değişken sayısal olsa dahi eşsiz değer sayısı 10'dan küçükse kategorik değişkendir diyeceğiz.

def grab_col_names(dataframe, cat_th=10, car_th=20):
        """
        Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

        Parameters
        ----------
        dataframe: dataframe
            değişken isimleri alınmak istenen dataframe'dir.
        cat_th: int, float
            numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, float
            kategorik fakat kardinal değişkenler için sınıf eşik değeri

        Returns
        -------
        cat_cols: list
            Kategorik değişken listesi
        num_cols: list
            Numerik değişken listesi
        cat_but_car: list
            Kategorik görünümlü kardinal değişken listesi

        Notes
        ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

        """

        cat_cols = [col for col in df.columns if str(df[col].dtype) in ['object', 'category', "bool"]]
        num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "int32", "float64"]]
        cat_but_car = [col for col in df.columns if
                       df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
        cat_cols = cat_cols + num_but_cat

        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
        num_cols = [col for col in num_cols if col not in cat_cols]

        print(f"Observations: {dataframe.shape[0]}")
        print(f"Variables: {dataframe.shape[1]}")
        print(f'cat_cols: {len(cat_cols)}')
        print(f'num_cols: {len(num_cols)}')
        print(f'cat_but_car: {len(cat_but_car)}')
        print(f'num_but_cat: {len(num_but_cat)}')

        return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")


cat_summary(df, "sex")
for col in cat_cols:
    cat_summary(df, col)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)


# BONUS : veri setindeki bool tipleri int yapmak ve görselleştirmek

df = sns.load_dataset("titanic")
df.info()
for col in df.columns:
    if df[col].dtypes == 'bool':
        df[col] = df[col].astype(int)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)







