########################################
# Pandas Series
########################################

import pandas as pd

s = pd.Series([10, 77, 12, 4, 5])
type(s)

s.index #index bilgisi
s.dtype
s.size
s.ndim
s.values  #numpy array olarak döndürür.
s.head(3)
s.tail

# Veri Okuma (Reading Data)

df = pd.read_csv("datasets/advertising.csv")
df.head()

#########################
# Veriye Hızlı Bakış
#########################

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape  #(891, 15)
df.info()   #veri setinin bilgilerini verir.
df.columns  #değişken isimlerine erişmek için
df.describe().T    #veri setinin özet bilgilerini verir
df.isnull().values.any()
df.isnull().sum()   #boş değerlerinin sayısını verir.

df["sex"].head()
df["sex"].value_counts()

#########################################
# Pandas'ta Seçim İşlemleri
#########################################

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")

df[0:13]
df.drop(0, axis=0).head()  #0. indexi siler, axis=0 satır

delete_indexes = [1, 3, 5, 7]
df.drop(delete_indexes, axis=0).head(10)


###########################
# Değişkeni INdexe Çevirmek
###########################

df["age"].head()
df.age.head()

df.index = df["age"]

df.drop("age", axis=1).head()

df.drop("age", axis=1, inplace=True)
df.head()

###########################
# Indexi Değişkene Çevirmek
###########################

# 1. yol
df.index

df["age"] = df.index
df.head()

# 2. yol
df = df.reset_index()
df.head()

##################################
# Değişkenler Üzerinde İşlemler
##################################

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

"age" in df   # age değişkeni veri seti içinde var mı?

df["age"].head()
type(df["age"].head())
#pandas.core.series.Series  df["age] liste formatında alır

df[["age"]].head()
type(df[["age"]].head())
#pandas.core.frame.DataFrame


# bir dataframe içinden birden fazla değişken seçmek istersek
df[["age", "alive"]]

col_names = ["age", "adult_male", "alive"]
df[col_names]


# dataframe'e bir değişkne ekleme
df["age2"] = df["age"]**2
df["age3"] = df["age"] / df["age2"]

df.drop("age3", axis=1).head()

#birden fazla değişken silmek istiyorsak
col_names = ["age", "adult_male", "alive"]
df.drop(col_names, axis=1).head()

# diyelim ki veri setinde belirli bir string ifadeyi barındıran değişkenleri silmke istiyorum

df.loc[:, ~df.columns.str.contains("age")].head()


###############################
# iloc (integer based selection)
###############################

df.iloc[0:3]   # 0,1,2 gelir
df.iloc[0, 0]

df.iloc[0:3, 0:3]


###############################
# loc (label based selection)
###############################

df.loc[0:3]   #0,1,2,3 gelir

df.loc[0:3, "age"]

col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names]

######################################
# Koşullu Seçim (Conditional Selection)
######################################

# veri setinde yaşı 50'den büyük olan kaç kişi var?
df[df["age"] > 50].head()
df[df["age"] > 50]["age"].count()

# yaşı 50'den büyük olanların class bilgisine erişme

df.loc[df["age"] > 50, "class"].head()

df.loc[df["age"] > 50, ["age", "class"]].head()

df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["age", "class"]].head()

df.loc[(df["age"] > 50)
       & (df["sex"] == "male")
       & (df["embark_town"] == "Cherbourg"),
       ["age", "class", "embark_town"]].head()

df.loc[(df["age"] > 50)
       & (df["sex"] == "male")
       & ((df["embark_town"] == "Cherbourg") | (df["embark_town"] == "Southampton")),
       ["age", "class", "embark_town"]].head()


######################################################
# Toplulaştırma ve Gruplama (Aggregation & Grouping)
######################################################

#Toplulaştırma, veri setindeki verileri özetleme veya bir araya getirme işlemidir.
# - count()
# - first()
# - last()
# - mean()
# - median()
# - min()
# - max()
# - std()
# - var()
# - sum()
# - pivot table

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

df["age"].mean()

#cinsiyete göre yaş ortalaması
df.groupby("sex")["age"].mean()

df.groupby("sex").agg({"age": "mean"})

df.groupby("sex").agg({"age": ["mean", "sum"]})

#neden pivot table fonsiyonuna ihtiyacımız var?
df.groupby("sex").agg({"age": ["mean", "sum"],
                       "embark_town": "count"})

df.groupby("sex").agg({"age": ["mean", "sum"],
                       "survived": "mean"})

df.groupby("sex").agg({"age": ["mean", "sum"],
                       "survived": "mean"})

df.groupby(["sex", "embark_town"]).agg({"age": "mean",
                       "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({"age": "mean",
                       "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({
    "age": "mean",
    "survived": "mean",
    "sex": "count"})

#######################
# Pivot Table
#######################

#df.pivot_table( value, kırılım(satır), kırılım(sütun))
#value değerleri default mean olarak gelir

df.pivot_table("survived", "sex", "embarked", aggfunc="std")

df.pivot_table("survived", "sex", ["embarked", "class"])

# yaş değişkenş hayatta kalma açısından nasıl değerlendirilir?

df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])
# yaş değişkeni kategorik değişkene çevirmek için --> cut() veya qcut()

df.head()

df.pivot_table("survived", "sex", "new_age")

df.pivot_table("survived", "sex", ["new_age", "class"])

pd.set_option('display.width', 500)


##########################################
# Apply ve Lambda
##########################################
# DataFrame üzerinde apply kullanımı
#df.apply(func, axis=0)  # Sütunlara (kolonlara) uygulamak için axis=0
#df.apply(func, axis=1)  # Satırlara uygulamak için axis=1

# Series üzerinde apply kullanımı
#series.apply(func)

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")

df["age2"] = df["age"]*2
df["age3"] = df["age"]*5

df.head()

(df["age"]/10).head()
(df["age2"]/10).head()

for col in df.columns:
    if "age" in col:
        print((df[col]/10).head())

for col in df.columns:
    if "age" in col:
        df[col] = df[col]/10

#bu işlemi apply ve lmabda kullanarak yap
df[["age", "age2", "age3"]].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head()
#df.columns.str.contains("age") --> içinde age barındıranları seçer

df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std()).head()


def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

#kaydetmek için
df.loc[:, ["age", "age2", "age3"]] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)

df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)

df.head()

############################
# Birleştirme İşlemleri
############################
import numpy as np
import pandas as pd
m = np.random.randint(1, 30, size=(5, 3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99

# iki datatframe'i alt alta birleştirmek için
pd.concat([df1, df2])

pd.concat([df1, df2], ignore_index= True)

##################################
# Merge ile Birleştirme İşlemleri
##################################

df1 = pd.DataFrame({'employees': ['john', 'dennis', 'mark', 'maria'],
                    'group': ['accounting', 'engineering', 'engineering', 'hr']})

df2 = pd.DataFrame({'employees': ['mark', 'john', 'dennis', 'maria'],
                    'start_date': [2010, 2009, 2014, 2019]})


pd.merge(df1, df2)
pd.merge(df1, df2, on="employees")

# Her çalışanın müdürünün bilgisine erişmek istiyoruz.
df3 = pd.merge(df1, df2)

df4 = pd.DataFrame({'group': ['accounting', 'engineering', 'hr'],
                    'manager': ['Caner', 'Mustafa', 'Berkcan']})

pd.merge(df3, df4)
