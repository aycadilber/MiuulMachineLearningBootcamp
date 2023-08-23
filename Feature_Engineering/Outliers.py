#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################

# Verideki genel eğilimin oldukça dısına çıkan değerlere denir.
# Özellikle doğrusal problemlerde aykırı değerlerin etkileri daha şiddetlidir. Ağaç yöntemlerinde bu etkiler daha düşüktür.
# Aykırı Değerler Neye Göre Belirlenir?
   #Sektör Bilgisi
   # Standart Sapma Yaklaşımı
   # Z-Skoru Yaklaşımı
   # Boxplot(interquirtile range -IQR) Yaklaşımı (tek değişkneli) => En çok tercih edilen yaklaşımdır. (LOF yöntemi çok değişkenli)

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
pd.options.display.max_columns = None

def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data

df = load_application_train()
df.head()

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()
df.head()

#########################################
# 1. Outliers (Aykırı Değerler)
#########################################

################################
# Aykırı Değerleri Yakalama
################################

sns.boxplot(x=df["Age"])
plt.show()

####################################
# Aykırı Değeerler Nasıl Yakalanır?
####################################

q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)
iqr = q3 - q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

df[(df["Age"] < low) | (df["Age"] > up)].index

# hızlı şekilde aykırı değer var mı yok mu diye sormak için

df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)

df[~((df["Age"] < low) | (df["Age"] > up))].any(axis=None)

#### FONKSİYONLAŞTIRMA

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

low, up = outlier_thresholds(df, "Age")

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Age")


###########################
# grab_col_names fonksiyonu
###########################
dff = load_application_train()
dff.head()


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: catCols + numCols + catButCar = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                 dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                 dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'catCols: {len(cat_cols)}')
    print(f'numCols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'numButCat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col, check_outlier(df, col))

# diğer veri seti için deneyelim
cat_cols, num_cols, catButCar = grab_col_names(dff)

num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]

for col in num_cols:
    print(col, check_outlier(dff, col))


#######################################
# Aykırı Değerlerin Kendilerine Erişmek
#######################################

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        #shape[0] gözlem sayısını getirir, shape[1] değişken sayısını getirir.
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


grab_outliers(df, "Age")

age_index = grab_outliers(df, "Age", True)

########################################
# AYKIRI DEĞER PROBLEMİNİ ÇÖZME
########################################

#############
# SİLME
#############

low, up = outlier_thresholds(df, "Fare")
df.shape   #(891, 12)

# aykırı değerler silindikten sonra kaç tane gözlem kalacak?
df[~((df["Fare"] < low) | (df["Fare"] > up))].shape  #(775, 12)

# elimizde birden fazla değişkne varsa nasıl sileriz?

# Aykırı olmayan gözlemleri dönen fonksiyon
def remove_outlier(dataframe,col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
df.shape

for col in num_cols:
    new_df = remove_outlier(df, col)

df.shape[0] - new_df.shape[0]          # aykrılıklardan kurtulduk

##############################################
# Baskılama Yöntemi (re-assigment with thresholds)
##############################################
# Bir aykırı gözlem silerken tam olan verilerden de oluyoruz.
# Bunun yerine bazı senaryolarda silmek yerine baskılama yöntemini seçebiliriz.

low, up = outlier_thresholds(df, "Fare")

df[((df["Fare"] < low) | (df["Fare"] > up))]["Fare"]
#bu işlemi loc ile de yapabiliriz
df.loc[((df["Fare"] < low) | (df["Fare"] > up)), "Fare"]

# Üst sınıra göre aykırı olan değerleri üst değere baskılayalım.
df.loc[(df["Fare"] > up), "Fare"] = up
#örneğin benim up değerim 100 idi, 100'ün üzerindeki değerleri bulup 100 yazdık.

df.loc[(df["Fare"] < low), "Fare"] = low

# fonksiyon kullanalım
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape   #(891, 12)

for col in num_cols:
    print(col, check_outlier(df, col))
#Age True
#Fare True

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))
#Age False
#Fare False

############################
# Recap
############################

df = load()
outlier_thresholds(df, "Age")  #öncelikle aykırı değeri saptadık
check_outlier(df, "Age")       # outlier var mı yok mu?
grab_outliers(df, "Age", index=True)   # outlierları bize getir dedik

remove_outlier(df, "Age").shape   #(880, 12) outlier'ları sildik.
replace_with_thresholds(df, "Age")   # baskılama yöntemi kullan dedik
check_outlier(df, "Age")  #False




###########################################################
# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor
###########################################################
#Tek başına aykırı değer olmayıp, bir başka değişken ile ele alınınca aykırı olan değişkenlere çok değişkenli aykırı değer denir.
#3 kere evlenmek aykırı bir değer olmayabilir fakat 17 yasında olup 3 kere evlenmek aykırıdır.
#  LOF YÖNTEMİ-->Çok değişkenli aykırı değer belirleme yöntemidir. Gözlemleri, bulundukları konumda yoğunluk tabanlı
# skorlayarak buna göre aykırı değer tanımı yapabilmemizi sağlar.'Belirli bir örneklemin yoğunluğunun komşularına göre
# yerel sapmasını ölçen' Bir noktanın lokal yoğunlugu demek, ilgili noktanın
# etrafındaki komşuluklar demektir. Eğer bir nokta komşuluklarının yoğunlugundan anlamlı bir şekilde düşük ise bu nokta
# daha seyrek bir bölgededir demek ki bu nokta aykırı değer olabilir yorumu yapILIR.

#Loft yöntemi bize bu komsuluklara göre uzaklık skorları hesaplamımıza sağlar.

df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
df.head()

#  bu dataframede aykırı değer var mı?
for col in df.columns:
    print(col, check_outlier(df, col))

low, up = outlier_thresholds(df, "carat")

#Bu carat değişkeninde kaç tane outlier var? --> shape
df[((df["carat"] < low) | (df["carat"] > up))].shape
# (1889, 7) 1889 tane aykırı değer var.Bu verisetinde 53940 gözlem var.

low, up = outlier_thresholds(df, "depth")
df[((df["depth"] < low) | (df["depth"] > up))].shape
# 2545 tane outlier var. (2545, 7)

# Outlier threshold 25 'e 75 'likti literatürde.
# Eğer konuya tek bir değişken üzerinden giderek 25 e 75 lik outlier silseydik bu sefer ciddi veri kaybı olacaktı.
# Silmeseydik ise gürültü eklemiş olacaktık.
# Ağaç yöntemleri kullanıyorsanız hiç dokunmamayı tercih etmelisiniz.

# çok değişkenli yakalaşalım.
# yöntemi getirdik
clf = LocalOutlierFactor(n_neighbors=20)
# local outlier factor skorlarını getirelim
clf.fit_predict(df)
# bu skorları tutmak için
df_scores = clf.negative_outlier_factor_

df_scores[0:5]  #array([-1.58352526, -1.59732899, -1.62278873, -1.33002541, -1.30712521])
# skorları eksi değerlerle değrlendirmek istemiyorsan df_scores = -df_scores

# en kötü 5 gözleme bakalım
np.sort(df_scores)[0:5]  #array([-8.60430658, -8.20889984, -5.86084355, -4.98415175, -4.81502092])


# eşik değer ne olmalı? en dik eğim değişikliğini eşik değeri olarak belirleyebilriz.
# grafiğe göre 3. indeksteki değeri seçebiliriz.
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style=".-")
plt.show()

th = np.sort(df_scores)[3]
#-4.984151747711709  bu değerden küçük olanlar aykırı değrlerdir.

df[df_scores < th]
df[df_scores < th].shape   #(3, 7) 3 tane aykırı değer çıktı.

# Peki bunlar acaba neden aykırı? Çok değişkneli olunca aykırı değer sayısı olduça aza indi neden?
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T
# bu tabloda 41918 satırının depth max değeri 79, ama radarımıza takılan depth değeri 78.2 max'a yakın olduğu için yakalanmış olabilir.
# 48410. satırda z değeri 31800, bu tabloda z max değeri 31800. bundan dolayı aykırı olmuş.

# aykırı değerleri silelim
df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

# eğer gözlem sayısı çok fazlaysa baskılama yöntmei ciddi problemlere sebep olabilri.
# eğer ağaç yöntemleriyle çalışıyorsak aykırı değerlere hiç dokunmamayı tercih etmeliyiz.

df["depth"].quantile(0.25)