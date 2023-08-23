##################################################
# ENCODING ( Label Encoding, One-Hot Encoding, Rare Encoding)
##################################################
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder,StandardScaler , RobustScaler

########################################
# Label Encoding (Binary Encoding)
########################################

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()
df.head()

df["Sex"].head()

le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]

# hangisine 0 hangisine 1 verdiğimizi unutursak
le.inverse_transform([0, 1])  #array(['female', 'male']

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

df = load()

# verisetindeki tüm binary(2 sınıflı değişken) değişkenleri nasıl seçeceğim?
binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"]
               and df[col].nunique() == 2]

binary_cols   #['Sex']

for col in binary_cols:
    label_encoder(df, col)

df.head()


def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data

df = load_application_train()
df.shape    #(307511, 122)

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"]
               and df[col].nunique() == 2]
binary_cols
#['NAME_CONTRACT_TYPE',
#'FLAG_OWN_CAR',
#'FLAG_OWN_REALTY',
#'EMERGENCYSTATE_MODE']

df[binary_cols].head()

for col in binary_cols:
    label_encoder(df, col)
# label encoding den geçirince eksik değerleri de doldurur.


########################
# One-Hot Encoding
########################

df = load()
df.head()
df["Embarked"].value_counts()

pd.get_dummies(df, columns=["Embarked"]).head()

pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()

pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()

# get_dummies ile hem one-hot encoding, hem de binary encoding işlemini yapabiliyoruz
#get_dummies fonksiyonu sayesinde 2 sınıflı değişkenleri binary encoding e  de cevirebiliriz
pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True).head()


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = load()

# cat_cols, num_cols, cat_but_car = grab_col_names(df)
#tüm kolonlarda gez . değişkenlerin eşsiz sayısı 2 den büyük 10 dan kücükse bunlara one hot encoder uygula

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

one_hot_encoder(df, ohe_cols).head()

################################
# Rare Encoding
################################

#1-Kategorik değişkenlerin azlık-çokluk durumunun analiz edilmesi
#2-Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi
#3-Rare encoder yazacağız.

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()

# kategorik değişkneleri seçelim önce
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


# kategorik değişkenlerin sınıflarını ve bu sınıfların oranlarını getiren fonksiyon
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col)

# CODE_GENDER değişkeninde XNA sınıfı dışarıda bırakılmalıdır.
# NAME_INCOME_TYPE DEĞİŞKENİNDE son 4 sınıfı rare encoding uygulayabilriz.
# kategorik değişkenleri dönüştürmeden önce analiz ediyoruz.

#######################################################
# 2. ADIM: rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi
#######################################################

df["NAME_INCOME_TYPE"].value_counts()

df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()
# çıktıda 1'e yakın olmak kredi ödeyememeyi ifade eder, 0'a yakın olması kredi ödeyebilmeyi ifade eder.

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN" : dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "TARGET", cat_cols)
# mesela bu fonksiyonun çıktısı sonucu %1 altında olanları biraraya getirebiliriz.


#####################################
# 3. ADIM: Rare encoder'ın yazılması
#####################################
# fonksiyon önce girilen dataframe'in bir kopyasını alır.
# rare_columns kısmında fonksiyona girilen rare oranından daha düşük sayıda bu kategorik değişkenin sınıf oranı varsa bunları getiriri.

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O" and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis = None)]
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])
    return temp_df

new_df = rare_encoder(df, 0.01)

rare_analyser(new_df, "TARGET", cat_cols)


########################################################
# FEATURE SCALING (ÖZELLİK ÖLÇEKLENDİRME)
########################################################

# ilk amaç modellerin değişkenlere eşit şartlar altında yaklaşmasını sağlamaktır.
# özellikle gradient descent kullanan algoritmaların train sürelerini kısaltmak için kullanılır.
# uzaklık temelli yöntemlerde yanlılığın önüne geçmek.(knn)

################
# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
################

df = load()
ss = StandardScaler()

df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()

####################
# RobustScaler : Medyanı çıkar iqr'a böl. veri setindeki aykırı değerlere karşı daha dayanıklı.
####################

rs = RobustScaler()
df["Age_robust_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

#####################
# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
#####################

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T


age_cols = [col for col in df.columns if "Age" in col]

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in age_cols:
    num_summary(df, col, plot=True)



#########################
# NUmeric to Categorical: Sayısal Değişkenleri Kategorik Değişkenlere Çevirme
# Binning
#########################

df["Age_qcut"] = pd.qcut(df['Age'], 5)