###########################################
# MISSING VALUES (EKSİK DEĞERLER)
###########################################

"""
Eksik Veri problemi nasıl çözülür ?

Silme
Değer Atama Yöntemleri (ortalama , medyan gibi değerleri atamak)
Tahmine Dayalı Yöntemler
Eksik veri ile çalısırken göz önünde bulundurulması gereken önemli konulardan birisi :
Eksik verinin rassalığıdır.Eksikliğin rastgele ortaya cıkıp cıkmadıgı durumudur.

Eğer eksiklikler rastgele ortaya cıktıysa rahatız silebiliriz , rastgele ortaya cıkmadıysa rahat değiliz.
Bu eksikliğin neyden ötürü cıktıgını tespit etmemiz gerekir.
Diğer değişkenler ile bağımlılıgını bulup çözmekle uğraşmalıyız.

"""
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


##############################
# Eksik Değerleirn Yakalanması
##############################

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()
df.head()

# eksik gözlem var mı  yok mu?
df.isnull().values.any()    #True

# değişkenlerdeki eksik değer sayısı?
df.isnull().sum()

# değişknelerdeki tam değer sayısı?
df.notnull().sum()

# veri setindeki toplam eksik değer sayısı
df.isnull().sum().sum()

# en az 1 tane eksik değere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]

# azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending=False)

# Eksikliğin veri seti içindeki oranını bulalım dersek;
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

# Sadece eksik değere sahip değişkenlerin isimlerini seçebiliriz.
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
na_cols

# Fonksiyonlaştırırsak   na_name true dersek değişken isimleri gelir
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)
missing_values_table(df, True)

#########################################
# Eksik Değer Problemini Çözme
#########################################

# Ağaç yöntemleri, doğrusal yöntemlerde yada optimizasyona dayalı yöntemlerde oldugu gibi değilde daha esnek ve dallara
# ayırmalı bir şekilde çalısıyor oldugundan dolayı bu noktadaki aykırılıklar ve eksikliklerin etkisi neredeyse yoka yakındır. Göz ardı edilebilir.

################
# Çözüm 1: Hızlıca Silmek
################

df.dropna().shape

##############################
# Çözüm 2: Basit Atama Yöntemleriyle Doldurmak
##############################

df["Age"].fillna(df["Age"].mean()).isnull().sum()
df["Age"].fillna(df["Age"].median()).isnull().sum()
df["Age"].fillna(0).isnull().sum() #sabit bir değer verebiliriz.


# Öyle bir işlem yapalım ki sadece sayısal değişkenleri mean ile doldur, kategorik değişkenler normal kalsın
dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()
# axis = 0 dememizin sebebi; satırlara göre gitmesini istiyorum. Satırlara göre gittiğinde sütun bazında aslında konuya bakıyoruz.
# Amacım aşağıya doğru bütün satırlara bakmak bu nedenle axis =0 diyoruz.

dff.isnull().sum().sort_values(ascending=False)

# Peki kategorik değişkenleri nasıl dolduracağız? Kategorik değişkenlerdeki eksikliklerin gidermenin yolu modunu almaktır.
df["Embarked"].mode()[0]
# # KAtegorik değişkenler için eksik değerleri doldurmak istersek modunu almalıyız
df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()

# boş değerleri missing ismi ile doldur diyebiliriz
df["Embarked"].fillna("missing")

# otomatik olarak eksik kategorik değişkenleri doldurmak
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()


#################################################
# Kategorik Değişken Kırılımında Değer Atama
#################################################

df.groupby("Sex")["Age"].mean()

df["Age"].mean()

# Cinsiyete göre verisetini grupla.Age değişkenini al.Bu age değişkeninin ortalamasını aynı groupby kırılımında gerekli yere yaz
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

# Loc yöntemi ile yapalım
df.groupby("Sex")["Age"].mean()["female"]  #kadınların yaş ortalaması

df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]
df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

df.isnull().sum()


##############################################
# Çözüm 3: Tahmine Dayalı Atama ile Doldurma
##############################################

#Bir makine öğrenmesi yöntemi ile tahmine dayalı bir şekilde modelleme işlemi gerçekleştireceğiz. Eksikliğe sahip
# değişkeni bağımlı değişken, diğer degişkenleri bağımsız değişken olarak kabul ederek bir modelleme işlemi gerçekleştireceğiz.

# 1- kategorik değişkenleri One hot encoding'e sokmamız lazım.
# 2- knn uzaklık temalı bir algoritma olduğundan dolayı değişkenleri standartlaştırmamız lazım.

df = load()

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

cat_cols, num_cols, cat_but_car = grab_col_names(dff)
num_cols = [col for col in num_cols if col not in "PassengerId"]

# one hot encoding ve label encoding işlemini aynı anda yapabilmek için get_dummies metodunu kullanırız.
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

dff.head()

# değişknelerin standartlaştırılması
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

# knn'in uygulanması
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

# doldurduğumuz değerleri görmek istiyoruz akat onlar standartlaştırılmış durumda olduğu için bunu geri alammız lazım
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

df["age_imputed_knn"] = dff[["Age"]]
# makine öğrenmesi tekniği ile doldurmus oldugumuz yeni yaş değerlerini eski dataframe içerisine kolon olarak ekliyorum
df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]

df.loc[df["Age"].isnull()]

##########################################
# Eksik Verinin Yapısını İncelemek
##########################################

# bar: verisetindeki tam olan gözlemlerin sayısını vermektedir
msno.bar(df)
plt.show()

# matrix: Değişkenlerdeki eksikliklerin birlikte cıkıp cıkmadıgı bilgisini verir
msno.matrix(df)
plt.show()

#heatmap : ısı haritası , eksiklikler birlikte mi cıkıyor bağımlıgınını anlayabilmek için kullanırız
#pozitif yönde korelasyon değişkenlerdeki eksikliklerin birlikte ortaya cıktıgı düşünülür .
msno.heatmap(df)

###################################
# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
###################################

missing_values_table(df, True)
na_cols = missing_values_table(df, True)

# target: bağımlı değişken
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy() # girilen dataframe 'in kopyasını olustur

    for col in na_columns:  # eksik değerlere sahip değişkenlerde gez, yedek dataframe içerisinde bu değişkenleri yeni isimlendirme ile olustur. NA olan yere 1 yoksa 0 olarak atama işlemi yap.
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns  # bu değişkenlerin isimlerini al

    for col in na_flags:  # bu na bulunduran kolonlarda gez
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "Survived", na_cols)