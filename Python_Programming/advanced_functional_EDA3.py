#########################################################
# 4. Hedef Değişken Analizi( Analysis of Target VAriable)
#########################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

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
        num_but_cat = [col for col in df.columns if
                       df[col].nunique() < 10 and df[col].dtypes in ["int64", "int32", "float64"]]
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

# survived değişkenini analiz etmek istiyoruz
# insanların hayatta kalma durumunu etkileyen şey nedir?

df["survived"].value_counts()
cat_summary(df, "survived")

#####################################################
# Hedef Değişkenin Kategorik Değişkenler ile Analizi
#####################################################

# cinsiyete göre groupby alalım

df.groupby("sex")["survived"].mean()
"""
sex
female    0.742038
male      0.188908   kadınlar %74 hayatta kalmış, erkekler %18
"""

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

target_summary_with_cat(df, "survived", "pclass")

#bütün kategorik değişkenlerle target'ın durumunu inceleyelim.
for col in cat_cols:
    target_summary_with_cat(df, "survived", col)


#####################################################
# Hedef Değişkenin Sayısal Değişkenler ile Analizi
#####################################################

df.groupby("survived")["age"].mean()
"""
survived
0    30.626179  hayatta kalamayanların yaş ortalaması 30
1    28.343690  hayatta kalanların yaş ortalaması 28
"""

df.groupby("survived").agg({"age": "mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

target_summary_with_num(df, "survived", "age")

for col in num_cols:
    target_summary_with_num(df, "survived", col)


#####################################################
# 5.Korelasyon Analizi
#####################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("datasets/breast_cancer.csv")
df = df.iloc[:, 1:-1]
df.head()

num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]

corr = df[num_cols].corr()
# korelasyon değişkenlerin birbiriyle olan ilişkisini ifade eder
# -1 veya +1'e yaklaştıkça ilişkinin şiddeti kuvvetlenir

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

###############################################
# Yüksek Korelasyonlu Değişkenlerin Silinmesi
###############################################

cor_matrix = df.corr().abs()

#           0         1         2         3
# 0  1.000000  0.117570  0.871754  0.817941
# 1  0.117570  1.000000  0.428440  0.366126
# 2  0.871754  0.428440  1.000000  0.962865
# 3  0.817941  0.366126  0.962865  1.000000

# yukarıdaki matriste gereksiz elemanları silip aşağıdaki matrisi elde edelim.
#     0        1         2         3
# 0 NaN  0.11757  0.871754  0.817941
# 1 NaN      NaN  0.428440  0.366126
# 2 NaN      NaN       NaN  0.962865
# 3 NaN      NaN       NaN       NaN


#(np.ones(cor_matrix.shape), k=1)  1'lerden oluşan oluşturduğumuz olduğumuz matris boyutunda numpy array'i oluşturuyoruz.
# yukarıda görmüş olduğumuz yapıya çevirmek için np.trio() fonkdiyonunu kullanırız.
# where() --> içindeki koşulu sağlayanları getirir
upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))


# upper_triangle_matrixte sütunlardaki elemanlardan %90'dan büyükse sil
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]

cor_matrix[drop_list]
df.drop(drop_list, axis=1)

# fonksiyon
def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()   #mutlak değere aldık
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

high_correlated_cols(df)

drop_list = high_correlated_cols(df, plot=True)
df.drop(drop_list, axis=1)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)