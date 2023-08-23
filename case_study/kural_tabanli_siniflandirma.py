#############################################
# Veri Seti Hikayesi
#############################################
# Persona.csv veri seti uluslararası bir oyun şirketinin sattığı ürünlerin fiyatlarını ve bu ürünleri satın alan kullanıcıların bazı
# demografik bilgilerini barındırmaktadır. Veri seti her satış işleminde oluşan kayıtlardan meydana gelmektedir. Bunun anlamı tablo
# tekilleştirilmemiştir. Diğer bir ifade ile belirli demografik özelliklere sahip bir kullanıcı birden fazla alışveriş yapmış olabilir.

# Price: Müşterinin harcama tutarı
# Source: Müşterinin bağlandığı cihaz türü
# Sex: Müşterinin cinsiyeti
# Country: Müşterinin ülkesi
# Age: Müşterinin yaşı

################# Uygulama Öncesi #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# Uygulama Sonrası #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C


import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv('../datasets/persona.csv')

df.head()
df.shape    #(5000, 5)
df.info()
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df.dtypes

df["SOURCE"].nunique()
df["SOURCE"].value_counts()

df["PRICE"].nunique()
df["PRICE"].value_counts()

df["COUNTRY"].value_counts()

df.groupby("COUNTRY").agg({"PRICE": "sum"})

df["SOURCE"].value_counts()

df.groupby("COUNTRY").agg({"PRICE": "mean"})

df.groupby("SOURCE").agg({"PRICE": "mean"})

df.groupby(["SOURCE", "COUNTRY"]).agg({"PRICE": "mean"})

# En çok satılan ürünleri bulma
most_sold_products = df['SOURCE'].value_counts().nlargest(5)

# Görselleştirme
plt.bar(most_sold_products.index, most_sold_products.values)
plt.xlabel('Ürün Kaynağı (Source)')
plt.ylabel('Satış Sayısı')
plt.title('En Çok Satılan Ürünler')
plt.show()


agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})
agg_df.sort_values("PRICE", ascending=False)


agg_df.index
agg_df = agg_df.reset_index()
agg_df.head()
agg_df.columns

# age değikeni nerelerden bölüneceği belirtmek için
age_bins = [0, 18, 23, 30, 40, agg_df["AGE"].max()]
# bölünen noktalara karşılık isimlendirmelerin ne oalcağını belirtmek için
age_labels = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())]
#age_labels = ['0_18', '19_23', '24_30', '31_40', f'41_{agg_df["AGE"].max()}']

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins=age_bins, labels=age_labels)
agg_df.head()

#customer_level_based adında değişken tanımla --> örneğin USA_ANDROİD_MALE_0_18 bu şekilde gözükmeli

agg_df.drop(["AGE", "PRICE"], axis=1).values
"""
array([['bra', 'android', 'female', '0_18'],
       ['bra', 'android', 'female', '0_18'],
       ['bra', 'android', 'female', '0_18'],
"""

liste = ["A", "B", "C"]
"_".join(liste)

agg_df["CUSTOMERS_LEVEL_BASED"] = ["_".join(i).upper() for i in agg_df.drop(["AGE", "PRICE"], axis=1).values]

"""
['BRA_ANDROID_FEMALE_0_18',
 'BRA_ANDROID_FEMALE_0_18',
 'BRA_ANDROID_FEMALE_0_18',
 'BRA_ANDROID_FEMALE_0_18',
 'BRA_ANDROID_FEMALE_19_23',
 'BRA_ANDROID_FEMALE_19_23'
"""
agg_df.head()
agg_df = agg_df[["CUSTOMERS_LEVEL_BASED", "PRICE"]]

"""
      CUSTOMERS_LEVEL_BASED      PRICE
0   BRA_ANDROID_FEMALE_0_18  38.714286
1   BRA_ANDROID_FEMALE_0_18  35.944444
2   BRA_ANDROID_FEMALE_0_18  35.666667
3   BRA_ANDROID_FEMALE_0_18  32.255814
4  BRA_ANDROID_FEMALE_19_23  35.206897
"""# customer_level_based değişkeninde tekrar eden değerler var bu çoklama probleminden kurtulmak için
agg_df = agg_df.groupby("CUSTOMERS_LEVEL_BASED")["PRICE"].mean().reset_index()

#PRICE'A göre segmentlere ayıralım
agg_df['SEGMENT'] = pd.qcut(agg_df["PRICE"], q=4, labels=['D', 'C', 'B', 'A'])
agg_df.head()

agg_df.groupby('SEGMENT').agg({'PRICE': ['mean', 'sum', 'min', 'max']})

#33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
new_user = 'TUR_ANDROID_FEMALE_31_40'
agg_df[agg_df['CUSTOMERS_LEVEL_BASED'] == new_user]
"""
CUSTOMERS_LEVEL_BASED      PRICE SEGMENT
72  TUR_ANDROID_FEMALE_31_40  41.833333       A
"""

#35 yaşında IOS kullanan bir Fransız kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
new_user = 'FRA_IOS_FEMALE_31_40'
agg_df[agg_df['CUSTOMERS_LEVEL_BASED'] == new_user]

"""
   CUSTOMERS_LEVEL_BASED      PRICE SEGMENT
63  FRA_IOS_FEMALE_31_40  32.818182       C
"""

d_segment_df = agg_df[agg_df["SEGMENT"] == "D"]

# Filtrelenen veriler içinde "Android" ve "iOS" kullanım sayılarını bulun
android_count = d_segment_df["CUSTOMERS_LEVEL_BASED"].str.contains("ANDROID").sum()
ios_count = d_segment_df["CUSTOMERS_LEVEL_BASED"].str.contains("IOS").sum()

print(f"D segmentine ait Android kullanım sayısı: {android_count}")
print(f"D segmentine ait iOS kullanım sayısı: {ios_count}")

agg_df["SEGMENT"].value_counts()
agg_df.shape

agg_df["SEGMENT"].values


def func(df, segment):
    segment_df = df[df["SEGMENT"] == segment]

    android_count = segment_df["CUSTOMERS_LEVEL_BASED"].str.contains("ANDROID").sum()
    ios_count = segment_df["CUSTOMERS_LEVEL_BASED"].str.contains("IOS").sum()

    return android_count, ios_count

segments = ["A", "B", "C", "D"]
for segment in segments:
    android_count, ios_count = func(agg_df, segment)
    print(f"{segment} segmentine ait Android kullanım sayısı: {android_count}")
    print(f"{segment} segmentine ait iOS kullanım sayısı: {ios_count}")

