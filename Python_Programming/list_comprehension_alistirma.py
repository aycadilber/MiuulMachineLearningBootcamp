#################################
# Görev 1: List Comprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin
# isimlerini büyük harfe çeviriniz ve başına NUM ekleyiniz.
# numeric olmayanların da isimleri büyümeli.
##################################

import seaborn as sns

df = sns.load_dataset("car_crashes")
df.columns

["NUM_" + col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns]


##################################
# Görev 2: List Comprehension yapısı kullanarak car_crashes verisinde isminde "no" barındırmayan
# değişkenlerin isimlerinin sonuna "FLAG" yazınız.
##################################


import seaborn as sns

df = sns.load_dataset("car_crashes")
df.columns

[col.upper() + "_FLAG" if "no" not in col else col.upper() for col in df.columns]


##################################
# Görev 3: List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden FARKLI olan değişkenlerin
# isimlerini seçiniz ve yeni bir dataframe oluşturunuz.
##################################

import seaborn as sns

df = sns.load_dataset("car_crashes")
df.columns

og_list = ["abbrev", "no_previous"]

new_cols = [col for col in df.columns if col not in og_list]
new_df = df[new_cols]
new_df.head()

