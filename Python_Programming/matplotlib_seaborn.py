###########################################
# VERİ GÖRSELLEŞTİRME: MATPLOTLIB & SEABORN
###########################################

######################
# MATPLOTLIB
######################

# Kategorik değişken varsa --> sütun grafik. countplot bar
# Sayısal değişken varsa --> hist, boxplot

#######################################
# Kategorik Değişken Görselleştirme
#######################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("datasets/advertising.csv")
df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts().plot(kind="bar")
plt.show()

#######################################
# Sayısal Değişken Görselleştirme
#######################################

plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show()

#######################################
# Matplotlib Özellikleri
#######################################

#########
#plot
#########

x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y)
plt.show()

plt.plot(x, y, 'o')
plt.show()

x = np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])
plt.plot(x, y)
plt.show()

plt.plot(x, y, "o")
plt.show()

##################
# marker
##################

y = np.array([13, 28, 11, 100])

plt.plot(y, marker='o')  # Grafikteki her kırılım noktasına belirlediğimiz şekli koyar.
plt.show()

plt.plot(y, marker="*")
plt.show()

markers = ['o', '*', '.', ',', 'x', 'X', '+', 'P', 's', 'D', 'd', 'p', 'H', 'h']

################
# line
################
y = np.array([13, 28, 11, 100])
plt.plot(y, linestyle="dashdot", color="r")
#linestyle grafiğin çizgi tipini belirler. color ile renk belirlenir.
plt.show()

###################
# Multiple Lines
###################
x = np.array([23, 18, 31, 10])
y = np.array([13, 28, 11, 100])
plt.plot(x)
plt.plot(y)
plt.show()

#############
# Labels
#############
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.plot(x, y)

# Başlık
plt.title("Bu ana başlık")

# X eksenini isimlendirme
plt.xlabel("X ekseni isimlendirmesi")

plt.ylabel("Y ekseni isimlendirmesi")

plt.grid()  #arkaplanını gösterir.
plt.show()

############################
#Subplots (Çoklu Grafik)
############################
# plot 1
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 2, 1)  # 1 satırlı 2 sütunluk bir grafik oluştur.
plt.title("1")
plt.plot(x, y)

# plot 2
x = np.array([8, 8, 9, 9, 10, 15, 11, 15, 12, 15])
y = np.array([24, 20, 26, 27, 280, 29, 30, 30, 30, 30])
plt.subplot(1, 2, 2)  # yine 1'e 2'lik grafik oluşturuyorum bu 2.si
plt.title("2")
plt.plot(x, y)
plt.show()

# 3 grafiği bir satır 3 sütun olarak konumlamak.
# plot 1
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 3, 1)
plt.title("1")
plt.plot(x, y)

# plot 2
x = np.array([8, 8, 9, 9, 10, 15, 11, 15, 12, 15])
y = np.array([24, 20, 26, 27, 280, 29, 30, 30, 30, 30])
plt.subplot(1, 3, 2)
plt.title("2")
plt.plot(x, y)

# plot 3
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 3, 3)
plt.title("3")
plt.plot(x, y)

plt.show()

#################################
# SEABORN
#################################

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df = sns.load_dataset("tips")
df.head()

df["sex"].value_counts()
sns.countplot(x=df["sex"], data=df)
plt.show()

# matplotlib ile
df["sex"].value_counts().plot(kind='bar')
plt.show()

##############################################
# Seaborn ile sayısal değişken görselleştirme
##############################################

sns.boxplot(x=df["total_bill"])
plt.show()

# pandas içinde yer alan hist() fonksiyonu ile
df["total_bill"].hist()
plt.show()

