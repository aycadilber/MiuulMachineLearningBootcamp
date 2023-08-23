####################################################################
# Sales Prediction with Linear Regression
####################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

pd.set_option("display.float_format", lambda x: "%.2f" % x)

######################################################
# Simple Linear Regression with OLS Using Scikit-Learn
######################################################

df = pd.read_csv("datasets/advertising.csv")
df.shape

X = df[["TV"]]
y = df[["sales"]]

##########################
# Model
##########################

reg_model = LinearRegression().fit(X, y)

# y_hat = b + w*x
# y_hat = b + w*TV

# sabit (b - bias)
reg_model.intercept_[0]

# tv'nin katsayısı (w1)
reg_model.coef_[0][0]

# reg_model.intercept_
# Out[6]: array([7.03259355])
# reg_model.coef_
# Out[7]: array([[0.04753664]])

##########################
# Tahmin
##########################

# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir?

reg_model.intercept_[0] + reg_model.coef_[0][0]*150

# 500 birimlik tv harcaması olsa ne kadar satış olur?

reg_model.intercept_[0] + reg_model.coef_[0][0]*500

df.describe().T


# Modelin Görselleştirilmesi
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")

g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()


##########################
# Tahmin Başarısı
##########################

#### MSE
# MSE metodu der ki ; bana gerçek ve tahmin edilen değerleri ver.Bunların farklarını alırım, karelerini alırım. Toplayıp
# ortalamasını alarak sana ortalama hatanı verebilirim der.

# y_pred = reg_model.predict(y, y_pred)

# Fakat elimizde tahmin edilen değerler yok. Ne yapmamız lazım ?
# reg_model i kullanarak predict metodunu çağırıyorum. Ve bu metoda bağımsız değişkenlerimi yani X 'i veriyorum ve diyorum ki
# bu X değerlerini sorsam bu değerlere göre bana bu modelin bağımlı değişkenlerini tahmin etsen.
# Dolayısıyla bağımsız değişkenleri modele sordum ve bağımlı değişkenleri tahmin etmesini isteyerek y_pred değişkenine atadım.

y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)
# 10.51
y.mean()  # satışların ortalaması 14
y.std()  # satışların standart sapması 5

# ortalaması 14. Standart sapması 5 çıktı. Yani ortalama 19 ve 9 arasında değerler değişiyor gibi gözüküyor.
# Bu durumda  elde ettiğim 10 değeri büyük mü kücük mü diye düşünecek olursak, büyük kaldı gibi .
# Ne kadar kücük o kadar iyi.  Fakat büyüklüğü de neye göre değerlendirmem gerektiğini bilmediğim durumda
# bağımlı değişkenin ortalamasına ve standart sapmasına bakıyoruz.


#### RMSE
np.sqrt(mean_squared_error(y, y_pred))  # 3.24

#### MAE --> mutlak hata
mean_absolute_error(y, y_pred)    # 2.54

# R-KARE
# R-KARE : score metodu ile gerçekleştiriyoruz. Regresyon modeline bir score hesapla talebini gönderiyoruz
# R-KARE : Doğrusal regresyon modellerinde modelin başarısına ilişkin cok önemli olan bir metriktir.
# Verisetindeki bağımsız değişkenlerin bağımlı değişkeni açıklama yüzdesidir.
# Televizyon değişkeninin satış değişkenindeki değişikliği açıklama yüzdesidir.
# Bu modelde bağımsız değişkenler , bağımlı değişkenin yüzde 61'ini açıklayabilmektedir şeklinde yorumlanır.

reg_model.score(X, y)
#0.61



######################################################
# Multiple Linear Regression
######################################################

df = pd.read_csv("datasets/advertising.csv")

X = df.drop('sales', axis=1)  #bağımsız değişkenler
y = df[["sales"]]

##########################
# Model
##########################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

X_test.shape
X_train.shape
y_test.shape
y_train.shape

reg_model = LinearRegression().fit(X_train, y_train)

# sabit (b - bias)
reg_model.intercept_[0]

# coefficients (w - weights)
reg_model.coef_


##########################
# Tahmin
##########################

# Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?

# TV: 30
# radio: 10
# newspaper: 40

# sabit: 2.90
# w : 0.0468431 , 0.17854434, 0.00258619

# model denklemi:
# Sales = 2.90  + TV * 0.04 + radio * 0.17 + newspaper * 0.002

2.90794702 + 30 * 0.0468431 + 10 * 0.17854434 + 40 * 0.00258619
#6.202131

# Diyelim ki modeli kurduk . Bu modeli canlı sistemle basit bir excel sistemiyle bir yerlerle entegre edeceğimizi düşünelim
# Diyelim ki bir departman ilgili TV,radyo ve gazete harcamalarını girerek bir tahmin sonucu alacak
# girilecek değerleri bir listeye çeviriyoruz
# ardından bu değerleri dataframe'e ceviriyoruz ve transpozunu alıyoruz.
yeni_veri = [[30], [10], [40]]
yeni_veri = pd.DataFrame(yeni_veri).T

# model nesneme tahmin et diyorum. Neyi tahmin edeyim diyor? Bana bağımsız değişkenleri ver, ben gideyim bu regresyon
# modeline onları sorayım diyor.Buna göre de bağımlı değişkenin ne olabileceği bilgisini sana vereyim diyor.
reg_model.predict(yeni_veri)
#array([[6.202131]])


#################################
# Tahmin Başarısını Değerlendirme
#################################

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 1.73  train hatası


# TRAIN RKARE
reg_model.score(X_train, y_train)
# 0.8959372632325174
# daha önce  yüzde 60'lar civarında olan değer 3 tane yeni değişken eklediğimizde  yüzde 90 civarına geldi
# Buradan anladığımız sey yeni değişken eklendiğinde başarının arttıgı, hatanın düştüğüdür.


# Test RMSE
# İlk defa modele  test setini soruyoruz. predict metodu tahmin etmek için kullanılır .
# Reg modele diyoruz ki, simdi sana bir set göndericem bunu bi değerlendir bakalım diyor
y_pred = reg_model.predict(X_test)  # test setini gönder
# (Test setinin x'leri yani bağımsız değişkenlerini soruyoruz modele) o da test setinin bağımlı değişkeninini tahmin ediyor.
np.sqrt(mean_squared_error(y_test, y_pred))  # bağımlı deeğişkenin bizde gerçek değeri var (y_test). Bağımlı değişkenin birde tahmin edilen değerleri var y_pred ' dir.
# 1.41
# normalde test hatası train hatasından daha yüksek cıkar. Burda düşük cıktı ,oldukça iyi bir durum


# Test RKARE
reg_model.score(X_test, y_test)
# veri setindeki bağımsız değişkenlerin bağımlı değişkenleri
# açıklama yüzdesi %90 civarında


# holdout yöntemi ile train,test olarak ayırdık.
# Train setinde model kurduk.
# Test setinde hatamızı değerlendirdik.
# bunun yerine 10 katlı cv da yapabilirdik.


# 10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))

# 1.69

#  neg_mean_squared_error metodunun cıktısı eksi değerlerdir.Bu nedenle - ile çarpıyoruz.
# Test hatamız 1.41, train hatamız 1.73 dü. Çapraz doğrulama ile elde ettiğimiz hata ise 1.69 cıktı.

# 5 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,  # tüm veri setini veriyoruz.
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 1.71

# bundan sonra bir regresyon kullanma ihtiyacımız oldugunda direkt olması gereken şey;
# - veri setini okuma
# - ilgili ön işleme, özellik mühendisliği işlemlerini yapma
# - model kurma basamağına gelince de modeli kurmadan önce verisetini 80' e  20 ayırmak
#  yada komple bütün veriye hata bakması gibi cross validation yapmak

