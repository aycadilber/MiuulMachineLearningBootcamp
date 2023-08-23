##################################
# NUMPY
##################################

"""
* Numerical Python
* Numpy bilimsel hesaplamalar, çok boyutlu arraylar ve matrisler üzerinde yüksek performanslı çalışma imkanı sağlar.
* Numpy kütüphanesinin listelerden farkı verimli veri saklama (sabit bir tipte veri tutar) ve vektörel işlemlerin yapılmasıdır.
* Daha az çabayla daha çok işlem yapma imkanı sağlar.
"""

import numpy as np

a = [1, 2, 3, 4]
b = [2, 3, 4, 5]

ab = []

for i in range(0, len(a)):
    ab.append(a[i] * b[i])

#numpy ile
a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a * b

#############################
# NumPy Array'i OLuşturmak
#############################

np.array([1, 2, 3, 4, 5])

type(np.array([1, 2, 3, 4, 5]))

np.zeros(10, dtype=int)
#array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

np.random.randint(0, 10, size=10)
#array([0, 9, 2, 1, 2, 2, 6, 9, 9, 0])

np.random.normal(10, 4, (3,4))

#############################
# NumPy Array Özellikleri
#############################
# Numpy arraylerinin özelliklerini öğrenmek için np.info()
# ndim: boyut sayısı
# shape: boyut bilgisi
# size: toplam eleman sayısı
# dtype: array veri tipi

a = np.random.randint(10, size=5) #5 tane 0'dan 10'a kadar sayı üretir

a.ndim
a.shape
a.size
a.dtype

##########################
# Reshaping
##########################

np.random.randint(1, 10, size=9)
b = np.random.randint(1, 10, size=9).reshape(3,3)
b.ndim


##################################
# Index Seçimi (Index Selection)
##################################

a = np.random.randint(10, size=10)
a[0]
a[0:5]
a[0] = 999

m = np.random.randint(10, size=(3, 5))
m[0, 0]
m[1, 1]
m[2, 3] = 999

m[2, 3] = 2.9
# 2.9 değil 2 gelir çümkü numpy tek tipte çalışır

m[:, 0]  #: bütün satırlar, 0. sütun
m[0:2, 0:3]

##################################
# Fancy Index
##################################

v = np.arange(0, 30, 3)  #arange(start, stop, step)
v[1]
v[4]

catch = [1, 2, 3]  #bu indexlere denk gelen değerleri getirir.
v[catch]

#########################
# Numpy Koşullu İşlemler
#########################

v = np.array([1, 2, 3, 4, 5])

#klasik döngü ile
ab = []
for i in v:
    if i < 3:
        ab.append(i)

#numpy ile
v < 3
v[v < 3]
v[v != 3]
v[v >= 3]


#############################
# Matematiksel İşlemler
#############################

v = np.array([1, 2, 3, 4, 5])

v / 5
v * 5 / 10
v ** 2
v - 1


np.subtract(v, 1)
np.add(v, 1)
np.mean(v)
np.min(v)
np.max(v)
np.sum(v)
np.var(v)
