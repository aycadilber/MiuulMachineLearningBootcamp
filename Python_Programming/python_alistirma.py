## GÖREV 1: Verilen değerlerin veri yapılarını inceleyiniz ##
x = 8

y = 3.2

z = 8j + 18

a = "hello world"

b = True

c = 23 < 22

l = [1, 2, 3, 4]

d = {"Name": "Jake",
     "age": 27,
     "address": "downtown"}

t = {"machine learning", "data science"}

s = {"Python", "Machine learning", "Data Science"}

new_list = [x, y, z, a, b, c, l, d, t, s]

for i in new_list:
    print(type(i))

############################
# Görev 2: Verilen String ifadenin tüm harflerini büyük harfe çeviriniz.
# virgül ve nokta yerine space koyunuz
# kelime kelime ayırınız
############################
text = "The goal is to turn data into information, and information into insight."
text.upper()
text.upper().replace(",", "").replace(".","").split()
text.replace(".","")


############################
# Görev 3: Verilen listeye aşağıdaki adımları uygulayınız
############################

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

# Adım1: Verilen listenin eleman sayısına bakınız.
# Adım2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.
# Adım3: Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz.
# Adım4: Sekizinci indeksteki elemanı siliniz.
# Adım5: Yeni bireleman ekleyiniz.
# Adım6: Sekizinci indekse"N" elemanını tekrar ekleyiniz.

len(lst)
lst[0]
lst[10]
lst[0:4]
lst.pop(8)

lst.append("q")

lst.insert(8, "N")

############################
# Görev 4: Verilen sözlük yapısına aşağıdaki adımları uygulayınız.
# Adım1: key değerlerine erişiniz.
# Adım2: value'lara erişiniz.
# Adım3: daisy key'ine ait 12 değerini 13 olarak güncelleyin.
# Adım4: key değeri ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.
# Adım5:  antonio'yu dictionary'den siliniz.
############################

dict = {"Christian": ["America", 18],
        "Daisy": ["England", 12],
        "Antonio": ["Spain", 22],
        "Dante": ["Italy", 25]}

dict.keys()
dict.values()
dict["Daisy"][1] = 13
dict.update({"Ahmet": ["Turkey", 24]})
dict.pop("Antonio")


############################
# Görev 5: Argüman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları ayrı listelere atayan
# ve bu listeleri return eden fonksiyon yazınız.
############################

l = [2, 13, 18, 93, 22]

def func(list):
    odd_list = []
    even_list = []
    for i in list:
        if i % 2 == 0:
            even_list.append(i)
        else:
            odd_list.append(i)
    return odd_list, even_list

even_list, odd_list =func(l)



############################
# Görev 6:  Sırasıyla ilk üç öğrenci mühendislik fakültesinin başarı sırasını temsil ederken son üç öğrenci de
# tıp fakültesi öğrenci sırasına aittir.Enumarate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız.
############################

ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]

for index, ogrenci in enumerate(ogrenciler, 1):
    if index <= 3:
        print(f"Mühendislik Fakültesi {index}. öğrenci: {ogrenci}")
    else:
        tıp_fakültesi_index = index - 3
        print(f"Tıp Fakültesi {tıp_fakültesi_index}. ogrenci: {ogrenci}")


############################
# Görev 7:Aşağıda 3 adet liste verilmiştir. Zip kullanarak ders bilgilerini bastırınız.
############################

ders_kod = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
kredi = [3, 4, 2, 4]
kontenjan = [30, 75, 150, 25]


for ders, krd, kont in zip(ders_kod, kredi, kontenjan):
    print(f"Kredisi {krd} olan {ders} kodlu dersin kontenjanı {kont} kişidir.")


############################
# Görev 8: Eğer 1. küme 2. kümeyi kapsıyor ise ortak elemanlarını kapsamıyorsa 2. kümenin
# 1. kümeden farkını yazdıracak fonksiyonu yaz.
############################


kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])

def func(set1, set2):
    if set1.issuperset(set2):
        print(set1.intersection(set2))
    else:
        print(set2.difference(set1))

func(kume1, kume2)


