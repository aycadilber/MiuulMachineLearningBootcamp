################################
# STRINGS (KARAKTER DİZİLERİ)
################################

print("ayça")

name = "ayça"

## Karakter Dizilerinin Elemanlarına Erişmek

name[0]

# Slice

name[0:2]

# string içinde karakter sorgulama

longStr = """Veri Yapıları: Hızlı Özet, 
Sayılar (Numbers): int, float, complex, 
Karakter Dizileri (Strings): str, 
List, Dictionary, Tuple, Set, 
Boolean (TRUE-FALSE): bool"""

"Veri" in longStr

##########################
#### String Metodları ####
##########################
dir(str)

## len fonksiyonu --> string değerin kaç elemandan oluştuğun verir
name = "ayça"
len(name)

# fonksiyonlar bağımsızdır, metodlar ise classlar içinde tanımlanmıştır

## upper() & lower() metodları ##

"miuul".upper()
"MIUUL".lower()

## replace metodu ##

hi = "Hello AI Era"
hi.replace("l", "p")

## split: böler ##

"Hello AI Era".split()

## strip: Baştan ve sondan boşlukları kırpar.
" ofofo ".strip()
"ofofo".strip("o")

## capitalize : İlk harfi büyütür.
"foo".capitalize()

