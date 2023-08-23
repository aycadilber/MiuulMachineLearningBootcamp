####################################
# FUNCTIONS
###################################

# Fonksiyon Tanımlama

def calculate(x):
    print(x**2)

calculate(5)

def summer(arg1, arg2):
    print(arg1 + arg2)

summer(8,5)


#Docstring
# - 1. bölümde fonksiyonun ne yaptığı ifade edilir.
# - 2. bölümde parametrelerin tipleri ve görevi  ifade edilir.
# - 3. bölümde return bilgisi girilir.
def summer(arg1, arg2):
    """
    Sum of two numbers
    Parameters
    ----------
    arg1: float, int
    arg2: float, int

    Returns:
        int, float
    -------

    """
    print(arg1, arg2)


# girilen değerleri bir liste içinde saklayacak fonksiyon.

list_store = []

def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)

add_element(1, 8)

# append metodu yeniden atama yapmaya gerek kalmaksızın kalıcı değişiklik yapar.

## Ön Tanımlı Argümanlar/Parametreler (Default Parameters/Arguments)
# Fonksiyonun hiçbir değer verilmezse sabit bir değer verilmesidir.

def divide(a, b=2):
    print(a / b)

divide(1)

#### Ne Zaman Fonksiyon Yazılır?

# Fonksiyonları birbirini tekrar eden durumlarda kullanırız.
# DRY (don't repeat yourself) prensibini sağlar.

def calculate(varm, moisture, charge):
    print((varm + moisture) / charge)

calculate(98, 12, 78) - 200


######################
# Return: Fonksiyon Çıktılarını Girdi OLarak KUllanmak
######################

def calculate(varm, moisture, charge):
    print((varm + moisture) / charge)

def calculate(varm, moisture, charge):
    return (varm + moisture) / charge

print(calculate(98, 12, 78) * 10)


def calculate(varm, moisture, charge):
    varm = varm * 2
    moisture = moisture * 2
    charge = charge * 2
    output = (varm + moisture) / charge
    return varm, moisture, charge, output

type(calculate(98, 12, 78))

varm, moisture, charge, output = calculate(98, 12, 78)

#############################
# Fonksiyon İçerisinden Fonksiyon Çağırmak
#############################

def calculate(varm, moisture, charge):
    return int((varm + moisture) / charge)

print(calculate(90, 12, 12) * 10)


def standardization(a, p):
    return a * 10 / 100 * p * p

print(standardization(45, 1))

def all_calculation(varm, moisture, charge, p):
    a = calculate(varm, moisture, charge)
    b = standardization(a, p)
    print(b * 10)

all_calculation(1, 3, 5, 12)


def all_calculation(varm, moisture, charge, a, p):
    print(calculate(varm, moisture, charge))
    b = standardization(a, p)
    print(b * 10)

all_calculation(1, 3, 5, 19, 12)
