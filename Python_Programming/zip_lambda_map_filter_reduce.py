########################
# Zip : gruplama işlemi yapar
########################

students = ["John", "Mark", "Venessa", "Mariam"]

departments = ["mathematics", "statistics", "physics", "astronomy"]

ages = [23, 30, 26, 22]

list(zip(students, departments, ages))

#################################
# lambda => Fonksiyonu tek satırda yazmayı sağlar. kullan-at fonksiyon
#################################

def summer(a, b):
    return a + b

summer(1, 3) * 9

new_sum = lambda a, b : a + b
new_sum(4, 5)


#######################
# map --> döngü yazmaktan kurtarır
#######################

salaries = [1000, 2000, 3000, 4000, 5000]

def new_salary(x):
    return x * 20 / 100 + x

new_salary(5000)

for salary in salaries:
    print(new_salary(salary))


list(map(new_salary, salaries))

list(map(lambda x: x * 20 / 100 + x, salaries))

list(map(lambda x: x ** 2, salaries))

###########################
# filter --> belirli koşulu sağlayanları seçmek için
###########################
list_store = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(list(filter(lambda x: x % 2 == 0, list_store)))


#########################
# reduce --> ilgili elemanlara tek tek işlem uygulamak
#########################

from functools import reduce
list_store = [1, 2, 3, 4]
print(reduce(lambda a, b: a + b, list_store))

