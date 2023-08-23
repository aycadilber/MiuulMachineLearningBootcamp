###########################
## KOŞULLAR (CONDITIONS) ##
###########################

if 1 == 1:
    print("something")


number = 11
if number == 11:
    print("number is 10")

def number_check(num):
    if num == 10:
        print("number is 10")

number_check(11)
number_check(10)


def number_check(number):
    if number > 10:
        print("greater than 10")
    elif number < 10:
        print("less than 10")
    else:
        print("equal to 10")

number_check(6)


####################################
######### DÖNGÜLER (LOOPS) #########
####################################

## for loop

students = ["John", "Mark", "Venessa", "Mariam"]
for student in students:
    print(student)

for student in students:
    print(student.upper())


salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    print(salary)

for salary in salaries:
    print(int(salary*30/100 + salary))

def new_salary(salary, rate):
    return int(salary*rate/100 + salary)
new_salary(1500,10)


for salary in salaries:
    print(new_salary(salary, 15))

for salary in salaries:
    if salary > 3000:
        print(new_salary(salary, 10))
    else:
        print(new_salary(salary, 20))


#######################
# Uygulama- Mülakat Sorusu
#######################

# Amaç: Aşağıdaki şekilde string değiştiren fonksiyon yazmak istiyoruz.

# before: "hi my name is john and i am learning python"
# after: "Hi mY NaMe iS JoHn aNd i aM LeArNiNg pYtHoN"

#range(len(str)) : bütün indexlerinde gez

def alternating(string):
    str = ""
    for string_index in range(len(string)):
        if string_index % 2 == 0:
            str += string[string_index].upper()
        else:
            str += string[string_index].lower()
    print(str)

alternating("hi my name is john and i am learning python")


####################################
## break & continue & while
####################################

# Break => akışı keserek döngüyü bitirir.
salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    if salary == 3000:
        break
    print(salary)

#continue => O elemanı pas geçerek döngüye devam eder.
for salary in salaries:
    if salary == 3000:
        continue
    print(salary)

# while 

number = 1
while number < 5:
    print(number)
    number += 1
