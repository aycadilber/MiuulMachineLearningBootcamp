#################################################
# Enumerate: Otomatik Counter/Indexer ile for loop
##################################################


students = ["John", "Mark", "Venessa", "Mariam"]

for student in students:
    print(student)
    #burada indeksler elimizde yok

for index, student in enumerate(students):
    print(index, student)

A = []
B = []

for index, student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)
print(A)
print(B)

#Uygulama - Mülakat sorusu
# divide_students fonksiyonu yazınız.
# Çift indexte yer alan öğrencileri bir listeye alınız.
# Tek indexte yer alan öğrencileri başka bir listeye alınız.
# Fakat bu iki liste tek bir liste olarak return olsun.

students = ["John", "Mark", "Venessa", "Mariam"]
def divide_students(students):
    lists = [[], []]
    for index,student in enumerate(students):
        if index % 2 == 0:
            lists[0].append(student)
        else:
            lists[1].append(student)
    print(lists)
    return lists

st = divide_students(students)
st[0]


# alternating fonksiyonunun enumerate ile yazılması
def alternatingWithEnumarate(string):
    str = ""
    for i, letter in enumerate(string):
        if i % 2 == 0:
            str += letter.upper()
        else:
            str += letter.lower()
    print(str)


alternatingWithEnumarate("hi my name is john and i am learning python")


