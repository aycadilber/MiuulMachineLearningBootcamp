###########################
# Set(Küme)
###########################

# - Değiştirilebilir.
# - Sırasız + Eşsizdir.
# - Kapsayıcıdır.

#### difference(): İki kümenin farkı

set1 = set([1, 3, 5])
set2 = set([1, 2, 3])
set1.difference(set2) #set1'de olup set2'de olmayanlar
set2.difference(set1)

#### symmetric_difference(): İki kümede de birbirlerine göre olmayanlar
set1.symmetric_difference(set2)
set2.symmetric_difference(set1)

#### intersection(): İki kümenin kesişimi
set1.intersection(set2)

#### union(): İki kümenin birleşimi
set1.union(set2)

#### isdisjoint(): İki kümenin kesişimi boş mu?
set1 = set([7, 8, 9])
set2 = set([5, 6, 7, 8, 9, 10])
set1.isdisjoint(set2)

#### issubset(): Bir küme diğer kümenin alt kümesi mi?
set1.issubset(set2)
set2.issubset(set1)

#### issuperset(): Bir küme diğer kümeyi kapsıyor mu?
set1.issuperset(set2)
set2.issuperset(set1)

name = "vbo_bootcamp"
type = "newt"
f"Name:{name} type:{type}"

total = 3.4 + 2.6
print(total)