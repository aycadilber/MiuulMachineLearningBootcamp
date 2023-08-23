# Liste (List)

# - Değiştirilebilir
# - Sıralıdır. Index işlemleri yapılabilir.
# - Kapsayıcıdır.Yani içinde birden fazla veri yapısını tutabilir.
# - Her tür veri tipi içeride olabilir.

notes = [1, 2, 3, 4]
type(notes)

names = ["a", "b", "v", "d"]

not_nam = [1, 2, 3, "a", "b", True, [1, 2, 3]]

not_nam[6]
not_nam[6][1]

notes[0] = 99
notes


not_nam[0:4]

#########################
# Lİste Metodları
#########################

dir(notes)

## len fonksiyonu ##

len(notes)


# append: Listenin sonuna eleman ekler.

notes.append(100)
notes

# pop: indexe göre silme işlemi yapar.
notes.pop(1)
notes

# insert: indexe göre eleman ekler
notes.insert(2, 98)
notes