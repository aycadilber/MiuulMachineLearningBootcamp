#########################
# Demet (Tuple)
#########################

# - Değiştirilemez.
# - Sıralıdır.
# - Kapsayıcıdır.

t = ("john", "mark", 1, 2)
type(t)

t[0]
t[0:3]

t[0] = 99 #TypeError: 'tuple' object does not support item assignment

t = list(t)
t[0] = 99
t = tuple(t)
