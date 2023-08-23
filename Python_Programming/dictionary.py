###########################
# Dictionary(Sözlük)
###########################

# Değiştirilebilir.
# Normal şartlarda sırasızdır. Ama 3.7 versiyonuyla birlikte sıralı özelliği kazanmıştır.
# Her veri tipini içerisinde barındırır.(kapsayıcı)
# key-value

dictionary = {"REG": "Regression",
              "LOG": "Logistic Regression",
              "CART": "Classification and Reg"}

dictionary["REG"]


dictionary = {"REG": ["RMSE", 10],
              "LOG": ["MSE", 20],
              "CART": ["SSE", 30]}

dictionary["CART"][1]

####### Key Sorgulama  #######

"YSA" in dictionary
"CART" in dictionary

####### Key'e Göre Value'ya Erişmek ######

dictionary["REG"]
dictionary.get("REG")


###### Value Değiştirmek ######

dictionary["REG"] = ["YSA", 10]


########  Tüm Key value Erişmek #########

dictionary.keys()
dictionary.values()

##### Tüm Çiftleri Tuple Halinde Listeye Çevirme ####
dictionary.items()


#### Key-Value Değerini Güncellemek ####
dictionary.update({"REG": 11})

#### Yeni Key-Value Eklemek ####
dictionary.update({"RF": 10})