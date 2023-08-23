
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

##############################
# 1. Exploratory Data Analysis
##############################

df = pd.read_csv("datasets/diabetes.csv")
df.head()
df.shape
df.describe().T
df["Outcome"].value_counts()

#############################################
# 2. Data Preprocessing & Feature Engineering
#############################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

# Kullanacak olduğumuz knn yöntemi uzaklık temelli bir yöntem. Uzaklık temelli yöntemlerde ve gradient descent temelli
# yöntemlerde değişkenlerin standart olması elde edilecek sonucların daha hızlı ve daha doğru olmasını sağlayacaktır.

X_scaled = StandardScaler().fit_transform(X)

X = pd.DataFrame(X_scaled, columns=X.columns)


##########################
# 3. Modeling & Prediction
##########################
knn_model = KNeighborsClassifier().fit(X, y)

random_user = X.sample(1, random_state=45)

knn_model.predict(random_user)

##########################
# 4. Model Evaluation
##########################

# confusion matrix için y_pred:
y_pred = knn_model.predict(X)

# AUC hesaplamak için bize 1 sınıfına ait olabilme olasılıkları lazım yani 1. indexteki değerleri getiriyoruz:
y_prob = knn_model.predict_proba(X)[:, 1]

# bu olasılık değerleri üzerinden roc auc score hesaplayacağız

print(classification_report(y, y_pred))
# accuracy 0.83 --> 100 kişiden 83'üne diyabet ya da diyabet değil dediğimizde bu doğru oluyor.%17'sinde tahminimiz başarısız oluyormuş.
# precision 0.79  --> 1 olarak tahmin ettiklerimizin başarısı
# recall 0.70   --> gerçekte 1 olanları 1 olarak tahmin etme başarımız
# f1 score 0.74

# AUC
roc_auc_score(y, y_prob)
# 0.90

# modelin görmediği verideki performansına bakmak için: holdout veya cross-validation

cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])


cv_results["test_accuracy"].mean()   # 0.73
cv_results["test_f1"].mean()         # 0.59
cv_results["test_roc_auc"].mean()    # 0.78

# bu başarı skorları nasıl yükseltebiliriz?
# veri boyutu arttırılabilir
# yeni değişkne türetme
# ilgili algoritma için optimizasyonlar yapılabilir

knn_model.get_params()

##################################################
# 5. Hyperparameter Optimization
##################################################

# dışarıdan kullanıcının ayyarlaması gereken parametreleri en doğru olarak nasıl ayarlamamaız gerektiğini öğreneceğiz.

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2, 50)}


# 2'den 50'ye kadar olanları arayacak bunun için GridSearchCV kullanılır
knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)

#n_jobs = -1--> işlemciler en yüksek performansıyla kullanılır.
# verbose= 1 --> rapor için 1

knn_gs_best.best_params_   #{'n_neighbors': 17}



##########################################
# 6. Final Model
##########################################

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_final,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])


cv_results["test_accuracy"].mean()   # 0.76
cv_results["test_f1"].mean()         # 0.61
cv_results["test_roc_auc"].mean()    # 0.81


random_user = X.sample(1)
knn_final.predict(random_user)   # array([0] diyabet değil