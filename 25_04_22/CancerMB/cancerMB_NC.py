import sklearn
# importar e carregar o dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
#organizar dados

label_names = data['target_names']
labels = data['target']
feature_names = ['feature_name']
features = data ['data']

#olhando para os dados
#print(label_names)
#print(labels[0])
#print(feature_names[0]) <- deveria imprimir 'mean radius'
#print(features[0])

from sklearn.model_selection import train_test_split

#dividir os dados
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)

from sklearn.naive_bayes import GaussianNB

#iniciar classificador

gnb = GaussianNB()
#treinar classificador
model = gnb.fit(train, train_labels)


#fazer previsões
preds = gnb.predict(test)
print(preds)


from sklearn.metrics import accuracy_score
# avaliar a precisão

print('Accuracy: ', accuracy_score(test_labels, preds))

