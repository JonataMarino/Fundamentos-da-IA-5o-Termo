"""
Vamos começar instalando o módulo Python Scikit-learn, um das melhores e mais bem
documentadas bibliotecas de machine learning para Python.

Para começar com nosso projeto de codificação, vamos ativar nosso ambiente de
programação Python 3.
"""

import sklearn

"""
Passo 2 — Importando o Dataset do Scikit-learn
O dataset com o qual estaremos trabalhando neste tutorial é o Breast Cancer Wisconsin Diagnostic Database. 
O dataset inclui várias informações sobre tumores de câncer de mama, bem como rótulos de classificação como 
malignos ou benignos. O dataset tem 569 instâncias, ou dados, sobre 569 tumores e inclui informações sobre 
30 atributos, ou características, tais como o raio do tumor, textura, suavidade, e área.

Utilizando este dataset, construiremos um modelo de machine learning para utilizar as informações sobre 
tumores para prever se um tumor é maligno ou benigno.

O Scikit-learn vem instalado com vários datasets que podemos carregar no Python, e o dataset que queremos 
está incluído.
"""

# importar e carregar o dataset

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
"""
A variável data representa um objeto Python que funciona como um dicionário. As chaves importantes do 
dicionário a considerar são os nomes dos rótulos de classificação (target_names), os rótulos reais (target), 
os nomes de atributo/característica (feature_names), e os atributos (data).

Atributos são uma parte crítica de qualquer classificador. Os atributos capturam características importantes 
sobre a natureza dos dados. Dado o rótulo que estamos tentando prever (tumor maligno versus benigno), os 
possíveis atributos úteis incluem o tamanho, raio, e a textura do tumor.

Crie novas variáveis para cada conjunto importante de informações e atribua os dados:
"""
#organizar dados

label_names = data['target_names']
labels = data['target']
feature_names = ['feature_name']
features = data ['data']
"""
Agora temos listas para cada conjunto de informações. Para entender melhor nosso conjunto de dados, 
vamos dar uma olhada em nossos dados imprimindo nossos rótulos de classe, o primeiro rótulo da 
instância de dados, nossos nomes de características, e os valores das características para a primeira 
instância de dados.
"""
#olhando para os dados
#print(label_names)
#print(labels[0])
#print(feature_names[0]) <- deveria imprimir 'mean radius'
#print(features[0])

"""
Passo 3 — Organizando Dados em Conjuntos

Para avaliar o desempenho de um classificador, você deve sempre testar o modelo em dados não visualizados. 
Portanto, antes da construção de um modelo, divida seus dados em duas partes: um conjunto de treinamento e 
um conjunto de testes.

Você usa o conjunto de testes para treinar e avaliar o modelo durante o estágio de desenvolvimento. Então 
você usa o modelo treinado para fazer previsões no conjunto de testes não visualizado. Essa abordagem lhe 
dá uma noção do desempenho e robustez do modelo.

Felizmente, o sklearn tem uma função chamada train_test_split(), que divide seus dados nesses conjuntos. 
Importe a função e em seguida utilize-a para dividir os dados:
"""

from sklearn.model_selection import train_test_split

#dividir os dados
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)

"""
A função divide aleatoriamente os dados usando o parâmetro test_size. Neste exemplo, agora temos um conjunto 
de testes (test) que representa 33% do dataset original. Os dados restantes (train) formam então os dados de 
treinamento. Também temos os respectivos rótulos para ambas as variáveis train/test, ou seja, train_labels e 
test_labels.

Agora podemos passar para o treinamento do nosso primeiro modelo.

Passo 4 — Construindo e Avaliando o Modelo
Existem muitos modelos para machine learning, e cada modelo tem seus pontos fortes e fracos. Neste tutorial, 
vamos nos concentrar em um algoritmo simples que geralmente funciona bem em tarefas de classificação binária, 
a saber Naive Bayes (NB).

Primeiro, importe o módulo GaussianNB. Em seguida inicialize o modelo com a função GaussianNB(), depois treine 
o modelo, ajustando-o aos dados usando gnb.fit():
"""
from sklearn.naive_bayes import GaussianNB

#iniciar classificador

gnb = GaussianNB()
#treinar classificador
model = gnb.fit(train, train_labels)

"""
Depois de treinarmos o modelo, podemos usar o modelo treinado para fazer previsões no nosso conjunto de teste, 
o que fazemos utilizando a função predict(). A função predict() retorna uma matriz de previsões para cada 
instância de dados no conjunto de testes. Podemos então, imprimir nossas previsões para ter uma ideia do que o 
modelo determinou.

Utilize a função predict() com o conjunto test e imprima os resultados:
"""
#fazer previsões
preds = gnb.predict(test)
print(preds)

"""
Como você vê na saída, a função predict() retornou uma matriz de 0s e 1s que representa nossos valores previstos 
para a classe tumor (maligno vs. benigno).

Agora que temos nossas previsões, vamos avaliar o desempenho do nosso classificador.


Passo 5 — Avaliando a Precisão do Modelo
Usando a matriz de rótulos de classe verdadeira, podemos avaliar a precisão dos valores previstos do nosso modelo 
comparando as duas matrizes (test_labels vs. preds). Utilizaremos a função accuracy_score() do sklearn para 
determinar a precisão do nosso classificador de machine learning.
"""
from sklearn.metrics import accuracy_score
# avaliar a precisão

print('Accuracy: ', accuracy_score(test_labels, preds))

"""
Como você vê na saída, o classificador NB é ~= 94.15% preciso. Isso significa que 94,15 porcento do tempo o 
classificador é capaz de fazer a previsão correta se o tumor é maligno ou benigno. Esses resultados sugerem 
que nosso conjunto de características de 30 atributos são bons indicadores da classe do tumor.
"""