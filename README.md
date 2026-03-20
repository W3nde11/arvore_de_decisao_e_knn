# Classificação de Flores Iris com Árvore de Decisão e KNN

## Descrição
Este projeto demonstra a aplicação de dois algoritmos de classificação, Árvore de Decisão e K-Nearest Neighbors (KNN), para classificar as espécies de flores Iris utilizando o famoso dataset Iris. O objetivo é comparar o desempenho desses modelos em termos de acurácia e visualizar a matriz de confusão para uma análise mais aprofundada.

## Funcionalidades
- Carregamento e pré-processamento do dataset Iris.
- Divisão dos dados em conjuntos de treino e teste.
- Treinamento de modelos de classificação: Árvore de Decisão e KNN.
- Avaliação da acurácia de cada modelo.
- Geração e visualização da matriz de confusão para o modelo de Árvore de Decisão.
- Comparação visual da acurácia dos modelos através de um gráfico de barras.

## Tecnologias Utilizadas
- Python 3.x
- `scikit-learn`: Para carregamento do dataset, divisão dos dados, algoritmos de classificação e métricas de avaliação.
- `matplotlib`: Para visualização dos resultados (gráficos).
- `numpy`: Para operações numéricas (utilizado implicitamente pelo matplotlib e scikit-learn).

## Instalação
Para executar este projeto, você precisará ter o Python instalado. Recomenda-se criar um ambiente virtual.

1. Clone o repositório (se aplicável) ou salve o código em um arquivo `iris_classification.py`.
2. Instale as dependências necessárias:

```bash
pip install scikit-learn matplotlib numpy
```

## Uso
Para executar o script e ver os resultados, navegue até o diretório onde o arquivo `iris_classification.py` está salvo e execute o seguinte comando:

```bash
python iris_classification.py
```

O script irá imprimir a acurácia de ambos os modelos, a matriz de confusão para o modelo de Árvore de Decisão e exibirá dois gráficos: um comparando as acurácias e outro mostrando a matriz de confusão.

## Código
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


dados = load_iris()


X = dados.data 

y = dados.target 


X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.3, random_state=42
)



modelo_arvore = DecisionTreeClassifier()
modelo_arvore.fit(X_treino, y_treino)


pred_arvore = modelo_arvore.predict(X_teste)
acc_arvore = accuracy_score(y_teste, pred_arvore)


modelo_knn = KNeighborsClassifier()
modelo_knn.fit(X_treino, y_treino)


pred_knn = modelo_knn.predict(X_teste)
acc_knn = accuracy_score(y_teste, pred_knn)


print("Acurácia Árvore:", acc_arvore)
print("Acurácia KNN:", acc_knn)


matriz = confusion_matrix(y_teste, pred_arvore)
print("\nMatriz de Confusão:\n", matriz)


fig, ax = plt.subplots(1, 2, figsize=(10, 4))


modelos = ["Árvore", "KNN"]
valores = [acc_arvore, acc_knn]


ax[0].bar(modelos, valores)
ax[0].set_title("Comparação de Modelos")
ax[0].set_ylabel("Acurácia")
ax[0].set_ylim(0, 1)


img = ax[1].imshow(matriz)


ax[1].set_title("Matriz de Confusão")


classes = ["Setosa", "Versicolor", "Virginica"]


ax[1].set_xticks(range(len(classes)))
ax[1].set_yticks(range(len(classes)))


ax[1].set_xticklabels(classes)
ax[1].set_yticklabels(classes)


ax[1].set_xlabel("Previsto")
ax[1].set_ylabel("Real")


for i in range(len(classes)):
    for j in range(len(classes)):
        ax[1].text(j, i, matriz[i, j], ha="center", va="center")


fig.colorbar(img, ax=ax[1])


plt.tight_layout()
plt.show()
```

## Resultados Esperados
Ao executar o script, você verá a saída no console com as acurácias dos modelos e a matriz de confusão. Além disso, duas janelas de gráfico serão exibidas:

1.  **Comparação de Modelos**: Um gráfico de barras mostrando a acurácia de cada modelo (Árvore de Decisão e KNN).
2.  **Matriz de Confusão**: Uma visualização da matriz de confusão para o modelo de Árvore de Decisão, indicando o número de previsões corretas e incorretas para cada classe.


