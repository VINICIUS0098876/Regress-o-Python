# Importando as bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Gerando dados de exemplo
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # Variável independente
y_reg = 4 + 3 * X + np.random.randn(100, 1)  # Variável dependente para Regressão Linear
y_log = (X + 0.5 * np.random.randn(100, 1)) > 1  # Variável dependente binária para Regressão Logística

# Dividindo os dados em conjunto de treinamento e teste
X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_train_log, X_test_log, y_log_train, y_log_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Criando e treinando o modelo de Regressão Linear
model_linear = LinearRegression()
model_linear.fit(X_train, y_reg_train)

# Criando e treinando o modelo de Regressão Logística
model_logistic = LogisticRegression()
model_logistic.fit(X_train_log, y_log_train)

# Criando e treinando o modelo de Regressão Ridge
model_ridge = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=0.1))
model_ridge.fit(X_train, y_reg_train)

# Fazendo previsões nos conjuntos de teste
y_reg_pred = model_linear.predict(X_test)
y_log_pred = model_logistic.predict(X_test_log)
y_ridge_pred = model_ridge.predict(X_test)

# Calculando métricas de desempenho para Regressão Linear
mse_linear = mean_squared_error(y_reg_test, y_reg_pred)

# Calculando métricas de desempenho para Regressão Logística
accuracy_logistic = accuracy_score(y_log_test, y_log_pred)

# Calculando métricas de desempenho para Regressão Ridge
mse_ridge = mean_squared_error(y_reg_test, y_ridge_pred)

# Plotando os resultados da Regressão Linear
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_test, y_reg_test, color='blue')
plt.plot(X_test, y_reg_pred, color='red', linewidth=2)
plt.title('Regressão Linear')
plt.xlabel('Variável Independente')
plt.ylabel('Variável Dependente')

# Plotando os resultados da Regressão Logística
plt.subplot(1, 3, 2)
plt.scatter(X_test, y_log_test, color='blue')
plt.scatter(X_test, y_log_pred, color='red', marker='x')
plt.title('Regressão Logística')
plt.xlabel('Variável Independente')
plt.ylabel('Classe')

# Plotando os resultados da Regressão Ridge
plt.subplot(1, 3, 3)
plt.scatter(X_test, y_reg_test, color='blue')
plt.scatter(X_test, y_ridge_pred, color='red')
plt.title('Regressão Ridge')
plt.xlabel('Variável Independente')
plt.ylabel('Variável Dependente')

plt.tight_layout()
plt.show()
