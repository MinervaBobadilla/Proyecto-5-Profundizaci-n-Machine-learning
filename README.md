# **Proyecto 5 Profundización “Machine learning”**
**Ficha Técnica: Proyecto 5 Profundización “Machine learning”**

**Título del Proyecto**

**Proyecto 5 Profundización**

**Objetivo**
Desarrollar un modelo de machine learning supervisado para anticipar y gestionar la rotación del personal, fortaleciendo así la capacidad de nuestra empresa para retener y desarrollar el talento clave.

**Equipo**
Trabajo personal

**Herramientas y Tecnologías**
Listado de herramientas y tecnologías utilizadas en el proyecto:
1) Google Colab (Python).
2) Workspace Google ( Presentaciones, Chat GPT y Documentos).
3) Videos Looms.
4) Git Hub.



**Procesamiento y análisis**
**Proceso de Análisis de Datos:**

**1.1 Procesar y preparar base de datos**
Conectar/importar datos a otras herramientas.
1) Identificar y manejar valores nulos.
2) Identificar y manejar valores duplicados.
3) Identificar y manejar datos fuera del alcance del análisis.
4) Identificar y manejar datos discrepantes en variables categóricas.
5) Identificar y manejar datos discrepantes en variables numéricas.
6) Comprobar y cambiar tipo de dato.
7) Dividir base en train y test.
8) Crear nuevas variables.



A continuación ejemplos de lo anterior:





```
import pandas as pd
import numpy as np
from scipy import stats

# 1. Corregir inconsistencias en variables categóricas
def corregir_inconsistencias_categoricas(df):
    # Seleccionar solo las columnas categóricas (tipo object o category)
    datos_categoricos = df.select_dtypes(include=['object', 'category'])

    # Diccionario para almacenar valores corregidos
    correcciones = {
        'Yes': 'yes',
        'YES': 'yes',
        'No': 'no',
        'NO': 'no'
        # Agrega más correcciones específicas según tu dataset
    }

    print("\nCorrigiendo inconsistencias en variables categóricas...")
    for col in datos_categoricos.columns:
        # Aplicar correcciones usando el diccionario
        df[col] = df[col].replace(correcciones)

    return df

# 2. Corregir outliers en variables numéricas usando IQR o Z-score
def corregir_outliers(df, metodo='iqr', z_threshold=3):
    # Seleccionar columnas numéricas
    datos_numericos = df.select_dtypes(include=[np.number])

    print("\nCorrigiendo outliers en variables numéricas...")

    for col in datos_numericos.columns:
        if metodo == 'iqr':
            # Método del rango intercuartil (IQR)
            Q1 = datos_numericos[col].quantile(0.25)
            Q3 = datos_numericos[col].quantile(0.75)
            IQR = Q3 - Q1

            # Definir umbral de outliers (1.5 * IQR)
            outlier_threshold_low = Q1 - 1.5 * IQR
            outlier_threshold_high = Q3 + 1.5 * IQR

            # Reemplazar outliers por la mediana
            mediana = datos_numericos[col].median()

            # Asegurarse de que la mediana es del mismo tipo de datos que la columna
            if pd.api.types.is_integer_dtype(df[col]):
                df.loc[(df[col] < outlier_threshold_low) | (df[col] > outlier_threshold_high), col] = int(mediana)
            else:
                df.loc[(df[col] < outlier_threshold_low) | (df[col] > outlier_threshold_high), col] = mediana

        elif metodo == 'zscore':
            # Método del Z-score
            z_scores = np.abs(stats.zscore(datos_numericos[col]))
            mediana = datos_numericos[col].median()

            # Reemplazar valores con z-score alto por la mediana
            if pd.api.types.is_integer_dtype(df[col]):
                df.loc[z_scores > z_threshold, col] = int(mediana)
            else:
                df.loc[z_scores > z_threshold, col] = mediana

    return df

# 3. Mostrar resultados
def mostrar_tabla_resultados(df):
    print("\nDatos corregidos:")
    print(df.head())

# Cargar los datos de ejemplo (reemplaza 'your_data.csv' con tu archivo)
df = datos

# Mostrar una descripción inicial de los datos
print("\nDescripción inicial de los datos:")
print(df.describe(include='all'))

# Corregir inconsistencias en variables categóricas
df_corregido = corregir_inconsistencias_categoricas(df)

# Corregir outliers en variables numéricas usando el método IQR
df_corregido = corregir_outliers(df_corregido, metodo='iqr')

# Mostrar la tabla de resultados después de las correcciones
mostrar_tabla_resultados(df_corregido)



  
```











**1.2 Análisis exploratorio**

1) Agrupar datos según variables categóricas.
2) Visualizar las variables categóricas.
3) Aplicar medidas de tendencia central.
4) Visualizar distribución.
5) Aplicar medidas de dispersión.



A continuación ejemplos de lo anterior:




```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos (reemplaza 'your_data.csv' con el nombre de tu archivo)
df = datos

# Variables categóricas y numéricas a analizar
variables = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike', 'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
             'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

# 1. Agrupar datos según variables categóricas
def agrupar_datos_categoricos(df, variable):
    print(f"\nDistribución de la variable '{variable}':")
    print(df[variable].value_counts())

# 2. Visualizar las variables categóricas
def visualizar_categoricas(df, variable):
    plt.figure(figsize=(8, 4))
    sns.countplot(x=variable, data=df)
    plt.title(f'Visualización de la variable {variable}')
    plt.xticks(rotation=45)
    plt.show()

# 3. Medidas de tendencia central y dispersión para variables numéricas
def calcular_tendencia_central_dispersion(df, variable):
    print(f"\nMedidas de tendencia central para '{variable}':")
    print(f"Media: {df[variable].mean()}")
    print(f"Mediana: {df[variable].median()}")
    print(f"Moda: {df[variable].mode()[0]}")

    print(f"\nMedidas de dispersión para '{variable}':")
    print(f"Desviación estándar: {df[variable].std()}")
    print(f"Varianza: {df[variable].var()}")
    print(f"Rango: {df[variable].max() - df[variable].min()}")

# 4. Visualizar distribución de las variables numéricas
def visualizar_distribucion(df, variable):
    plt.figure(figsize=(8, 4))
    sns.histplot(df[variable], kde=True, bins=30)
    plt.title(f'Distribución de {variable}')
    plt.xlabel(variable)
    plt.ylabel('Frecuencia')
    plt.show()

# Aplicar a todas las variables del conjunto de datos
for variable in variables:
    if df[variable].dtype == 'object':  # Si es categórica
        agrupar_datos_categoricos(df, variable)
        visualizar_categoricas(df, variable)
    else:  # Si es numérica
        calcular_tendencia_central_dispersion(df, variable)
        visualizar_distribucion(df, variable)


  
```




**1.3 Aplicar técnica de análisis**

a)  Machine learning (supervisado).

A continuación ejemplos de lo anterior:










```

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer  # Para manejar los NaN
import warnings  # Para suprimir advertencias

# Suprimir las advertencias de XGBoost
warnings.filterwarnings(action='ignore', category=UserWarning, module='xgboost')

# Cargar los datos (reemplaza 'datos' con tu DataFrame)
df = datos

# 1. Preprocesamiento de datos
# Convertir variables categóricas a numéricas con Label Encoding
label_encoder = LabelEncoder()
categorical_columns = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18']

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Definir las variables independientes (X) y dependiente (y)
X = df[['Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'Education', 'EducationField',
        'EmployeeCount', 'EmployeeID', 'Gender', 'JobLevel', 'JobRole', 'MaritalStatus', 'MonthlyIncome',
        'NumCompaniesWorked', 'Over18', 'PercentSalaryHike', 'StandardHours', 'StockOptionLevel',
        'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager']]
y = df['Attrition']

# Manejar valores faltantes (NaN) usando SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# 2. Entrenar y evaluar los modelos

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg_train = log_reg.predict(X_train)
y_pred_log_reg_test = log_reg.predict(X_test)
acc_log_reg_train = accuracy_score(y_train, y_pred_log_reg_train)
acc_log_reg_test = accuracy_score(y_test, y_pred_log_reg_test)

# XGBoost
xgboost_model = xgb.XGBClassifier(eval_metric='logloss', max_depth=4, learning_rate=0.1, n_estimators=100)
xgboost_model.fit(X_train, y_train)
y_pred_xgb_train = xgboost_model.predict(X_train)
y_pred_xgb_test = xgboost_model.predict(X_test)
acc_xgb_train = accuracy_score(y_train, y_pred_xgb_train)
acc_xgb_test = accuracy_score(y_test, y_pred_xgb_test)

# Random Forest
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)
y_pred_rf_train = random_forest.predict(X_train)
y_pred_rf_test = random_forest.predict(X_test)
acc_rf_train = accuracy_score(y_train, y_pred_rf_train)
acc_rf_test = accuracy_score(y_test, y_pred_rf_test)

# 3. Mostrar resultados en una tabla
resultados = pd.DataFrame({
    'Modelo': ['Logistic Regression', 'XGBoost', 'RandomForest'],
    'Exactitud en Train': [acc_log_reg_train, acc_xgb_train, acc_rf_train],
    'Exactitud en Test': [acc_log_reg_test, acc_xgb_test, acc_rf_test]
})

print("\nResultados de los modelos (Train vs Test):")
print(resultados)

# 4. Visualización de resultados: comparación entre exactitud en entrenamiento y prueba
plt.figure(figsize=(12, 8))
barplot = sns.barplot(x='Modelo', y='Exactitud en Train', data=resultados, color='blue', label="Train", dodge=True)
sns.barplot(x='Modelo', y='Exactitud en Test', data=resultados, color='red', label="Test", dodge=True)
plt.title('Comparación de Exactitud entre Modelos en Train y Test')
plt.ylabel('Exactitud')
plt.legend()

# Agregar etiquetas de los valores sobre las barras
for index, value in enumerate(resultados['Exactitud en Train']):
    plt.text(index - 0.2, value + 0.01, f'{value:.2f}', color='blue', ha="center")
for index, value in enumerate(resultados['Exactitud en Test']):
    plt.text(index + 0.2, value + 0.01, f'{value:.2f}', color='red', ha="center")

plt.show()


  
```







```

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer  # Para manejar los NaN
import warnings

# Desactivar advertencias
warnings.filterwarnings(action='ignore')

# Cargar los datos (reemplaza 'df' con tu DataFrame)
df = datos

# 1. Preprocesamiento de datos
# Convertir variables categóricas a numéricas con Label Encoding
label_encoder = LabelEncoder()
categorical_columns = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18']

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Definir las variables independientes (X) y dependiente (y)
X = df[['Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'Education', 'EducationField',
        'EmployeeCount', 'EmployeeID', 'Gender', 'JobLevel', 'JobRole', 'MaritalStatus', 'MonthlyIncome',
        'NumCompaniesWorked', 'Over18', 'PercentSalaryHike', 'StandardHours', 'StockOptionLevel',
        'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager']]
y = df['Attrition']

# Manejar valores faltantes (NaN) usando SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# 2. Entrenar y evaluar los modelos con validación cruzada

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
acc_log_reg = accuracy_score(y_test, y_pred_log_reg)

# XGBoost con ajuste de hiperparámetros
xgboost_model = xgb.XGBClassifier(eval_metric='logloss', max_depth=4, learning_rate=0.1, n_estimators=100, subsample=0.8)
xgboost_model.fit(X_train, y_train)
y_pred_xgb = xgboost_model.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)

# Random Forest con ajuste de hiperparámetros
random_forest = RandomForestClassifier(random_state=42, max_depth=10, n_estimators=100, min_samples_split=10, min_samples_leaf=4)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

# 3. Mostrar resultados en una tabla
resultados = pd.DataFrame({
    'Modelo': ['Logistic Regression', 'XGBoost', 'RandomForest'],
    'Exactitud (Accuracy)': [acc_log_reg, acc_xgb, acc_rf]
})

print("\nResultados de los modelos:")
print(resultados)

# 4. Verificar sobreajuste usando Cross-Validation
# Para Logistic Regression
log_reg_cv = cross_val_score(log_reg, X_train, y_train, cv=5, scoring='accuracy').mean()
# Para XGBoost
xgboost_cv = cross_val_score(xgboost_model, X_train, y_train, cv=5, scoring='accuracy').mean()
# Para Random Forest
random_forest_cv = cross_val_score(random_forest, X_train, y_train, cv=5, scoring='accuracy').mean()

print(f"\nExactitud promedio con cross-validation (Logistic Regression): {log_reg_cv:.3f}")
print(f"Exactitud promedio con cross-validation (XGBoost): {xgboost_cv:.3f}")
print(f"Exactitud promedio con cross-validation (Random Forest): {random_forest_cv:.3f}")

# 5. Visualizar las matrices de confusión

# Función para mostrar la matriz de confusión
def mostrar_matriz_confusion(y_test, y_pred, model_name):
    plt.figure(figsize=(12,8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.xlabel('Predicciones')
    plt.ylabel('Valores Reales')
    plt.show()

# Matrices de confusión para cada modelo
mostrar_matriz_confusion(y_test, y_pred_log_reg, "Logistic Regression")
mostrar_matriz_confusion(y_test, y_pred_xgb, "XGBoost")
mostrar_matriz_confusion(y_test, y_pred_rf, "RandomForest")

# 6. Visualización del rendimiento de los modelos con etiquetas en las barras
plt.figure(figsize=(12, 8))
barplot = sns.barplot(x='Modelo', y='Exactitud (Accuracy)', data=resultados, hue='Modelo', palette='viridis', dodge=False)
plt.legend([],[], frameon=False)  # Elimina la leyenda si no se necesita
plt.title('Comparación de Exactitud entre Modelos')
plt.ylabel('Exactitud')

# Agregar etiquetas de los valores sobre las barras
for index, value in enumerate(resultados['Exactitud (Accuracy)']):
    barplot.text(index, value + 0.01, f'{value:.2f}', color='black', ha="center")

plt.show()



  
```




















**Resultados y Conclusiones**


Durante el desarrollo del proyecto se utilizó el proceso de desarrollo del modelo de machine learning, desde la recopilación y preparación de los datos hasta la implementación y evaluación del modelo.
El análisis inicial de base de datos fue Dividir la base en train y test, luego se entrenaron 3 modelos de regresión (Logistica, XG-Bosst Y RandomForest), en dos oportunidades, para comparar la exactitud de cada algoritmo en la base de test. Cuyos resultados fueron los siguientes:
En el primera iteración: XGBoost y Random Forest tienen una precisión muy alta en este conjunto de datos, lo que sugiere que pueden estar sobreajustados a los datos de entrenamiento. Regresión Logística, siendo un modelo más simple, ofrece una precisión decente y es menos propenso a sobreajustar. Sin embargo, no captura tanta complejidad en los datos como XGBoost o Random Forest. Como la precisión fue más alta que en el conjunto de prueba, es necesario hacer una validación cruzada.
Segunda iteración : XGBoost es el mejor modelo hasta ahora (88.32%), y pequeños ajustes en los hiperparámetros podrían mejorar aún más su rendimiento. Logistic Regression es un modelo más sencillo, pero su desempeño es consistente y confiable. Random Forest es competitivo, pero no supera a XGBoost en este caso, aunque tiene un buen equilibrio y generaliza bien.


Seguir ajustando hiperparámetros en XGBoost y Random Forest para intentar mejorar la generalización. Considerar reducir ligeramente la complejidad del modelo para asegurarte de que no esté memorizando partes del conjunto de entrenamiento.


**Limitaciones/Próximos Pasos**
Sin observaciones.

**Enlaces de interés**



[Proyecto 5 Profundización: Machine learning](https://colab.research.google.com/drive/1XaPDQbOWlvFNajqSp4p2xgJhtBkd5-Aa?usp=sharing)


[Proyecto 5 Profundización: Machine learning](https://docs.google.com/presentation/d/1F82bOv815Vend0mR2ljCJm8XFyVwZF8jW1WdsM2dar4/edit?usp=drive_link)


[Presentación video](https://www.loom.com/share/5ad45658034b4d6d89a1f1db8e4c825f?sid=6b9d6243-e58e-454d-b12c-d3d42cee0651)

