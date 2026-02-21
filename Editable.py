import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

#Cargamos el csv 
datos_csv = pd.read_csv("C:\\Users\\jalvarezg1601\\Desktop\\ESCOM\\5TO SEMESTRE\Analítica y Visualización de Datos\\Prácticas\\Practica1y2_AlvarezGarciaJoseFrancisco_5AV1\\used_cars_data.csv")

#%%1. Análisis exploratorio de datos (EDA, Exploratory Data Analysis) 

#Datos generales
#a. Características de cada dimensión
info = datos_csv.info()

for col in datos_csv.columns:
    print(f"Columna: {col}")
    print(f"Tipo de dato: {datos_csv[col].dtype}")
    print(f"¿Es numérica?: {datos_csv[col].dtype in ['int64', 'float64']}")
    print('-' * 50)

#b. Desplegar los primeros 10 registros
primeros = datos_csv.head(10)
print(primeros)

#c. Desplegar los úlitmos 10 registros
ultimos = datos_csv.tail(10)
print(ultimos)

#Análisis numérico
#d. Estadísticas descriptivas (media, desviación estándar, mínimo, máximo, moda, rango, cuartiles)
estadistica_descriptiva = datos_csv.describe()
print(estadistica_descriptiva)

# Moda:
for col in datos_csv.select_dtypes(include=['int64', 'float64']).columns:
    print(f"Moda de {col}: {datos_csv[col].mode()[0]}")

# Rango (máximo - mínimo):
for col in datos_csv.select_dtypes(include=['int64', 'float64']).columns:
    print(f"Rango de {col}: {datos_csv[col].max() - datos_csv[col].min()}")

#e. Identificar la cantidad y porcentaje de valores nulos
# Cantidad de valores nulos
print(datos_csv.isnull().sum())

# Porcentaje de valores nulos
print(datos_csv.isnull().mean() * 100)

#f. Identificar la cantidad valores duplicados. 
print(f"Cantidad de valores duplicados: {datos_csv.duplicated().sum()}")

#Análisis gráfico
datos_csv_numericos = datos_csv.select_dtypes(include=['int64', 'float64']) #Guardamos únicamente las columnas numéricas
#g. Graficar la distribución de cada una de las dimensiones numéricas. 
for col in datos_csv_numericos.columns:
    plt.figure(figsize=(10, 8))
    sns.histplot(datos_csv_numericos[col], kde=True)
    plt.title(f'Distribución de {col}')
    plt.show()

#h. Comparar entre sí cada una de las dimensiones numéricas (análisis bivariado) con un gráfico del tipo “pairplot”. 
sns.pairplot(datos_csv_numericos)
plt.show()

#i. Realizar un mapa de calor para identificar la correlación entre todas las variables numericas (análisis multivariado). 
correlation_matrix = datos_csv_numericos.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Mapa de Calor de Correlación')
plt.show()

#j. Graficar la distribución de cada una de las dimensiones categóricas. 
for col in datos_csv.select_dtypes(include=['object']).columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=datos_csv[col])
    plt.title(f'Distribución de {col}')
    plt.xticks(rotation=45)
    plt.show()

#k. Comparar cada una de las dimensiones categóricas contra la dimensión "Precio":
precio_col = 'Price'  

for col in datos_csv.select_dtypes(include=['object']).columns:
    if col != precio_col:
        mean_price_by_category = datos_csv.groupby(col)[precio_col].mean().sort_values(ascending=False)
        plt.figure(figsize=(8, 6))
        mean_price_by_category.plot(kind='bar')
        plt.title(f'Comparación entre {col} y {precio_col}')
        plt.ylabel(f'Media de {precio_col}')
        plt.xticks(rotation=45)
        plt.show()
    

#%%2. Limpieza de los datos
datos_csv_limpieza = pd.read_csv("C:\\Users\\jalvarezg1601\\Desktop\\ESCOM\\5TO SEMESTRE\Analítica y Visualización de Datos\\Prácticas\\Practica1y2_AlvarezGarciaJoseFrancisco_5AV1\\used_cars_data.csv")
# a) Eliminar las dimensiones “S. No.” y “New Price”
datos_csv_limpieza.drop(columns=['S.No.', 'New_Price'], inplace=True)

# b) Sustituir el nombre en inglés de cada dimensión por su traducción en español
datos_csv_limpieza.rename(columns={
    'Name': 'Nombre',
    'Location': 'Ubicacion',
    'Year': 'Año',
    'Kilometers_Driven': 'Kilometraje',
    'Fuel_Type': 'Combustible',
    'Transmission': 'Transmision',
    'Owner_Type': 'Propietario',
    'Mileage' : 'Millaje',
    'Engine': 'Motor',
    'Power': 'Potencia',
    'Seats': 'Asientos',
    'Price': 'Precio'
}, inplace=True)

# c) Eliminar observaciones que tengan valores nulos en la dimensión "Asientos"
datos_csv_limpieza = datos_csv_limpieza.dropna(subset=['Asientos'])

# d) Sustituir valores nulos en la dimensión "Millaje" por la media de la dimensión
# Convertir "Millaje" a numérico, eliminando la unidad "kmpl"
datos_csv_limpieza['Millaje'] = datos_csv_limpieza['Millaje'].astype(str).str.replace(" kmpl", "", regex=True).str.replace(" km/kg", "", regex=True)
datos_csv_limpieza['Millaje'] = pd.to_numeric(datos_csv_limpieza['Millaje'], errors='coerce')  # Convierte a float, poniendo NaN en valores inválidos
# Reemplazar valores 0.0 por NaN para tratarlos como nulos
datos_csv_limpieza['Millaje'].replace(0.0, np.nan, inplace=True)
# Sustituir valores nulos en "Millaje" por la media (sin contar los nulos originales)
datos_csv_limpieza['Millaje'].fillna(datos_csv_limpieza['Millaje'].mean(), inplace=True)

# e) Sustituir valores nulos en la dimensión "Motor" por la moda de la dimensión
datos_csv_limpieza['Motor'] = datos_csv_limpieza['Motor'].astype(str).str.replace(" CC", "", regex=True)
datos_csv_limpieza['Motor'] = pd.to_numeric(datos_csv_limpieza['Motor'], errors='coerce')
# Reemplazar valores 0 por NaN para tratarlos como nulos
datos_csv_limpieza['Millaje'].replace(0, np.nan, inplace=True)
datos_csv_limpieza['Motor'] = datos_csv_limpieza['Motor'].fillna(datos_csv_limpieza['Motor'].mode()[0])

#%%3. Transformación de datos
# a) Sustituir valores en la dimensión "Nombre"
datos_csv_limpieza['Nombre'] = datos_csv_limpieza['Nombre'].str.replace('ISUZU', 'Isuzu', regex=True)
datos_csv_limpieza['Nombre'] = datos_csv_limpieza['Nombre'].str.replace('Mini', 'MiniCooper', regex=True)
datos_csv_limpieza['Nombre'] = datos_csv_limpieza['Nombre'].str.replace('Land', 'LandRover', regex=True)

# b) Eliminar caracteres no numéricos en "Millas por Galón", "Motor" y "Potencia"
#Millas y Motor ya han sido cambiados en el apartado 2.d y 2.e
datos_csv_limpieza["Potencia"] = datos_csv_limpieza["Potencia"].astype(str).str.replace(' bhp', '', regex=True).replace("null", np.nan)
datos_csv_limpieza["Potencia"] = pd.to_numeric(datos_csv_limpieza["Potencia"], errors='coerce')  # Convertir a float

# c) Redondear al número entero más cercano
columnas_a_limpiar = ["Millaje","Motor","Potencia"]
datos_csv_limpieza[columnas_a_limpiar] = datos_csv_limpieza[columnas_a_limpiar].applymap(lambda x: int(round(x)) if pd.notna(x) else x)# Mantiene NaN si existen

# d) Multiplicar por mil los valores en la dimensión "Precio"
datos_csv_limpieza['Precio'] = datos_csv_limpieza['Precio'] * 1000

# e) Obtener el logaritmo de "Precio" y "Kilómetros"
datos_csv_limpieza['Log_Precio'] = np.log(datos_csv_limpieza['Precio']) #Logaritmo en base natural e(~2.718)
datos_csv_limpieza['Log_Kilometros'] = np.log(datos_csv_limpieza['Kilometraje'])

#%%4. Ingeniería de caracteríticas 
# a) Crear la dimensión "Antigüedad"
año_actual = datetime.now().year
datos_csv_limpieza['Antiguedad'] = año_actual - datos_csv_limpieza['Año']

# b) Extraer "Marca" y "Modelo" desde la dimensión "Nombre"
datos_csv_limpieza[['Marca', 'Modelo']] = datos_csv_limpieza['Nombre'].str.split(' ', n=1, expand=True)
datos_csv_limpieza['Modelo'] = datos_csv_limpieza['Modelo'].str.replace(' ', '', regex=True)  # Eliminar espacios en blanco en "Modelo"

#%%5. Análisis exploratorio de datos posterior a etapas de limpieza, transformación e ingeniería de características 

#Datos generales
#a. Características de cada dimensión
info_limpieza = datos_csv_limpieza.info()
datos_csv_limpieza.fillna(np.nan)

for col in datos_csv_limpieza.columns:
    print(f"Columna: {col}")
    print(f"Tipo de dato: {datos_csv_limpieza[col].dtype}")
    print(f"¿Es numérica?: {datos_csv_limpieza[col].dtype in ['int64', 'float64']}")
    print('-' * 50)

#b. Desplegar los primeros 10 registros
primeros = datos_csv_limpieza.head(10)
print(primeros)

#c. Desplegar los úlitmos 10 registros
ultimos = datos_csv_limpieza.tail(10)
print(ultimos)

#Análisis numérico
#d. Estadísticas descriptivas (media, desviación estándar, mínimo, máximo, moda, rango, cuartiles)
estadistica_descriptiva = datos_csv_limpieza.describe()
print(estadistica_descriptiva)

# Moda:
for col in datos_csv_limpieza.select_dtypes(include=['int64', 'float64']).columns:
    print(f"Moda de {col}: {datos_csv_limpieza[col].mode()[0]}")

# Rango (máximo - mínimo):
for col in datos_csv_limpieza.select_dtypes(include=['int64', 'float64']).columns:
    print(f"Rango de {col}: {datos_csv_limpieza[col].max() - datos_csv_limpieza[col].min()}")

#e. Identificar la cantidad y porcentaje de valores nulos
# Cantidad de valores nulos
print(datos_csv_limpieza.isnull().sum())

# Porcentaje de valores nulos
print(datos_csv_limpieza.isnull().mean() * 100)

#f. Identificar la cantidad valores duplicados. 
print(f"Cantidad de valores duplicados: {datos_csv_limpieza.duplicated().sum()}")

#Análisis gráfico
datos_csv_numericos_limpios = datos_csv_limpieza.select_dtypes(include=['int64', 'float64']) #Guardamos únicamente las columnas numéricas
#g. Graficar la distribución de cada una de las dimensiones numéricas. 
for col in datos_csv_numericos_limpios.columns:
    plt.figure(figsize=(10, 8))
    sns.histplot(datos_csv_numericos_limpios[col], kde=True)
    plt.title(f'Distribución de {col}')
    plt.show()

#h. Comparar entre sí cada una de las dimensiones numéricas (análisis bivariado) con un gráfico del tipo “pairplot”. 
sns.pairplot(datos_csv_numericos_limpios)
plt.show()

#i. Realizar un mapa de calor para identificar la correlación entre todas las variables numericas (análisis multivariado). 
correlation_matrix = datos_csv_numericos_limpios.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Mapa de Calor de Correlación')
plt.show()

#j. Graficar la distribución de cada una de las dimensiones categóricas. 
for col in datos_csv_limpieza.select_dtypes(include=['object']).columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=datos_csv_limpieza[col])
    plt.title(f'Distribución de {col}')
    plt.xticks(rotation=45)
    plt.show()

#k. Comparar cada una de las dimensiones categóricas contra la dimensión "Precio":
precio_col = 'Precio'  

for col in datos_csv_limpieza.select_dtypes(include=['object']).columns:
    if col != precio_col:
        mean_price_by_category = datos_csv_limpieza.groupby(col)[precio_col].mean().sort_values(ascending=False)
        plt.figure(figsize=(8, 6))
        mean_price_by_category.plot(kind='bar')
        plt.title(f'Comparación entre {col} y {precio_col}')
        plt.ylabel(f'Media de {precio_col}')
        plt.xticks(rotation=45)
        plt.show()
