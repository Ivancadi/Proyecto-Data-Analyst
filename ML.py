# Importo librerias para trabajar nuestro
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# Importamos dataset
df = pd.read_csv(r'C:\Users\PERSONAL\Desktop\Data scince\HENRY\HENRY - Data scince\LABS\LABS 2\general.csv')

# Definimos los datos con los que se van a trabajar
df = df.iloc[:,1:5]

# Codificamos la variable Provincia
encoder = OrdinalEncoder()
categorical_data = df[['Provincia']]
encoded_data = encoder.fit_transform(categorical_data)

# La agregamos al df
df['Provincia'] = encoded_data
df['Provincia'] = df['Provincia'].astype(int)

''' Empezamos a definir nuetro modelo de ML'''

X = df[['Año', 'Trimestre', 'Provincia']]
y = df['Accesos por cada 100 hogares']

# Separamos los datos de entrenamiento y de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenamos el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Establecemos datos para ver como puede terminar el 2022
df_nuevo = df.iloc[72:168, 0:3]
df_nuevo['Año'] = 2022

# Predecimos el año 2022
prediccion_nueva = model.predict(df_nuevo)
df_nuevo['Accesos por cada 100 hogares'] = prediccion_nueva

# Establecemos datos para el año 2023
df_nuevo2023 = df.iloc[72:168, 0:3]
df_nuevo2023['Año'] = 2023

# Predecimos el año 2023
prediccion_nueva2023 = model.predict(df_nuevo2023)
df_nuevo2023['Accesos por cada 100 hogares'] = prediccion_nueva2023

# Concatenamos los datos que han sido predictivos con el dataframe original
df_concat = pd.concat([df_nuevo2023, df_nuevo, df.iloc[72:,:]])

# Importamos librerias para graficar 
import matplotlib.pyplot as plt 

# Agrupamos por año para graficar
df_agrupado = df_concat.groupby(['Año'], as_index= False).sum()

# Graficamos 
x = df_agrupado['Año']
y = round(df_agrupado['Accesos por cada 100 hogares'],0)

fig, ax = plt.subplots()
ax.plot(x, y, marker='o')

for i, j in zip(x, y):
    ax.annotate(str(j), xy=(i, j))

plt.xlabel('Años')
plt.ylabel('Total acceso por 100 hogares')
plt.title('Total de accesos por cada 100 hogares por año')

plt.show()
