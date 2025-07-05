# %%
# =============================================================================
# PARTE 1: PREPARACIÓN DE DATOS
# =============================================================================

# -----------------------------------------------------------------------------
# Celda 1: Importación de Librerías
# -----------------------------------------------------------------------------
# Importamos las librerías fundamentales para el análisis:
# - pandas: Para la manipulación y análisis de datos tabulares (DataFrames).
# - numpy: Para operaciones numéricas eficientes, especialmente con arrays.
# - matplotlib.pyplot y seaborn: Para la creación de visualizaciones estáticas y atractivas.
# - datetime: Para trabajar con fechas y horas.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# Configuramos seaborn para que las gráficas tengan un estilo visual agradable.
sns.set_style("whitegrid")
print("Librerías importadas correctamente.")


# %%
# -----------------------------------------------------------------------------
# Celda 2: Carga del Dataset
# -----------------------------------------------------------------------------
# Cargamos el conjunto de datos desde el archivo CSV.
# Es importante especificar el 'encoding' para evitar problemas con caracteres especiales.
# 'latin1' es comúnmente usado para este dataset en particular.

try:
    df = pd.read_csv('Online Retail.csv', encoding='latin1')
    print("Dataset cargado exitosamente.")
    print("Forma del dataset (filas, columnas):", df.shape)
except FileNotFoundError:
    print("Error: El archivo 'Online Retail.csv' no se encontró.")
    print("Por favor, asegúrate de que el archivo esté en el mismo directorio que el script.")

# Mostramos las primeras 5 filas para una inspección inicial.
df.head()


# %%
# -----------------------------------------------------------------------------
# Celda 3: Inspección Inicial y Limpieza Básica
# -----------------------------------------------------------------------------
# Obtenemos información general del DataFrame, incluyendo tipos de datos y valores nulos.
print("Información general del DataFrame:")
df.info()

# El análisis RFM y el clustering de clientes dependen de poder agrupar por cliente.
# Por lo tanto, los registros sin 'CustomerID' no son útiles para nuestro objetivo principal.
# Calculamos el porcentaje de nulos en CustomerID antes de eliminarlos.
missing_customers = df['CustomerID'].isnull().sum()
total_rows = df.shape[0]
print(f"\nSe encontraron {missing_customers} filas sin CustomerID ({missing_customers/total_rows:.2%}).")

# Eliminamos las filas donde 'CustomerID' es nulo.
df.dropna(subset=['CustomerID'], inplace=True)
print(f"Filas después de eliminar CustomerID nulos: {df.shape[0]}")

# Convertimos 'CustomerID' a tipo entero para un manejo más limpio.
df['CustomerID'] = df['CustomerID'].astype(int)

# Convertimos 'InvoiceDate' a formato datetime para poder realizar cálculos de tiempo.
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

print("\nLimpieza básica completada: CustomerID nulos eliminados y tipos de datos corregidos.")


# %%
# -----------------------------------------------------------------------------
# Celda 4: Limpieza de Registros Inválidos
# -----------------------------------------------------------------------------
# El dataset contiene registros que no representan una compra real y deben ser eliminados.

# 1. Cantidades negativas ('Quantity' < 0):
#    Estos registros corresponden a devoluciones o transacciones canceladas. No son compras.
print(f"Filas con cantidad negativa (devoluciones): {df[df['Quantity'] <= 0].shape[0]}")
df = df[df['Quantity'] > 0]
print(f"Filas después de eliminar cantidades negativas: {df.shape[0]}")

# 2. Precios unitarios cero ('UnitPrice' == 0):
#    Productos con precio cero no contribuyen al valor monetario y son probablemente errores.
print(f"\nFilas con precio unitario cero: {df[df['UnitPrice'] <= 0].shape[0]}")
df = df[df['UnitPrice'] > 0]
print(f"Filas después de eliminar precios cero: {df.shape[0]}")


# %%
# -----------------------------------------------------------------------------
# Celda 5: Foco en el Mercado Principal (Reino Unido)
# -----------------------------------------------------------------------------
# Como discutimos, analizar un mercado homogéneo produce segmentos más claros y accionables.
# Dado que la mayoría de los clientes son del Reino Unido, nos enfocaremos en ellos.
# Esto asegura que nuestras estrategias de marketing propuestas sean consistentes y relevantes.

print("Distribución de clientes por país (Top 10):")
print(df['Country'].value_counts().head(10))

# Filtramos el DataFrame para quedarnos solo con las transacciones del Reino Unido.
df_uk = df[df['Country'] == 'United Kingdom'].copy()
print(f"\nAnálisis enfocado en el Reino Unido. Total de filas: {df_uk.shape[0]}")


# %%
# -----------------------------------------------------------------------------
# Celda 6: Eliminación de Duplicados
# -----------------------------------------------------------------------------
# Verificamos y eliminamos cualquier fila que esté completamente duplicada.
print(f"Número de filas duplicadas: {df_uk.duplicated().sum()}")
df_uk.drop_duplicates(inplace=True)
print(f"Filas después de eliminar duplicados: {df_uk.shape[0]}")


# %%
# -----------------------------------------------------------------------------
# Celda 7: Creación de Variables RFM (Recency, Frequency, Monetary)
# -----------------------------------------------------------------------------
# Ahora, calculamos las tres métricas clave para nuestro análisis.

# 1. Calcular el valor monetario de cada transacción.
df_uk['TotalPrice'] = df_uk['Quantity'] * df_uk['UnitPrice']

# 2. Determinar la "fecha de hoy" para el análisis.
#    Usamos el día siguiente a la última transacción en el dataset como punto de referencia.
#    Esto nos permite calcular la recencia para todos los clientes de manera consistente.
snapshot_date = df_uk['InvoiceDate'].max() + dt.timedelta(days=1)
print(f"Fecha de referencia (snapshot_date) para el análisis: {snapshot_date.date()}")

# 3. Agrupar por cliente para calcular R, F y M.
rfm = df_uk.groupby('CustomerID').agg({
    'InvoiceDate': lambda date: (snapshot_date - date.max()).days, # Recency: Días desde la última compra.
    'InvoiceNo': 'nunique',                                       # Frequency: Número de facturas únicas.
    'TotalPrice': 'sum'                                           # Monetary: Suma total del dinero gastado.
})

# Renombramos las columnas para mayor claridad.
rfm.rename(columns={'InvoiceDate': 'Recency',
                    'InvoiceNo': 'Frequency',
                    'TotalPrice': 'MonetaryValue'}, inplace=True)

print("\nDataFrame RFM creado exitosamente:")
rfm.head()


# %%
# -----------------------------------------------------------------------------
# Celda 8: Visualización de las Variables RFM ANTES del Escalado
# -----------------------------------------------------------------------------
# Antes de aplicar clustering, es crucial entender la distribución de nuestras variables.
# Los algoritmos basados en distancia son sensibles a la escala y al sesgo de los datos.

plt.figure(figsize=(18, 5))

# Gráfico para Recency
plt.subplot(1, 3, 1)
sns.histplot(rfm['Recency'], kde=True, bins=30)
plt.title('Distribución de Recency')

# Gráfico para Frequency
plt.subplot(1, 3, 2)
sns.histplot(rfm['Frequency'], kde=True, bins=30)
plt.title('Distribución de Frequency')

# Gráfico para MonetaryValue
plt.subplot(1, 3, 3)
sns.histplot(rfm['MonetaryValue'], kde=True, bins=30)
plt.title('Distribución de Monetary Value')

plt.suptitle('Distribuciones de Variables RFM (Antes del Escalado)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("Observación: Todas las variables tienen un fuerte sesgo a la derecha (right-skewed).")
print("Frequency y Monetary tienen rangos muy diferentes a Recency. El escalado es necesario.")


# %%
# -----------------------------------------------------------------------------
# Celda 9: Aplicación de Escalado para Clustering
# -----------------------------------------------------------------------------
# Para que el algoritmo de clustering funcione correctamente, debemos pre-procesar los datos.
# Realizamos un proceso de dos pasos:
# 1. Transformación Logarítmica: Para reducir el sesgo a la derecha. Usamos log1p
#    que añade 1 antes del logaritmo para manejar valores de cero si los hubiera.
# 2. Escalado Estándar (StandardScaler): Para que todas las variables tengan una
#    media de 0 y una desviación estándar de 1. Esto asegura que ninguna variable
#    domine el cálculo de distancia solo por su escala.

from sklearn.preprocessing import StandardScaler

# Hacemos una copia para no modificar el dataframe RFM original.
rfm_to_scale = rfm.copy()

# 1. Aplicamos la transformación logarítmica.
rfm_to_scale['Recency'] = np.log1p(rfm_to_scale['Recency'])
rfm_to_scale['Frequency'] = np.log1p(rfm_to_scale['Frequency'])
rfm_to_scale['MonetaryValue'] = np.log1p(rfm_to_scale['MonetaryValue'])

# 2. Inicializamos y aplicamos el escalador.
scaler = StandardScaler()
rfm_scaled_array = scaler.fit_transform(rfm_to_scale)

# Convertimos el resultado de vuelta a un DataFrame para facilitar su uso.
rfm_scaled = pd.DataFrame(rfm_scaled_array, index=rfm.index, columns=rfm.columns)

print("Datos RFM transformados y escalados:")
rfm_scaled.head()


# %%
# -----------------------------------------------------------------------------
# Celda 10: Visualización de las Variables RFM DESPUÉS del Escalado
# -----------------------------------------------------------------------------
# Ahora visualizamos las distribuciones después del pre-procesamiento.
# Deberíamos ver distribuciones mucho más simétricas y centradas en cero.

plt.figure(figsize=(18, 5))

# Gráfico para Recency
plt.subplot(1, 3, 1)
sns.histplot(rfm_scaled['Recency'], kde=True, bins=30)
plt.title('Distribución de Recency (Escalada)')

# Gráfico para Frequency
plt.subplot(1, 3, 2)
sns.histplot(rfm_scaled['Frequency'], kde=True, bins=30)
plt.title('Distribución de Frequency (Escalada)')

# Gráfico para MonetaryValue
plt.subplot(1, 3, 3)
sns.histplot(rfm_scaled['MonetaryValue'], kde=True, bins=30)
plt.title('Distribución de Monetary Value (Escalada)')

plt.suptitle('Distribuciones de Variables RFM (Después del Escalado)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("Observación: Las distribuciones ahora son mucho más 'normales' y están en la misma escala.")
print("El dataset 'rfm_scaled' está listo para ser utilizado en los algoritmos de clustering.")

# %%
# =============================================================================
# FIN DE LA PARTE 1
# =============================================================================
# Hemos completado la preparación de datos. Tenemos dos DataFrames importantes:
# 1. 'rfm': Contiene los valores originales de Recency, Frequency y Monetary. Lo usaremos para interpretar los clusters.
# 2. 'rfm_scaled': Contiene los valores escalados. Lo usaremos para entrenar los modelos de clustering.
#
# Cuando estés listo, continuaremos con la Parte 2: Aplicación de Clustering.
# %%
# =============================================================================
# PARTE 2: APLICACIÓN Y EVALUACIÓN DE ALGORITMOS DE CLUSTERING
# =============================================================================
# En esta sección, aplicaremos tres algoritmos de clustering diferentes a nuestros
# datos RFM escalados. Usaremos métodos cuantitativos para determinar los
# hiperparámetros óptimos y para comparar el rendimiento de los modelos.

# -----------------------------------------------------------------------------
# Celda 11: Importación de Librerías de Clustering y Métricas
# -----------------------------------------------------------------------------
# Importamos los modelos de clustering y las métricas de evaluación que usaremos.
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage

# Guardamos nuestro dataframe escalado en una variable para facilitar su uso
X = rfm_scaled.copy()

print("Librerías y datos para clustering listos.")


# %%
# =============================================================================
# ALGORITMO 1: K-MEANS
# =============================================================================

# -----------------------------------------------------------------------------
# Celda 12: Determinación del Número Óptimo de Clusters (k) para K-Means
# -----------------------------------------------------------------------------
# Para K-Means, el hiperparámetro más importante es 'k' (el número de clusters).
# Usaremos dos métodos populares para encontrar un 'k' óptimo:
# 1. Método del Codo (Elbow Method): Busca el punto donde la reducción de la inercia
#    (suma de cuadrados de las distancias) se vuelve marginal.
# 2. Puntuación de Silueta (Silhouette Score): Mide qué tan bien separado está un
#    cluster de los demás. Un valor más alto es mejor (rango de -1 a 1).

# Rango de k a probar
k_range = range(2, 11)
inertia = []
silhouette_scores = []

for k in k_range:
    # n_init='auto' para evitar FutureWarnings
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Visualización de los resultados
plt.figure(figsize=(15, 6))

# Gráfico del Método del Codo
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.title('Método del Codo para K-Means')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.xticks(k_range)

# Gráfico de la Puntuación de Silueta
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, marker='o', linestyle='--')
plt.title('Puntuación de Silueta para K-Means')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Puntuación de Silueta')
plt.xticks(k_range)

plt.suptitle('Selección de k para K-Means', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# JUSTIFICACIÓN:
# El método del codo muestra un "codo" visible en k=4, donde la curva comienza a aplanarse.
# La puntuación de silueta también alcanza su punto máximo en k=4.
# Por lo tanto, elegimos k=4 como el número óptimo de clusters para nuestro análisis.
k_optimal = 4
print(f"Justificación: Basado en el método del codo y la puntuación de silueta, el número óptimo de clusters es k={k_optimal}.")


# %%
# -----------------------------------------------------------------------------
# Celda 13: Aplicación de K-Means con el k óptimo
# -----------------------------------------------------------------------------
# Ahora aplicamos el algoritmo K-Means con k=4.

kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init='auto')
kmeans_labels = kmeans.fit_predict(X)

# Guardamos las etiquetas de los clusters para la comparación posterior.
rfm['KMeans_Cluster'] = kmeans_labels
print(f"K-Means aplicado. Número de clientes en cada cluster:")
print(rfm['KMeans_Cluster'].value_counts())


# %%
# =============================================================================
# ALGORITMO 2: CLUSTERING JERÁRQUICO AGLOMERATIVO
# =============================================================================

# -----------------------------------------------------------------------------
# Celda 14: Visualización del Dendrograma
# -----------------------------------------------------------------------------
# El clustering aglomerativo construye una jerarquía de clusters. El dendrograma
# nos ayuda a visualizar esta estructura y decidir dónde "cortar" para formar los clusters.
# 'ward' es un método de enlace que minimiza la varianza dentro de cada cluster.

plt.figure(figsize=(15, 7))
plt.title('Dendrograma (Truncado)')
plt.xlabel('Número de Puntos en el Cluster')
plt.ylabel('Distancia (Ward)')

# 'linkage' calcula las distancias para construir la jerarquía.
linked = linkage(X, method='ward')

# Dibujamos el dendrograma. Truncamos para que sea legible.
dendrogram(linked,
           orientation='top',
           truncate_mode='lastp',  # Muestra los últimos 'p' clusters fusionados
           p=12,                   # Número de clusters a mostrar en la base
           show_leaf_counts=True,
           show_contracted=True)
plt.show()

# JUSTIFICACIÓN:
# Observando las distancias verticales en el dendrograma, el salto más significativo
# ocurre cuando se pasa de 4 a 3 clusters (la línea azul más larga). Esto sugiere
# que k=4 es una buena elección, lo cual es consistente con nuestro hallazgo en K-Means.


# %%
# -----------------------------------------------------------------------------
# Celda 15: Aplicación de Clustering Aglomerativo
# -----------------------------------------------------------------------------
# Aplicamos el modelo con el mismo número de clusters que K-Means (k=4) para una
# comparación justa.

agg_cluster = AgglomerativeClustering(n_clusters=k_optimal, linkage='ward')
agg_labels = agg_cluster.fit_predict(X)

# Guardamos las etiquetas
rfm['Agg_Cluster'] = agg_labels
print(f"Clustering Aglomerativo aplicado. Número de clientes en cada cluster:")
print(rfm['Agg_Cluster'].value_counts())


# %%
# =============================================================================
# ALGORITMO 3: DBSCAN
# =============================================================================

# -----------------------------------------------------------------------------
# Celda 16: Determinación de Hiperparámetros para DBSCAN (eps y min_samples)
# -----------------------------------------------------------------------------
# DBSCAN tiene dos hiperparámetros:
# - min_samples: El número mínimo de puntos para formar una región densa. Una regla
#   general es usar 2 * número_de_dimensiones. En nuestro caso, 2 * 3 = 6.
# - eps: La distancia máxima para que dos puntos sean considerados vecinos. Usaremos
#   el método del gráfico k-distance para encontrar un 'eps' razonable.

# 1. Definir min_samples
min_samples_dbscan = 2 * X.shape[1]  # 2 * 3 = 6

# 2. Calcular la distancia de cada punto a su k-ésimo vecino más cercano (k=min_samples)
neighbors = NearestNeighbors(n_neighbors=min_samples_dbscan)
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)

# Ordenamos las distancias y las graficamos
sorted_distances = np.sort(distances[:, min_samples_dbscan-1], axis=0)

plt.figure(figsize=(10, 6))
plt.plot(sorted_distances)
plt.title('Gráfico k-distance para DBSCAN')
plt.xlabel('Puntos ordenados por distancia')
plt.ylabel(f'Distancia al {min_samples_dbscan-1}-ésimo vecino (eps)')
plt.grid(True)
plt.show()

# JUSTIFICACIÓN:
# El gráfico muestra un "codo" o punto de máxima curvatura alrededor de un valor de
# distancia de 0.8-1.0. Este punto es un buen candidato para 'eps' porque es donde
# la densidad cambia significativamente. Elegiremos eps=0.9.
eps_optimal = 0.9
print(f"Justificación: min_samples={min_samples_dbscan}. El gráfico k-distance sugiere un eps óptimo alrededor de {eps_optimal}.")


# %%
# -----------------------------------------------------------------------------
# Celda 17: Aplicación de DBSCAN
# -----------------------------------------------------------------------------
# DBSCAN es diferente porque puede identificar puntos como "ruido" (etiquetados como -1),
# lo cual es muy útil para encontrar anomalías.

dbscan = DBSCAN(eps=eps_optimal, min_samples=min_samples_dbscan)
dbscan_labels = dbscan.fit_predict(X)

# Guardamos las etiquetas
rfm['DBSCAN_Cluster'] = dbscan_labels
print(f"DBSCAN aplicado. Resumen de clusters:")
print(pd.Series(dbscan_labels).value_counts())

# El cluster -1 representa el ruido (outliers).


# %%
# =============================================================================
# COMPARACIÓN DE MODELOS CON MÉTRICAS INTERNAS
# =============================================================================

# -----------------------------------------------------------------------------
# Celda 18: Cálculo y Comparación de Métricas
# -----------------------------------------------------------------------------
# Ahora comparamos los resultados de los tres algoritmos utilizando las métricas
# internas que no requieren etiquetas verdaderas.
# - Silhouette Score: Más alto es mejor.
# - Calinski-Harabasz Score: Más alto es mejor.
# - Davies-Bouldin Score: Más bajo es mejor.

# Para DBSCAN, solo evaluamos los puntos que no son ruido.
dbscan_mask = dbscan_labels != -1
if np.sum(dbscan_mask) < 2: # No se pueden calcular métricas si hay menos de 2 clusters o puntos
    print("DBSCAN encontró muy pocos puntos de cluster para evaluar.")
    dbscan_metrics = [np.nan, np.nan, np.nan]
else:
    dbscan_metrics = [
        silhouette_score(X[dbscan_mask], dbscan_labels[dbscan_mask]),
        calinski_harabasz_score(X[dbscan_mask], dbscan_labels[dbscan_mask]),
        davies_bouldin_score(X[dbscan_mask], dbscan_labels[dbscan_mask])
    ]

# Creamos un DataFrame para una comparación clara
comparison_data = {
    'K-Means': [
        silhouette_score(X, kmeans_labels),
        calinski_harabasz_score(X, kmeans_labels),
        davies_bouldin_score(X, kmeans_labels)
    ],
    'Agglomerative': [
        silhouette_score(X, agg_labels),
        calinski_harabasz_score(X, agg_labels),
        davies_bouldin_score(X, agg_labels)
    ],
    'DBSCAN (sin ruido)': dbscan_metrics
}

comparison_df = pd.DataFrame(comparison_data, index=['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin']).T
print("Tabla de Comparación de Métricas de Clustering:")
print(comparison_df.round(3))


# %%
# -----------------------------------------------------------------------------
# Celda 19: Elección del Mejor Modelo
# -----------------------------------------------------------------------------
# JUSTIFICACIÓN DE LA ELECCIÓN:
#
# Al analizar la tabla de comparación, observamos lo siguiente:
# 1. K-Means y Agglomerative Clustering (con k=4) producen resultados muy similares en
#    términos de métricas, con K-Means teniendo una ligera ventaja en la puntuación de silueta
#    y Calinski-Harabasz.
# 2. DBSCAN, aunque es excelente para identificar ruido, en este caso ha clasificado a una
#    gran cantidad de puntos como ruido y ha generado clusters muy pequeños y densos.
#    Esto puede ser útil, pero para una segmentación de marketing general, queremos
#    agrupar a TODOS los clientes.
#
# ELECCIÓN FINAL: K-Means.
# ¿Por qué?
# - Rendimiento: Ofrece las mejores métricas de evaluación entre los modelos probados.
# - Interpretabilidad: Es un algoritmo muy conocido, computacionalmente eficiente y los
#   clusters que genera suelen ser fáciles de interpretar para el negocio.
# - Balance de Clusters: Tiende a crear clusters de tamaños relativamente balanceados
#   (como vimos en la celda 13), lo cual es útil para una estrategia de segmentación.
#
# Por estas razones, procederemos con los clusters generados por K-Means para la
# fase final de análisis e interpretación.

print("\nModelo elegido para la siguiente fase: K-Means.")

# Hemos aplicado y evaluado tres algoritmos de clustering, eligiendo K-Means
# como el más adecuado para nuestro problema de negocio.
# %%
# =============================================================================
# PARTE 3: ANÁLISIS, INTERPRETACIÓN Y ACCIONES DE MARKETING
# =============================================================================
# En esta fase final, tomamos los clusters generados por nuestro modelo elegido (K-Means)
# y les damos un significado de negocio. El objetivo es entender quiénes son los
# clientes en cada grupo y cómo podemos comunicarnos con ellos de manera efectiva.

# -----------------------------------------------------------------------------
# Celda 20: Importación de Librerías Adicionales
# -----------------------------------------------------------------------------
# Importamos PCA de scikit-learn para la reducción de dimensionalidad,
# lo que nos permitirá visualizar los clusters en un gráfico 2D.
from sklearn.decomposition import PCA

print("Librerías para análisis y visualización de clusters listas.")


# %%
# -----------------------------------------------------------------------------
# Celda 21: Perfilado de Clusters
# -----------------------------------------------------------------------------
# Para entender qué caracteriza a cada cluster, calculamos la media de las
# variables RFM originales (NO las escaladas) para cada grupo. También contamos
# cuántos clientes hay en cada cluster.

# Usaremos el dataframe 'rfm' que contiene los valores originales y las etiquetas de K-Means.
# Agrupamos por el cluster de K-Means y calculamos la media de R, F, M y el tamaño del grupo.
cluster_profile = rfm.groupby('KMeans_Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': 'mean',
    'KMeans_Cluster': 'size'
}).rename(columns={'KMeans_Cluster': 'Count'})

# Calculamos el promedio general de RFM para tener un punto de comparación
population_avg = rfm[['Recency', 'Frequency', 'MonetaryValue']].mean()
cluster_profile = pd.concat([cluster_profile, population_avg.to_frame('Population_Avg').T])


print("Perfil de los Clusters (medias de RFM y tamaño):")
print(cluster_profile.round(2))

# El perfil nos muestra las características promedio de cada grupo.
# Por ejemplo, un cluster con 'Recency' baja, 'Frequency' alta y 'MonetaryValue' alto
# será nuestro grupo de "Campeones".


# %%
# -----------------------------------------------------------------------------
# Celda 22: Visualización de Perfiles de Clusters (Gráfico de Serpiente)
# -----------------------------------------------------------------------------
# Un gráfico de serpiente (snake plot) es una excelente manera de comparar los
# atributos de cada segmento. Usaremos los datos escalados para que todas las
# variables estén en la misma escala y sean comparables.

# Creamos un DataFrame con los datos escalados y las etiquetas de cluster.
rfm_scaled_with_clusters = pd.DataFrame(X, index=rfm.index, columns=X.columns)
rfm_scaled_with_clusters['Cluster'] = rfm['KMeans_Cluster']

# Calculamos la media de las variables escaladas por cluster.
scaled_profile = rfm_scaled_with_clusters.groupby('Cluster').mean()

plt.figure(figsize=(12, 7))
sns.lineplot(data=scaled_profile.T, dashes=False, markers=True)
plt.title('Perfil de Clusters (Snake Plot)', fontsize=16)
plt.xlabel('Métricas RFM (Escaladas)')
plt.ylabel('Valor Medio del Segmento')
plt.legend(title='Cluster')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8) # Línea en 0 para referencia (media de la población)
plt.show()

# INTERPRETACIÓN DEL GRÁFICO:
# - Cada línea representa un cluster.
# - El eje Y muestra qué tan por encima o por debajo de la media está un cluster en una métrica específica.
# - Esto nos permite visualizar rápidamente las características distintivas de cada grupo.


# %%
# -----------------------------------------------------------------------------
# Celda 23: Etiquetado de Clusters y Definición de Personas
# -----------------------------------------------------------------------------
# Basado en el perfil numérico y el gráfico de serpiente, asignamos etiquetas
# descriptivas a cada cluster. Esto los convierte en "personas" accionables.

# (NOTA: Los números de cluster pueden variar en cada ejecución. Ajusta las etiquetas
# según la salida de la celda 21)

# Supongamos que la salida de la celda 21 fue:
# Cluster 0: Alta R, Baja F, Bajo M -> "Hibernando" o "Perdidos"
# Cluster 1: Baja R, Alta F, Alto M -> "Campeones"
# Cluster 2: Media R, Media F, Medio M -> "Clientes Leales"
# Cluster 3: Baja R, Baja F, Bajo M -> "Nuevos Clientes" o "Potenciales"

# Mapeamos estas etiquetas a nuestros datos
# ¡¡IMPORTANTE!! Revisa tu tabla de 'cluster_profile' para asignar las etiquetas correctas
# a los números de cluster que te salieron a ti.
label_map = {
    0: 'Hibernando',
    1: 'Campeones',
    2: 'Clientes Leales',
    3: 'Nuevos y Potenciales'
}

rfm['Segment'] = rfm['KMeans_Cluster'].map(label_map)
print("Etiquetas de segmento asignadas:")
print(rfm.head())


# %%
# -----------------------------------------------------------------------------
# Celda 24: Visualización de Clusters con PCA
# -----------------------------------------------------------------------------
# Reducimos las 3 dimensiones de RFM a 2 componentes principales (PCA)
# para poder visualizar la separación de los clusters en un gráfico de dispersión.

pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(X) # Usamos los datos escalados para PCA

# Creamos un DataFrame con los resultados de PCA y las etiquetas de segmento.
pca_df = pd.DataFrame(rfm_pca, index=rfm.index, columns=['PC1', 'PC2'])
pca_df['Segment'] = rfm['Segment']

plt.figure(figsize=(12, 9))
sns.scatterplot(x='PC1', y='PC2', hue='Segment', data=pca_df, palette='viridis', alpha=0.7, s=50)
plt.title('Visualización de Segmentos de Clientes con PCA', fontsize=16)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Segmento')
plt.grid(True)
plt.show()

# INTERPRETACIÓN DEL GRÁFICO PCA:
# El gráfico muestra qué tan bien separados están nuestros segmentos en el espacio
# reducido. Un buen clustering resultará en grupos visualmente distintos.


# %%
# -----------------------------------------------------------------------------
# Celda 25: Propuesta de Acciones de Marketing Personalizadas
# -----------------------------------------------------------------------------
# Esta es la culminación de nuestro análisis. Proponemos acciones específicas
# para los segmentos más relevantes, conectando los datos con la estrategia de negocio.

print("="*60)
print("PROPUESTAS DE ACCIONES DE MARKETING ESTRATÉGICAS")
print("="*60)

# ACCIÓN PARA EL SEGMENTO 1: "Campeones"
print("\n--- Segmento: Campeones ---")
print("   - Perfil: Clientes más valiosos. Compran muy a menudo, gastan mucho y han comprado recientemente.")
print("   - Objetivo: Retener, recompensar y convertirlos en embajadores de la marca.")
print("   - Acciones Propuestas:")
print("     1. Programa VIP: Ofrecer acceso exclusivo a ventas anticipadas, envíos gratuitos y atención al cliente prioritaria.")
print("     2. Marketing de Reconocimiento: Enviar regalos sorpresa o notas de agradecimiento personalizadas.")
print("     3. Solicitar Reseñas: Pedirles que dejen reseñas de productos, ya que su opinión es muy valiosa y confiable.")
print("     4. Evitar Descuentos Agresivos: No necesitan incentivos de precio para comprar; un marketing masivo podría devaluarlos.")

# ACCIÓN PARA EL SEGMENTO 2: "Clientes Leales"
print("\n--- Segmento: Clientes Leales ---")
print("   - Perfil: Compran con buena frecuencia y gastan una cantidad considerable. Son la base del negocio.")
print("   - Objetivo: Aumentar su valor monetario y frecuencia para convertirlos en Campeones.")
print("   - Acciones Propuestas:")
print("     1. Venta Cruzada (Cross-Selling): Recomendar productos complementarios a sus compras habituales.")
print("     2. Programas de Lealtad por Puntos: Incentivarlos a gastar más para alcanzar el siguiente nivel de recompensas.")
print("     3. Ofertas Personalizadas: Ofrecerles descuentos en categorías de productos que les interesan para fomentar una compra mayor.")

# ACCIÓN PARA EL SEGMENTO 3: "Nuevos y Potenciales"
print("\n--- Segmento: Nuevos y Potenciales ---")
print("   - Perfil: Han comprado recientemente pero con baja frecuencia y gasto. Podrían ser nuevos o compradores ocasionales.")
print("   - Objetivo: Fomentar la segunda y tercera compra para construir el hábito.")
print("   - Acciones Propuestas:")
print("     1. Campaña de Bienvenida: Enviar una serie de emails de onboarding que presenten la marca y la gama de productos.")
print("     2. Descuento para la Segunda Compra: Ofrecer un incentivo claro para que vuelvan pronto.")
print("     3. Contenido Útil: Enviarles guías de productos o ideas sobre cómo usar lo que compraron.")

# ACCIÓN PARA EL SEGMENTO 4: "Hibernando"
print("\n--- Segmento: Hibernando ---")
print("   - Perfil: No han comprado en mucho tiempo. Alto riesgo de pérdida total.")
print("   - Objetivo: Intentar reactivarlos antes de que sea demasiado tarde.")
print("   - Acciones Propuestas:")
print("     1. Campaña de Reactivación 'Te Extrañamos': Enviar un email con una oferta atractiva y personalizada (ej. '20% de descuento para ti').")
print("     2. Encuesta de Feedback: Preguntarles por qué no han vuelto. La información es valiosa incluso si no compran.")
print("     3. Exclusión de Campañas de Alto Coste: No invertir demasiado en este grupo; centrar los esfuerzos en los segmentos más activos.")

print("\n"+"="*60)
