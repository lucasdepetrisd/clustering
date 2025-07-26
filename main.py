# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# Carga del Dataset
try:
    df = pd.read_csv('Online Retail.csv', sep=';', encoding='utf-8-sig', decimal=',')
    df.columns = df.columns.str.strip()

    print("Dataset cargado exitosamente.")
    print("Forma del dataset (filas, columnas):", df.shape)
except FileNotFoundError:
    print("Error: El archivo 'Online Retail.csv' no se encontró.")
    print("Por favor, asegúrate de que el archivo esté en el mismo directorio que el script.")

# Mostramos las primeras 5 filas para una inspección inicial.
print(df.columns.tolist())
df.columns = df.columns.str.strip()
print('InvoiceNo' in df.columns)  # → True o False

df.head()


# %%
# Inspección Inicial y Limpieza Básica
# Obtenemos información general del DataFrame, incluyendo tipos de datos y valores nulos.
print("Información general del DataFrame:")
df.info()

# El análisis RFM y el clustering de clientes dependen de poder agrupar por cliente.
missing_customers = df['CustomerID'].isnull().sum()
total_rows = df.shape[0]
print(f"\nSe encontraron {missing_customers} filas sin CustomerID ({missing_customers/total_rows:.2%}).")

# Eliminamos las filas donde 'CustomerID' es nulo.
df.dropna(subset=['CustomerID'], inplace=True)
print(f"Filas después de eliminar CustomerID nulos: {df.shape[0]}")

# Convertimos 'CustomerID' a tipo entero para un manejo más limpio.
df['CustomerID'] = df['CustomerID'].astype(int)

# Convertimos 'InvoiceDate' a formato datetime para poder realizar cálculos de tiempo.
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=True)

# %%

# Limpieza de Registros Inválidos

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

# Foco en el Mercado Principal (Reino Unido)

# Dado que la mayoría de los clientes son del Reino Unido nos enfocamos solo en estos.

print("Distribución de clientes por país (Top 10):")
print(df['Country'].value_counts().head(10))

df_uk = df[df['Country'] == 'United Kingdom'].copy()
print(f"\nAnálisis enfocado en el Reino Unido. Total de filas: {df_uk.shape[0]}")

# %%
# Eliminación de Duplicados
print(f"Número de filas duplicadas: {df_uk.duplicated().sum()}")
df_uk.drop_duplicates(inplace=True)
print(f"Filas después de eliminar duplicados: {df_uk.shape[0]}")


# %%

# Creación de Variables RFM (Recency, Frequency, Monetary)

import pandas as pd
import datetime as dt

# Asegurar tipos correctos
df_uk['Quantity'] = pd.to_numeric(df_uk['Quantity'], errors='coerce')
df_uk['UnitPrice'] = pd.to_numeric(df_uk['UnitPrice'], errors='coerce')
df_uk['InvoiceDate'] = pd.to_datetime(df_uk['InvoiceDate'], dayfirst=True, errors='coerce')

# 1. Calcular el valor monetario de cada transacción.
df_uk['TotalPrice'] = df_uk['Quantity'] * df_uk['UnitPrice']

# 2. Definir fecha de snapshot
snapshot_date = df_uk['InvoiceDate'].max() + dt.timedelta(days=1)
print(f"Fecha de referencia (snapshot_date) para el análisis: {snapshot_date.date()}")

# 3. Agrupar para RFM
rfm = df_uk.groupby('CustomerID').agg({
    'InvoiceDate': lambda date: (snapshot_date - date.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalPrice': 'MonetaryValue'
})

print("\nDataFrame RFM creado exitosamente:")
print(rfm.head())



# %%

# Visualización de las Variables RFM ANTES del Escalado
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


# %%

# Aplicación de Escalado para Clustering
# Para que el algoritmo de clustering funcione correctamente, debemos pre-procesar los datos.
# Realizamos un proceso de dos pasos:
# 1. Transformación Logarítmica: Para reducir el sesgo a la izquierda. Usamos log1p
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

rfm_scaled = pd.DataFrame(rfm_scaled_array, index=rfm.index, columns=rfm.columns)

print("Datos RFM transformados y escalados:")
rfm_scaled.head()


# %%

# Visualización de las Variables RFM DESPUÉS del Escalado
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

print("Las distribuciones ahora son mucho más 'normales' y están en la misma escala.")


# %%

# APLICACIÓN Y EVALUACIÓN DE ALGORITMOS DE CLUSTERING
# En esta sección, aplicaremos tres algoritmos de clustering diferentes a nuestros
# datos RFM escalados. Usaremos métodos cuantitativos para determinar los
# hiperparámetros óptimos y para comparar el rendimiento de los modelos.

# Importamos los modelos de clustering y las métricas de evaluación que usaremos.

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage

X = rfm_scaled.copy()

# ALGORITMO 1: K-MEANS
# Determinación del Número Óptimo de Clusters (k) para K-Means
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
# El método del codo muestra un "codo" visible en k=3, donde la curva comienza a aplanarse.
# Elegimos k=3 como el número óptimo de clusters para nuestro análisis.
k_optimal = 3
print(f"Justificación: Basado en el método del codo y la puntuación de silueta, el número óptimo de clusters es k={k_optimal}.")


# %%

# Aplicación de K-Means con el k óptimo

kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init='auto')
kmeans_labels = kmeans.fit_predict(X)

# Guardamos las etiquetas de los clusters para la comparación posterior.
rfm['KMeans_Cluster'] = kmeans_labels
print(f"K-Means aplicado. Número de clientes en cada cluster:")
print(rfm['KMeans_Cluster'].value_counts())


# %%

# ALGORITMO 2: CLUSTERING JERÁRQUICO AGLOMERATIVO
#  Visualización del Dendrograma:
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
# ocurre cuando se pasa de 2 a 3 clusters. Esto sugiere que k=3 es una buena elección, lo cual es consistente con nuestro hallazgo en K-Means.

# %%

# Aplicación de Clustering Aglomerativo
# Aplicamos el modelo con el mismo número de clusters que K-Means (k=3) para una
# comparación justa.

agg_cluster = AgglomerativeClustering(n_clusters=k_optimal, linkage='ward')
agg_labels = agg_cluster.fit_predict(X)

# Guardamos las etiquetas
rfm['Agg_Cluster'] = agg_labels
print(f"Clustering Aglomerativo aplicado. Número de clientes en cada cluster:")
print(rfm['Agg_Cluster'].value_counts())


# %%

# ALGORITMO 3: DBSCAN
# Determinación de Hiperparámetros para DBSCAN (eps y min_samples)
# DBSCAN tiene dos hiperparámetros:
# - min_samples: El número mínimo de puntos para formar una región densa. Una regla
#   general es usar 2 * número_de_dimensiones. En nuestro caso, 2 * 3 = 6.
# - eps: La distancia máxima para que dos puntos sean considerados vecinos. Usaremos
#   el método del gráfico k-distance para encontrar un 'eps' razonable.

# 1. Definir min_samples
min_samples_dbscan = 2 * X.shape[1]

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
# distancia de 0.4-0.7. Este punto es un buen candidato para 'eps' porque es donde
# la densidad cambia significativamente. Elegiremos eps=0.7.
eps_optimal = 0.5
print(f"Justificación: min_samples={min_samples_dbscan}. El gráfico k-distance sugiere un eps óptimo alrededor de {eps_optimal}.")


# %%

# Aplicación de DBSCAN

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
# COMPARACIÓN DE MODELOS CON MÉTRICAS INTERNAS

# Cálculo y Comparación de Métricas

# Ahora comparamos los resultados de los tres algoritmos utilizando las métricas
# internas que no requieren etiquetas verdaderas.
# - Silhouette Score: Más alto es mejor.
# - Calinski-Harabasz Score: Más alto es mejor.
# - Davies-Bouldin Score: Más bajo es mejor.

# Para DBSCAN, solo evaluamos los puntos que no son ruido.
dbscan_mask = dbscan_labels != -1
unique_labels = set(dbscan_labels[dbscan_mask])  # Etiquetas de clusters sin ruido

if len(unique_labels) < 2:
    print("DBSCAN encontró menos de 2 clusters para evaluar.")
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

# Elección del Mejor Modelo

# K-Means es la mejor opción: tiene el mayor Silhouette (0.345) y Calinski-Harabasz (3348.488),
# lo que indica clústers bien definidos. Aunque DBSCAN tiene el mejor Davies-Bouldin (0.846),
# su bajo Silhouette sugiere menor separación entre clústers. En conjunto, K-Means ofrece el mejor balance.

print("\nModelo elegido para la siguiente fase: K-Means.")

# %%

#  ANÁLISIS, INTERPRETACIÓN Y ACCIONES DE MARKETING

# En esta fase final, tomamos los clusters generados por nuestro modelo elegido (K-Means)
# y les damos un significado de negocio. El objetivo es entender quiénes son los
# clientes en cada grupo.

from sklearn.decomposition import PCA

# Para entender qué caracteriza a cada cluster, calculamos la media de las
# variables RFM originales (NO las escaladas) para cada grupo. También contamos
# cuántos clientes hay en cada cluster.

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

# Visualización de Perfiles de Clusters (Gráfico de Serpiente)

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

# Asignamos nombres descriptivos ("personas") a cada cluster basado en su perfil numérico
# y análisis previo (como el gráfico de serpiente).

# Ejemplo de mapeo basado en un análisis previo:
label_map = {
    0: 'Hibernando',          # Clientes con alta Recency, baja frecuencia y bajo gasto (Peores)
    1: 'Clientes Leales',     # Clientes con valores medios en R, F y M (Intermedios)
    2: 'Campeones',           # Clientes con baja Recency, alta frecuencia y alto gasto (Mejores)
}

# Creamos una nueva columna 'Segment' en el dataframe rfm con las etiquetas asignadas
rfm['Segment'] = rfm['KMeans_Cluster'].map(label_map)

# Mostramos las primeras filas para verificar que la asignación fue correcta
print("Etiquetas de segmento asignadas:")
print(rfm.head())


# %%

# Visualización de Clusters con PCA

# Reducimos las 3 dimensiones de RFM a 2 componentes principales (PCA)
# para poder visualizar la separación de los clusters en un gráfico de dispersión.

pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(X)

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


# %%

print("="*60)
print("CONCLUSIÓN DEL ANÁLISIS DE CLUSTERING")
print("="*60)

print("Campeones (cluster a la derecha):\nClientes con alta frecuencia y gasto, que compran con regularidad.\n"
      "Son el segmento más valioso y deben recibir acciones de fidelización como programas exclusivos,\n"
      "ofertas personalizadas y acceso anticipado a nuevos productos para maximizar su lealtad y mantener su alto nivel de compra.\n")

print("Clientes Leales (centro):\nCompran con frecuencia moderada y tienen un gasto promedio.\n"
      "Se recomienda impulsar su compromiso mediante campañas de upselling y cross-selling,\n"
      "así como incentivos que los motiven a aumentar su frecuencia o valor de compra.\n")

print("Hibernando (izquierda):\nClientes con baja frecuencia y bajo gasto, posiblemente inactivos o en riesgo de abandono.\n"
      "Es prioritario diseñar campañas de recuperación con promociones especiales, recordatorios y comunicación personalizada\n"
      "para reactivar su interés y evitar que se pierdan definitivamente.\n")

# %%
