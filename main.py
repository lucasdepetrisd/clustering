# %%
# Online Retail Dataset Analysis
# Carga y limpieza rigurosa del dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("Libraries imported successfully!")

# %%
# Cargar el dataset Online Retail desde UCI
print("=" * 60)
print("CARGANDO DATASET ONLINE RETAIL")
print("=" * 60)

print("Fetching Online Retail dataset from UCI...")
online_retail = fetch_ucirepo(id=352)

# Obtener los datos
df = online_retail.data.features

print(f"✅ Dataset cargado exitosamente!")
print(f"📊 Shape original: {df.shape}")
print(f"📋 Columnas: {list(df.columns)}")
print(f"📅 Rango de fechas: {df['InvoiceDate'].min()} a {df['InvoiceDate'].max()}")

# Mostrar primeras filas
print("\nPrimeras 5 filas del dataset:")
print(df.head())

# %%
# Información general del dataset
print("=" * 60)
print("INFORMACIÓN GENERAL DEL DATASET")
print("=" * 60)

print("Dataset Info:")
print("-" * 30)
df.info()

print("\nEstadísticas descriptivas:")
print("-" * 30)
print(df.describe())

# %%
# Análisis de valores nulos
print("=" * 60)
print("ANÁLISIS DE VALORES NULOS")
print("=" * 60)

missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing Count': missing_values,
    'Missing Percentage': missing_percentage
})
missing_df = missing_df[missing_df['Missing Count'] > 0]

if len(missing_df) > 0:
    print("Valores nulos encontrados:")
    print(missing_df)
else:
    print("✅ No se encontraron valores nulos!")

# %%
# Printear tipos de datos del df
print("=" * 60)
print("TIPOS DE DATOS DEL DATASET")
print("=" * 60)

print(df.dtypes)
print(type(df['CustomerID'][0]))

# %%
# Análisis de tipos de datos y conversiones necesarias
print("=" * 60)
print("ANÁLISIS DE TIPOS DE DATOS")
print("=" * 60)

print("Tipos de datos actuales:")
print(df.dtypes)

# Convertir InvoiceDate a datetime porque es un string
print("\nConvirtiendo InvoiceDate a datetime...")
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Convertir CustomerID a int (eliminando nulos primero)
print("Convirtiendo CustomerID a int...")
df['CustomerID'] = df['CustomerID'].astype('Int64')  # Permite valores nulos

print("\nTipos de datos después de conversión:")
print(df.dtypes)


# %%
# Printeamos descripción del dataset
df.describe()

# %%
# Hay valores negativos para Quantity. Esto podria deberse a devoluciones. Por lo tanto eliminamos los valores negativos de Quantity.
devoluciones = df[df['Quantity'] < 0]
print(f"Devoluciones encontradas: {len(devoluciones):,}")

# Eliminamos las devoluciones
df = df[df['Quantity'] >= 0]
print(f"Eliminados devoluciones. Registros restantes: {len(df):,}")

# %%
# Hay 2 transacciones con UnitPrice negativo. Esto no nos brinda informacion relevante. Por lo tanto eliminamos los valores negativos de UnitPrice.
precio_negativo = df[df['UnitPrice'] < 0]
print(f"Transacciones con UnitPrice negativo encontradas: {len(precio_negativo):,}")

# Eliminamos las transacciones con UnitPrice negativo
df = df[df['UnitPrice'] >= 0]
print(f"Eliminados transacciones con UnitPrice negativo. Registros restantes: {len(df):,}")

# %%
# Hay transacciones con precio 0. Esto no nos brinda informacion relevante. Por lo tanto eliminamos los valores 0 de UnitPrice.
precio_cero = df[df['UnitPrice'] == 0]
print(f"Transacciones con UnitPrice 0 encontradas: {len(precio_cero):,}")

# Eliminamos las transacciones con UnitPrice 0
df = df[df['UnitPrice'] > 0]
print(f"Eliminados transacciones con UnitPrice 0. Registros restantes: {len(df):,}")

# %%
# Filas sin descripcion
print(df[df['Description'].isna()])
# Se observa que ya no quedan filas sin descripcion tras eliminar los valores sin sentido.

# %%
# Revisamos duplicados y los eliminamos
print(df.duplicated().sum())
df = df.drop_duplicates()

# %%
# Filas con CustomerID nulo y las eliminamos
print(len(df[df['CustomerID'].isna()]))
df = df[df['CustomerID'].notna()]

# %%
# Analisis de Description
print("=" * 60)
print("ANÁLISIS DE DESCRIPTION")
print("=" * 60)

# Información básica
print(f"Total de descripciones únicas: {df['Description'].nunique():,}")
print(f"Total de registros: {len(df):,}")
print(f"Descripciones con valores nulos: {df['Description'].isnull().sum():,}")

# Palabras más comunes
print("\nPalabras más comunes en descripciones:")
all_words = ' '.join(df['Description'].dropna()).lower().split()
word_counts = pd.Series(all_words).value_counts().head(10)
for word, count in word_counts.items():
    print(f"'{word}': {count:,} veces")

# %%
# Convertimos description a minusculas y chequeamos unicos
df['Description'] = df['Description'].str.lower()
print(df['Description'].nunique())

# %%
# Quitamos signos de puntuacion y chequeamos unicos
df['Description'] = df['Description'].str.replace(r'[^\w\s]', '', regex=True)
print(df['Description'].nunique())

# %%
# Agregamos features de fecha por año, mes y dia de la semana
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
df['Hour'] = df['InvoiceDate'].dt.hour
df['MonthYear'] = pd.to_datetime(df[['Year', 'Month']].assign(Day=1))

# %%
df.head()

# %%
# Agregamos columna de total por transaccion
df['Total'] = df['Quantity'] * df['UnitPrice']

# %%
print(f"📋 Columnas: {list(df.columns)}")

# %%
pd.DataFrame(df['UnitPrice'].describe())
# %%
df[df['UnitPrice']>50]['Description'].unique().tolist()
# %%
sns.distplot(df[df['UnitPrice']>50]['UnitPrice'], kde=False, rug=True);


# %%
customer_country=df[['Country','CustomerID']].drop_duplicates()
customer_country.groupby(['Country'])['CustomerID'].aggregate('count').reset_index().sort_values('CustomerID', ascending=False)

# %%
print("Transactions were made in", len(df['Country'].unique().tolist()), "different countries")


# %%
print("Number of transactions where country is unspecified:", len(df[df['Country']=='Unspecified']))

# %%
plot1 = pd.DataFrame(df.groupby(['Country'])['Total'].sum()).reset_index()
plot1 = plot1.sort_values(['Total']).reset_index(drop=True)
plot2 = pd.DataFrame(df.groupby(['Country'])['Total'].count()).reset_index()
plot2 = plot2.sort_values(['Total']).reset_index(drop=True)
# %%
