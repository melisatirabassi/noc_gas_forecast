# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 12:34:35 2023

@author: Meli
"""

## cargamos las librerías que usaremos
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import random
warnings.filterwarnings('ignore')

#chequeo mi carpeta de trabajo
os.getcwd()

# defino funciones para la formatear los datasets
# dataset a minuscula
def lowercase (df):
    for col in df:
        if df[col].dtype == object:
            df[col] = df[col].str.lower()
        else:
            continue
    return df

#eliminar espacios en blanco
def delate_spaces(df):
    for col in df:
        if df[col].dtype == object:
            df[col] = df[col].str.rstrip()
            df[col] = df[col].str.lstrip()
            df[col] = df[col].str.replace(' ' , '_')
        else:
            continue
    return df

def delate_characters(df):
    for col in df:
        if df[col].dtype == object:
            df[col] = df[col].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        else:
            continue
    return df

#%%
# cargo el primer dataset
df_frac = pd.read_csv('datos-de-fractura-de-pozos-de-hidrocarburos-adjunto-iv-actualizacin-diaria.csv', sep = ',')

# analisis exploratorio del dataset
df_frac.shape

df_frac.columns

df_frac.describe()

df_frac.info()

display(df_frac.isnull().any())

# formateo del dataset
df_frac_mod = lowercase(df_frac)
df_frac_mod = delate_spaces(df_frac_mod)
df_frac_mod = delate_characters(df_frac_mod)

# chequeo si hay id de pozo duplicados
df_frac_mod[df_frac_mod.duplicated('idpozo')] # hay valores dulicados porque se fracturó en dos etapas

# de los valores duplicados me quedo con el de mayor cantidad de fracturas
df_frac_mod.sort_values(['cantidad_fracturas'], ascending = True, inplace = True)
df_frac_mod.drop_duplicates(subset = 'idpozo', keep = 'last', inplace = True)

# creacion de nueva variable
df_frac_mod['arena_bombeada_total_tn'] = df_frac_mod['arena_bombeada_nacional_tn'] + df_frac_mod['arena_bombeada_importada_tn']
df_frac_mod.head()

# lista de columnas de interes
column_filter_df_frac = ['idpozo', 'sigla', 'formacion_productiva', 'cuenca','yacimiento', 'longitud_rama_horizontal_m', 'cantidad_fracturas','agua_inyectada_m3', 'arena_bombeada_total_tn',]
df_frac_mod = df_frac_mod[column_filter_df_frac]
df_frac_mod_vm = df_frac_mod [(df_frac_mod['formacion_productiva'] == 'vaca_muerta')]
display(df_frac_mod_vm.isnull().any())

# elimino las filas cuya longitud horizontal sea cero
df_frac_mod_vm = df_frac_mod_vm[(df_frac_mod_vm[['longitud_rama_horizontal_m']] != 0).all(axis=1)]

# voy a trabajar solo con algunos yacimientos cercanos
yacimientos = ['aguada_pichana_este_vaca_muerta', 'fortin_de_piedra', 'rincon_del_mangrullo', 'aguada_pichana_oeste', 'aguada_de_castro']
df_frac_mod_vm = df_frac_mod_vm[df_frac_mod_vm['yacimiento'].isin(yacimientos)]
df_frac_mod_vm.shape

# me fijo cuales son los valores unicos en idpozo ya que asi puedo identificar solo los horizontales
pozos_horizontales_gas = df_frac_mod_vm['idpozo'].unique().tolist()

#%%
# cargo el segundo dataset
df_prod = pd.read_csv('produccin-de-pozos-de-gas-y-petrleo-no-convencional.csv', sep=',')

# analisis exploratorio del dataset
df_prod.shape

df_prod.columns

df_prod.describe()

df_prod.info()

display(df_prod.isnull().any())

# formateo del dataset
df_prod_mod = lowercase(df_prod)
df_prod_mod = delate_spaces(df_prod)
df_prod_mod = delate_characters(df_prod)

# lista de columnas de interes
column_filter_df_prod = ['idpozo','sigla','cuenca','areayacimiento','fecha_data','prod_gas','prod_pet','tef','sub_tipo_recurso','tipopozo']
df_prod_mod = df_prod_mod[column_filter_df_prod]
display(df_prod_mod.isnull().any())

# chequeo si las celdas con valores nulos tienen produccion de gas o petróleo
df_prod_mod.loc[df_prod_mod['tipopozo'].isnull(),'prod_gas'].sum()
df_prod_mod.loc[df_prod_mod['tipopozo'].isnull(),'prod_pet'].sum()

# creo el df para pozos de gas de vm de la cuenca nequina
df_prod_mod_vm_gas = df_prod_mod[(df_prod_mod['sub_tipo_recurso'] == 'shale') & (df_prod_mod['tipopozo'] == 'gasifero') & (df_prod_mod['cuenca'] == 'neuquina')]
df_prod_mod_vm_gas.drop('prod_pet', axis = 1, inplace = True)

# voy a trabajar solo con algunos yacimientos cercanos
#yacimientos = ['aguada_pichana_este_vaca_muerta', 'fortin_de_piedra', 'rincon_del_mangrullo', 'aguada_pichana_oeste', 'aguada_de_castro']
df_prod_mod_vm_gas = df_prod_mod_vm_gas[df_prod_mod_vm_gas['areayacimiento'].isin(yacimientos)]
df_prod_mod_vm_gas.shape

# voy a trabajar solo con los pozos horizontales filtrados en el dataset 1
df_prod_mod_vm_gas = df_prod_mod_vm_gas[df_prod_mod_vm_gas['idpozo'].isin(pozos_horizontales_gas)]
df_prod_mod_vm_gas.shape

# ordeno el dataset por pozo id y fecha dato en dorma ascendente
df_prod_mod_vm_gas.sort_values(['areayacimiento','idpozo', 'fecha_data'], ascending = [True, True, True], inplace = True)

# contador para la cantidad de meses en produccion de cada pozo
df_prod_mod_vm_gas['prod_meses'] = df_prod_mod_vm_gas.groupby('idpozo').cumcount() + 1

# contador para cacular la acumulada de gas
df_prod_mod_vm_gas['prod_gas_cum_Mm3'] = df_prod_mod_vm_gas[['idpozo','prod_gas']].groupby('idpozo').cumsum()/1000

# calculo la produccion en km3/d de gas
df_prod_mod_vm_gas['prod_gas_km3d'] = df_prod_mod_vm_gas['prod_gas'] / df_prod_mod_vm_gas['tef']

# me creo un dataframe con idpozo y cum_gas a los 18 meses para luego concatenar con el dataset frac
df_cum_gas_18m = df_prod_mod_vm_gas[(df_prod_mod_vm_gas['prod_meses'] == 18)]
df_cum_gas_18m = df_cum_gas_18m[['idpozo', 'prod_gas_cum_Mm3']]
df_cum_gas_18m['idpozo'].count()
pozos_horizontales_18m = df_cum_gas_18m['idpozo'].unique().tolist()

# guardo el dataset de producciones modificado 
#df_prod_mod_vm_gas.to_csv('prod_pruebaDeepAR.csv')

#%%
# creo un nuevo dataset que viene de concatenar el primer y segundo dataset con pozos horozontales de los yacimientos 
# seleccionados de vaca muerta con mas de 24 meses de produccion

df_frac_mod_vm_18m = df_frac_mod_vm[df_frac_mod_vm['idpozo'].isin(pozos_horizontales_18m)]
df_frac_mod_vm_18m['idpozo'].count()

# agrego la columna acumulada de gas a los dos años al dataset de parametros de fractura
df_vm_gas = pd.merge(df_frac_mod_vm_18m, df_cum_gas_18m, on = 'idpozo')

# veo como se relacionan las variables
import seaborn as sns

sns.pairplot(df_vm_gas, corner = True)

corr = df_vm_gas.corr()
plt.figure(figsize = (16,6))
sns.heatmap(corr, cmap='BrBG', annot = True)

#grafico si hay una relacion entre cantidad de arena y fluido bombeado
plt.figure(figsize = (16,6))
plt.plot(df_vm_gas['arena_bombeada_total_tn'], df_vm_gas['agua_inyectada_m3'], 'o')
plt.title('Relacion agua inyectada vs arena bombeada')
plt.xlabel('Arena bombeada')
plt.ylabel('Agua inyectada')   
plt.show()

# me voy a quedar solo con algunas variables para crear los modelos
df_vm_gas_mod = df_vm_gas.drop(columns = ['cantidad_fracturas', 'agua_inyectada_m3'])

#%%
# aplico clustering

os.environ["OMP_NUM_THREADS"] = '1'
from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# dataset para clustering
df_vm_gas_mod_cluster = df_vm_gas_mod.drop(columns = ['idpozo', 'sigla', 'formacion_productiva','cuenca', 'yacimiento']) 

# normalizamos las dimensiones
scaler = MinMaxScaler()
df_vm_gas_mod_cluster_ss = pd.DataFrame(scaler.fit_transform(df_vm_gas_mod_cluster), columns = df_vm_gas_mod_cluster.columns)

random.seed(12)

# elijamos k usando el método del codo
wcss=[]
for n_cluster in range(2, 11):
    km = KMeans(n_clusters = n_cluster, init='k-means++', max_iter = 10000, random_state = 12, verbose = 0, tol = 0.0001)
    preds = km.fit_predict(df_vm_gas_mod_cluster_ss)
    wcss.append(km.inertia_)

# graficamos el método del codo
plt.plot(range(2,11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')   # within cluster sum of squares
plt.show()

# planteamos nuestro modelo con 4 clusters y lo hacemos clasificar
km = KMeans(n_clusters = 4, n_init = 100, max_iter= 10000, init='random', random_state = 12, verbose = 0, tol = 0.0001)
preds = km.fit_predict(df_vm_gas_mod_cluster_ss)

# veo como quedo la clasificacion en 3D
fig = plt.figure(figsize=(16,6))
ax = plt.axes(projection="3d")
ax.scatter3D(df_vm_gas_mod_cluster['longitud_rama_horizontal_m'],
             df_vm_gas_mod_cluster['prod_gas_cum_Mm3'], 
             df_vm_gas_mod_cluster['arena_bombeada_total_tn'],
             c=preds, cmap='Dark2')
plt.title('Clasificacion de productividad')
#ax.set_xlabel('Longitud rama horizontal')
#ax.set_ylabel('Produccion de gas acumulada')
#ax.set_zlabel('Arena bombeada total')
plt.show()

print('              CENTROIDES')
print(' longitud_rama_horizontal_m  arena_bombeada_total_tn  prod_gas_cum_Mm3')
print(scaler.inverse_transform(km.cluster_centers_))

df_vm_gas_mod_cluster['preds'] = preds 
df_vm_gas_mod_2 = df_vm_gas_mod
df_vm_gas_mod_2['preds'] = preds 
df_vm_gas_mod_2.head(10)

# guardo el dataset etiquetado
#df_vm_gas_mod_2.to_csv('pruebaDeepAR.csv')

#%%
# arboles de decision para clasificar nuevos pozos

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

# armo el set de variables independentes y dependiente
X = df_vm_gas_mod_2[['longitud_rama_horizontal_m', 'arena_bombeada_total_tn']]
y = df_vm_gas_mod_2['preds']

# separo en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# armo el arbol de decision
scaler = MinMaxScaler()
tree_clas = DecisionTreeClassifier(random_state = 0)
pipe_tree = Pipeline(steps = [('scaler', scaler), ('tree_clas', tree_clas)])
param_tree = [{'tree_clas__max_depth': [2,3,4,5,6],'tree_clas__min_samples_leaf': [3,4,5,6]}]
gs_tree = GridSearchCV(estimator=pipe_tree, param_grid = param_tree, scoring = 'accuracy', cv = None, n_jobs = -1, verbose = 1)
gs_tree.fit(X_train, y_train)
gs_tree.best_params_
gs_tree.score(X_train, y_train)
print("\n The best estimator across ALL searched params:\n",gs_tree.best_estimator_)
print("\n The best score across ALL searched params:\n",gs_tree.best_score_)
print("\n The best parameters across ALL searched params:\n",gs_tree.best_params_)

# genero el mejor arbol
model_tree_clas = DecisionTreeClassifier(max_depth = 2, min_samples_leaf = 3)
model_tree_clas = model_tree_clas.fit(X_train, y_train)

# grafico el arbol
plt.figure(figsize=(10,8), dpi=150)
plot_tree(model_tree_clas, feature_names=X_train.columns)

# exporto el grafico del arbol
dot_data = export_graphviz(model_tree_clas,
                           feature_names = X_train.columns,
    class_names = y_test.unique().astype("str"),
    filled = True,
    rounded = True,
    special_characters = True
)
graph = graph_from_dot_data(dot_data)
graph.write_png('productividad.png')

# calculo como predice el mejor arbol
y_pred = model_tree_clas.predict(X_test)
matrix_tree_pred = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión:')
s = sns.heatmap(matrix_tree_pred, annot=True)
s.set(ylabel='True label', xlabel='Predicted label')
plt.show()
print('\n acurracy:\n',accuracy_score(y_test, y_pred))
print('\n precision_score:\n',precision_score(y_test, y_pred, average='micro'))

#%%
# me voy a quedar con el cluster 0 y arena bombeada mayor a 4000 tn para las predicciones que hago en Sagemaker

#%%
# voy a armar un dataset para testar el modelo en Sagemaker con pozos de entre 5 y 18 meses
df_cum_gas_menos18m = df_prod_mod_vm_gas[~df_prod_mod_vm_gas['idpozo'].isin(pozos_horizontales_18m)]
pozos_horizontales_menos18m = df_cum_gas_menos18m['idpozo'].unique().tolist()

df_frac_mod_vm_menos18m = df_frac_mod_vm[df_frac_mod_vm['idpozo'].isin(pozos_horizontales_menos18m)]
df_frac_mod_vm_menos18m['idpozo'].count()

df_vm_gas_new_wells = df_frac_mod_vm_menos18m.copy()

X_new_wells = df_vm_gas_new_wells[['longitud_rama_horizontal_m', 'arena_bombeada_total_tn']]

# utilizo el arbol de decision para clasificar esos pozos
y_pred_new_wells = model_tree_clas.predict(X_new_wells)
df_vm_gas_new_wells['preds'] = y_pred_new_wells

#del df_vm_gas_new_wells['pred']

# guardo el archivo para predecir en Sagemaker
#df_vm_gas_new_wells.to_csv('new_wells_DeepAR.csv')

