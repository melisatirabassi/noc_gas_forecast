# Production Forecast with DeepAR de AWS

Se descargaron datos públicos la pagina de secretaria de energia https://datos.gob.ar/. Se utilizaron dos dataset, uno con las caracteristicas de los pozos de no convencional y otro con las producciones mensuales de cada pozo. El objetivo de este trabajo es clasificar las pozos de no convencional gas según su simulitud y predecir la producción de cada uno de los pozos que pertenecen a un grupo.

![Site-Merch_Amazon-Timestream_SocialMedia_2](https://github.com/melisatirabassi/production_forecast_DeepAR_AWS/assets/124107756/dbfa65ed-819f-432f-b816-628075a2514b)


#EDA

En Spyder se eliminaron los espacios en blanco y acentos, se cambió la letra a miniscula, se identificaron filas duplicadas, valores nulos, se calculó la correlación entre variables, se generaron nuevas variables.


#ML

En Spyder se clasificaron los pozos con clustering, un algoritmo no supervisado y luego se generó un modelo supervisado con árbol de decisión para predecir la clasificación de nuevos pozos.

<img width="576" alt="DecisionTree" src="https://github.com/melisatirabassi/production_forecast_DeepAR_AWS/assets/124107756/5e53e5b2-9ffb-4314-bce5-e4e9ce62435f">

<img width="576" alt="DecisionTree" src="https://github.com/melisatirabassi/production_forecast_DeepAR_AWS/assets/124107756/a211d25e-4e63-4952-9b00-5abb07a1c75b">



#AWS
Se utilizó DeepAR de AWS predecir la producción de los pozos de un cluster.Luego se utilizaron nuevos pozos que pertenecerían a dicho cluster para predecir su comportamiento. DeepAR se encuentra dentro de Sagemaker que solo interactua con archivos que se encuentran en S3.
Por qué se utilizó DeepAr de AWS y no redes LSTM, ARIMA o Prophet? Principalmente porque se ultizó como base en el research article  Machine learning based decline curve analysis for short term oil porduction forecast. En este articulo en la introducción se hace referencia a varios papers donde se realizo forecast de producción en pozos no convencionales con redes LSTM principalmente en campos se USA. En el el research article se compara el uso de DeepAR y Prophet para decir la producción de petroleo no convencional de 22 pozos de Midland en USA. El resultado fue que el forecast a 24 meses fue bueno pero a 48 meses no para ambos algoritmos.

DeepAr es un algoritmo de aprendizaje supervisado utilizado para predecir series temporales de una dimensión utilizando redes neuronales recurrentes. El target es una o mas series temporales. La salida es probabilística y es buen algoritmo para realizar cold forecast (es decir predecir series temporales con poca o nula historia)
Los hiperparametros de DeepAr son los siguientes: entre los mas inmportantes esta
time_freq: granulometría de las series temporales
context_length: nro de puntos en el tiempo que el modelo visualiza antes de realizar la predicción
prediction_length: nro de puntos en el tiempo para los que el modelo fue entrenado para predecir
likelihood:  tipo de distribucion utilizada para generar la salida del modelo
num_cells: nro de neuronas en cada capa oculta
num_layers: numero de capas ocultas
epochs: nro max de veces que el modelo puede pasar por todo el set de entrenamiento
dropout_rate: utilizado para la regularización, algunas neuronas de forma random no se actualizan en cada iteración
learning_rate: tasa de aprendizaje
likelihood:  el modelo genera un forecast probabilístico y puede dar los quantiles de una distribución
mini_batch_size: el tamaño de cada mini batch usado para entrenar

<img width="1434" alt="Captura de pantalla 2023-05-28 a la(s) 21 37 38" src="https://github.com/melisatirabassi/production_forecast_DeepAR_AWS/assets/124107756/db655430-9b6c-436b-b15a-82606d9e4cb8">

<img width="1434" alt="Captura de pantalla 2023-05-28 a la(s) 21 36 27" src="https://github.com/melisatirabassi/production_forecast_DeepAR_AWS/assets/124107756/bbd148dc-b22a-4e6d-8428-3f88fe02c527">







