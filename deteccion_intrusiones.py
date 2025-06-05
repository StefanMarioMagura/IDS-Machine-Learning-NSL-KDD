# 1. Importar las librerías necesarias
import pandas as pd
import numpy as np # Aunque no se usa mucho aquí, es bueno tenerlo por convención

# 2. Definir los nombres de las columnas (características)
# El dataset NSL-KDD no tiene encabezados, así que necesitamos informarlos.
# Estos nombres están estandarizados para el dataset KDD/NSL-KDD.
column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level'
]

# 3. Cargar el dataset de entrenamiento
# El separador es una coma (',') y no hay encabezados en el archivo,
# así que usamos 'names' para asignar los nombres de las columnas que definimos.
print("Cargando el dataset de entrenamiento...")
df_train = pd.read_csv('KDDTrain+.txt', sep=',', names=column_names)
print("Dataset de entrenamiento cargado con éxito!")

# 4. Cargar el dataset de prueba
print("\nCargando el dataset de prueba...")
df_test = pd.read_csv('KDDTest+.txt', sep=',', names=column_names)
print("Dataset de prueba cargado con éxito!")

# 5. Inspeccionar los primeros registros (filas) de cada DataFrame
print("\n--- Primeras 5 filas del DataFrame de Entrenamiento ---")
print(df_train.head())

print("\n--- Primeras 5 filas del DataFrame de Prueba ---")
print(df_test.head())

# 6. Obtener información general sobre los DataFrames
# Esto muestra el número de filas, columnas, tipos de datos y memoria utilizada.
print("\n--- Información sobre el DataFrame de Entrenamiento ---")
df_train.info()

print("\n--- Información sobre el DataFrame de Prueba ---")
df_test.info()

# 7. Verificar el conteo de tipos de ataques (columna 'attack_type')
# En el dataset de entrenamiento, podemos ver la distribución de los ataques.
print("\n--- Conteo de tipos de ataque en el DataFrame de Entrenamiento ---")
print(df_train['attack_type'].value_counts())

print("\n--- Conteo de tipos de ataque en el DataFrame de Prueba ---")
print(df_test['attack_type'].value_counts())



# --- Paso 4: Preprocesamiento de Datos ---

# 4.1. Simplificar la columna 'attack_type' a binario (ataque o normal)
# Primero, identifiquemos todos los tipos de ataque que NO son 'normal'.
# df_train['attack_type'].value_counts() nos mostró todos los tipos.
# Cualquier valor que no sea 'normal' será considerado un ataque (1), 'normal' será 0.

# Mapeamos la columna 'attack_type' a 0 (normal) o 1 (ataque)
# Creamos una copia para evitar SettingWithCopyWarning
df_train_processed = df_train.copy()
df_test_processed = df_test.copy()

# En el dataset de entrenamiento
is_attack_train = df_train_processed['attack_type'] != 'normal'
df_train_processed['is_attack'] = is_attack_train.astype(int) # Convertimos True/False a 1/0

# En el dataset de prueba
is_attack_test = df_test_processed['attack_type'] != 'normal'
df_test_processed['is_attack'] = is_attack_test.astype(int) # Convertimos True/False a 1/0

print("\n--- Conteo de 'is_attack' en el DataFrame de Entrenamiento (0:Normal, 1:Ataque) ---")
print(df_train_processed['is_attack'].value_counts())

print("\n--- Conteo de 'is_attack' en el DataFrame de Prueba (0:Normal, 1:Ataque) ---")
print(df_test_processed['is_attack'].value_counts())

# Eliminamos la columna original 'attack_type' y 'difficulty_level'
# ya que ya no las necesitamos para el entrenamiento del modelo
df_train_processed = df_train_processed.drop(['attack_type', 'difficulty_level'], axis=1)
df_test_processed = df_test_processed.drop(['attack_type', 'difficulty_level'], axis=1)

print("\nColumnas después de simplificar 'attack_type':")
print(df_train_processed.columns)

# 4.2. Identificar columnas categóricas para One-Hot Encoding
# Las columnas con tipo 'object' son generalmente categóricas
categorical_cols = df_train_processed.select_dtypes(include=['object']).columns
print(f"\nColumnas categóricas identificadas: {list(categorical_cols)}")

# 4.3. Aplicar One-Hot Encoding a las columnas categóricas
# Usamos pd.get_dummies para convertir las variables categóricas en numéricas (0s y 1s)

# Juntamos los datasets de entrenamiento y prueba antes del encoding
# para asegurar que todas las columnas categóricas tengan el mismo conjunto de categorías
# Esto evita problemas si una categoría aparece en un dataset pero no en el otro.
# Reseteamos el índice para evitar problemas de unión.
combined_df = pd.concat([df_train_processed.reset_index(drop=True), df_test_processed.reset_index(drop=True)], axis=0)

# Aplicar One-Hot Encoding
print("\nAplicando One-Hot Encoding a las columnas categóricas...")
combined_df_encoded = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=True) # drop_first=True evita la multicolinealidad
print("One-Hot Encoding completado.")

# Separar nuevamente los datasets de entrenamiento y prueba
df_train_encoded = combined_df_encoded.iloc[:len(df_train_processed)]
df_test_encoded = combined_df_encoded.iloc[len(df_train_processed):]

# Asegurarse de que ambos DataFrames tengan las mismas columnas, en el mismo orden
# Esto es CRÍTICO para que el modelo funcione correctamente
train_cols = set(df_train_encoded.columns)
test_cols = set(df_test_encoded.columns)

missing_in_test = list(train_cols - test_cols)
missing_in_train = list(test_cols - train_cols)

# Si faltan columnas en el test, las añadimos con ceros
for col in missing_in_test:
    df_test_encoded[col] = 0

# Si faltan columnas en el train (lo cual es menos común después de concat, pero por seguridad)
for col in missing_in_train:
    df_train_encoded[col] = 0

# Asegurarse de que el orden de las columnas sea idéntico (excepto la columna 'is_attack')
# Sacamos 'is_attack' temporalmente para ordenar y luego la volvemos a añadir
target_train = df_train_encoded['is_attack']
target_test = df_test_encoded['is_attack']

df_train_encoded = df_train_encoded.drop(columns=['is_attack'])
df_test_encoded = df_test_encoded.drop(columns=['is_attack'])

# Asegurar el mismo orden de columnas
df_test_encoded = df_test_encoded[df_train_encoded.columns]

# Volver a añadir 'is_attack'
df_train_encoded['is_attack'] = target_train
df_test_encoded['is_attack'] = target_test


print("\n--- Primeras 5 filas del DataFrame de Entrenamiento después de One-Hot Encoding ---")
print(df_train_encoded.head())

print("\n--- Información del DataFrame de Entrenamiento después de One-Hot Encoding ---")
df_train_encoded.info()

# 4.4. Separar características (X) y la variable objetivo (y)
# X son las características de entrada, y es lo que queremos predecir
X_train = df_train_encoded.drop('is_attack', axis=1) # Todas las columnas excepto 'is_attack'
y_train = df_train_encoded['is_attack'] # Solo la columna 'is_attack'

X_test = df_test_encoded.drop('is_attack', axis=1)
y_test = df_test_encoded['is_attack']

print(f"\nDimensiones de X_train (características de entrenamiento): {X_train.shape}")
print(f"Dimensiones de y_train (objetivo de entrenamiento): {y_train.shape}")
print(f"Dimensiones de X_test (características de prueba): {X_test.shape}")
print(f"Dimensiones de y_test (objetivo de prueba): {y_test.shape}")

# Opcional pero recomendado para Machine Learning: Escalado de características
# Ahora, todas nuestras características son numéricas. Pero algunas tienen valores muy grandes (como src_bytes).
# Esto puede afectar el rendimiento de algunos modelos. Usaremos StandardScaler.
from sklearn.preprocessing import StandardScaler

print("\nEscalando características numéricas...")
scaler = StandardScaler()

# Ajustamos el escalador SOLO con los datos de entrenamiento para evitar fuga de datos
X_train_scaled = scaler.fit_transform(X_train)

# Transformamos los datos de prueba usando el escalador ajustado al entrenamiento
X_test_scaled = scaler.transform(X_test)

print("Características escaladas con éxito.")

# Convertimos de nuevo a DataFrame para mantener los nombres de las columnas (opcional, pero útil para inspección)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("\n--- Primeras 5 filas del DataFrame de Entrenamiento escalado ---")
print(X_train_scaled_df.head())



# --- Paso 5: Entrenamiento del Modelo de Machine Learning ---

# 5.1. Importar el algoritmo de Árbol de Decisión
from sklearn.tree import DecisionTreeClassifier

# 5.2. Crear una instancia del modelo de Árbol de Decisión
# random_state: asegura que los resultados sean reproducibles.
# Esto significa que si ejecutas el código varias veces, obtendrás los mismos resultados.
model_dt = DecisionTreeClassifier(random_state=42)

print("\nEntrenando el modelo de Árbol de Decisión...")

# 5.3. Entrenar el modelo
# 'fit' es el método que inicia el proceso de aprendizaje del modelo.
# Le pasamos nuestras características escaladas (X_train_scaled) y nuestra variable objetivo (y_train).
model_dt.fit(X_train_scaled, y_train)

print("¡Modelo de Árbol de Decisión entrenado con éxito!")

# --- Paso 6: Evaluación del Modelo ---
# Aunque aún no lo hemos definido, el siguiente paso lógico es evaluar qué tan bien funciona.
# Solo para verificar, podemos hacer una predicción simple y ver el accuracy.
# Esto se explicará en detalle en el próximo paso.

# Realizar predicciones en el conjunto de prueba
y_pred_dt = model_dt.predict(X_test_scaled)

# Importar la métrica de precisión
from sklearn.metrics import accuracy_score

# Calcular la precisión
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"\nPrecisión inicial del modelo de Árbol de Decisión en los datos de prueba: {accuracy_dt:.4f}")

# Guardar el modelo (opcional, pero útil para portfolio)
# Necesitarás la librería 'joblib' para esto. Si no la tienes: pip install joblib
import joblib
joblib.dump(model_dt, 'modelo_arbol_decision_ids.pkl')
print("\nModelo de Árbol de Decisión guardado como 'modelo_arbol_decision_ids.pkl'")



# --- Paso 6: Evaluación Detallada del Modelo ---

# 6.1. Importar métricas de evaluación
from sklearn.metrics import classification_report, confusion_matrix

print("\n--- Evaluación Detallada del Modelo de Árbol de Decisión ---")

# 6.2. Calcular la Matriz de Confusión
# La matriz de confusión nos ayuda a ver los Verdaderos Positivos, Verdaderos Negativos,
# Falsos Positivos y Falsos Negativos.
# En nuestro caso:
# Fila 0, Columna 0: Verdaderos Negativos (Normal correctamente clasificado)
# Fila 0, Columna 1: Falsos Positivos (Normal clasificado como Ataque)
# Fila 1, Columna 0: Falsos Negativos (Ataque clasificado como Normal)
# Fila 1, Columna 1: Verdaderos Positivos (Ataque correctamente clasificado)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
print("\nMatriz de Confusión:")
print(conf_matrix_dt)

# 6.3. Generar un Reporte de Clasificación completo
# Este reporte incluye Precisión, Recall, F1-Score y soporte para cada clase.
# '0' representa 'Normal', '1' representa 'Ataque'.
class_report_dt = classification_report(y_test, y_pred_dt, target_names=['Normal', 'Ataque'])
print("\nReporte de Clasificación:")
print(class_report_dt)

print("\n--- Análisis de la Matriz de Confusión ---")
# Extraer valores de la matriz para un análisis más claro
# Para la clase 'Normal' (0)
true_negatives = conf_matrix_dt[0, 0]
false_positives = conf_matrix_dt[0, 1]

# Para la clase 'Ataque' (1)
false_negatives = conf_matrix_dt[1, 0]
true_positives = conf_matrix_dt[1, 1]

print(f"Verdaderos Positivos (Ataques detectados correctamente): {true_positives}")
print(f"Verdaderos Negativos (Tráfico normal detectado correctamente): {true_negatives}")
print(f"Falsos Positivos (Tráfico normal clasificado como Ataque - Falsas Alarmas): {false_positives}")
print(f"Falsos Negativos (Ataques no detectados - Peligroso): {false_negatives}")

# Calcular Recall para ataques (lo más importante en ciberseguridad)
recall_attacks = true_positives / (true_positives + false_negatives)
print(f"\nRecall para la detección de ataques: {recall_attacks:.4f}")

# Calcular Precisión para ataques
precision_attacks = true_positives / (true_positives + false_positives)
print(f"Precisión para la detección de ataques: {precision_attacks:.4f}")



# --- Paso 8: Implementación y Evaluación del Modelo Random Forest ---

# 8.1. Importar el algoritmo Random Forest
from sklearn.ensemble import RandomForestClassifier

# 8.2. Crear una instancia del modelo Random Forest
# n_estimators: número de árboles en el bosque (más árboles = más robusto, pero más lento)
# random_state: asegura la reproducibilidad
print("\n--- Entrenando el Modelo Random Forest ---")
model_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 usa todos los núcleos de CPU

# 8.3. Entrenar el modelo Random Forest
print("Entrenando el modelo Random Forest...")
model_rf.fit(X_train_scaled, y_train)
print("¡Modelo Random Forest entrenado con éxito!")

# 8.4. Realizar predicciones en el conjunto de prueba con Random Forest
y_pred_rf = model_rf.predict(X_test_scaled)

# 8.5. Evaluar el modelo Random Forest
print("\n--- Evaluación Detallada del Modelo Random Forest ---")

# Precisión
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nPrecisión del modelo Random Forest en los datos de prueba: {accuracy_rf:.4f}")

# Matriz de Confusión
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print("\nMatriz de Confusión para Random Forest:")
print(conf_matrix_rf)

# Reporte de Clasificación
class_report_rf = classification_report(y_test, y_pred_rf, target_names=['Normal', 'Ataque'])
print("\nReporte de Clasificación para Random Forest:")
print(class_report_rf)

print("\n--- Análisis de la Matriz de Confusión de Random Forest ---")
true_positives_rf = conf_matrix_rf[1, 1]
true_negatives_rf = conf_matrix_rf[0, 0]
false_positives_rf = conf_matrix_rf[0, 1]
false_negatives_rf = conf_matrix_rf[1, 0]

print(f"Verdaderos Positivos (Ataques detectados correctamente): {true_positives_rf}")
print(f"Verdaderos Negativos (Tráfico normal detectado correctamente): {true_negatives_rf}")
print(f"Falsos Positivos (Tráfico normal clasificado como Ataque - Falsas Alarmas): {false_positives_rf}")
print(f"Falsos Negativos (Ataques no detectados - Peligroso): {false_negatives_rf}")

recall_attacks_rf = true_positives_rf / (true_positives_rf + false_negatives_rf)
print(f"\nRecall para la detección de ataques con Random Forest: {recall_attacks_rf:.4f}")

precision_attacks_rf = true_positives_rf / (true_positives_rf + false_positives_rf)
print(f"Precisión para la detección de ataques con Random Forest: {precision_attacks_rf:.4f}")

# Guardar el modelo Random Forest
joblib.dump(model_rf, 'modelo_random_forest_ids.pkl')
print("\nModelo Random Forest guardado como 'modelo_random_forest_ids.pkl'")



# --- Paso 9: Ajuste de Hiperparámetros (Weighted Random Forest) ---

# 9.1. Re-importar el algoritmo Random Forest (ya debería estar importado)
# from sklearn.ensemble import RandomForestClassifier

# 9.2. Crear una nueva instancia del modelo Random Forest con 'class_weight'
# 'balanced' ajusta automáticamente los pesos inversamente proporcionales a las frecuencias de clase.
# Esto significa que la clase minoritaria (ataque) tendrá un peso mayor.
print("\n--- Entrenando el Modelo Random Forest con 'class_weight' ---")
model_rf_weighted = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')

# 9.3. Entrenar el modelo Random Forest con pesos ajustados
print("Entrenando el modelo Random Forest (con pesos de clase)...")
model_rf_weighted.fit(X_train_scaled, y_train)
print("¡Modelo Random Forest (con pesos) entrenado con éxito!")

# 9.4. Realizar predicciones en el conjunto de prueba
y_pred_rf_weighted = model_rf_weighted.predict(X_test_scaled)

# 9.5. Evaluar el modelo Random Forest con pesos ajustados
print("\n--- Evaluación Detallada del Modelo Random Forest (con pesos) ---")

# Precisión
accuracy_rf_weighted = accuracy_score(y_test, y_pred_rf_weighted)
print(f"\nPrecisión del modelo Random Forest (con pesos) en los datos de prueba: {accuracy_rf_weighted:.4f}")

# Matriz de Confusión
conf_matrix_rf_weighted = confusion_matrix(y_test, y_pred_rf_weighted)
print("\nMatriz de Confusión para Random Forest (con pesos):")
print(conf_matrix_rf_weighted)

# Reporte de Clasificación
class_report_rf_weighted = classification_report(y_test, y_pred_rf_weighted, target_names=['Normal', 'Ataque'])
print("\nReporte de Clasificación para Random Forest (con pesos):")
print(class_report_rf_weighted)

print("\n--- Análisis de la Matriz de Confusión de Random Forest (con pesos) ---")
true_positives_rf_weighted = conf_matrix_rf_weighted[1, 1]
true_negatives_rf_weighted = conf_matrix_rf_weighted[0, 0]
false_positives_rf_weighted = conf_matrix_rf_weighted[0, 1]
false_negatives_rf_weighted = conf_matrix_rf_weighted[1, 0]

print(f"Verdaderos Positivos (Ataques detectados correctamente): {true_positives_rf_weighted}")
print(f"Verdaderos Negativos (Tráfico normal detectado correctamente): {true_negatives_rf_weighted}")
print(f"Falsos Positivos (Tráfico normal clasificado como Ataque - Falsas Alarmas): {false_positives_rf_weighted}")
print(f"Falsos Negativos (Ataques no detectados - Peligroso): {false_negatives_rf_weighted}")

recall_attacks_rf_weighted = true_positives_rf_weighted / (true_positives_rf_weighted + false_negatives_rf_weighted)
print(f"\nRecall para la detección de ataques con Random Forest (con pesos): {recall_attacks_rf_weighted:.4f}")

precision_attacks_rf_weighted = true_positives_rf_weighted / (true_positives_rf_weighted + false_positives_rf_weighted)
print(f"Precisión para la detección de ataques con Random Forest (con pesos): {precision_attacks_rf_weighted:.4f}")

# Guardar el modelo Random Forest ponderado
joblib.dump(model_rf_weighted, 'modelo_random_forest_weighted_ids.pkl')
print("\nModelo Random Forest (con pesos) guardado como 'modelo_random_forest_weighted_ids.pkl'")



# --- Paso 10: Optimización de Hiperparámetros con GridSearchCV ---

# 10.1. Importar GridSearchCV y otras métricas necesarias
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, recall_score

print("\n--- Iniciando la optimización de hiperparámetros con GridSearchCV ---")

# 10.2. Definir los hiperparámetros a buscar y sus rangos
# 'n_estimators': número de árboles.
# 'max_features': número de características a considerar para la mejor división en cada nodo.
# 'min_samples_leaf': número mínimo de muestras requeridas para estar en un nodo hoja.
# 'class_weight': para manejar el desequilibrio de clases.
param_grid = {
    'n_estimators': [100, 200], # Probaremos 100 y 200 árboles
    'max_features': ['sqrt', 'log2'], # sqrt(n_features) o log2(n_features)
    'min_samples_leaf': [1, 2], # Mínimo de muestras en una hoja
    'class_weight': [None, 'balanced'] # Sin peso o con peso balanceado
}

# 10.3. Definir la métrica de scoring a optimizar
# Queremos maximizar el recall para la clase positiva (ataque, que es 1).
# 'pos_label=1' asegura que el recall se calcule para la clase de ataque.
scorer = make_scorer(recall_score, pos_label=1)

# 10.4. Crear la instancia de GridSearchCV
# estimator: el modelo base (RandomForestClassifier)
# param_grid: el diccionario con los hiperparámetros a probar
# scoring: la métrica que GridSearchCV intentará maximizar
# cv: número de pliegues para la validación cruzada (ej. 5)
# n_jobs: -1 para usar todos los núcleos de la CPU (acelera el proceso)
# verbose: nivel de detalle de la salida (2 para ver el progreso)
grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                              param_grid=param_grid,
                              scoring=scorer,
                              cv=3, # Usamos 3 pliegues para que no tarde demasiado, podrías usar 5
                              n_jobs=-1,
                              verbose=2)

# 10.5. Ejecutar la búsqueda en los datos de entrenamiento
print("Ejecutando Grid Search. Esto puede tardar varios minutos...")
grid_search_rf.fit(X_train_scaled, y_train)

print("Grid Search completado.")

# 10.6. Mostrar los mejores hiperparámetros encontrados
print("\nMejores hiperparámetros encontrados para Random Forest:")
print(grid_search_rf.best_params_)

# 10.7. Mostrar el mejor recall obtenido con esos hiperparámetros
print(f"Mejor recall en el conjunto de entrenamiento (validación cruzada): {grid_search_rf.best_score_:.4f}")

# 10.8. Obtener el mejor modelo entrenado
best_rf_model = grid_search_rf.best_estimator_

# 10.9. Evaluar el mejor modelo en el conjunto de prueba
print("\n--- Evaluación del Mejor Modelo Random Forest (de Grid Search) en el conjunto de Prueba ---")
y_pred_best_rf = best_rf_model.predict(X_test_scaled)

# Precisión
accuracy_best_rf = accuracy_score(y_test, y_pred_best_rf)
print(f"\nPrecisión del Mejor Random Forest en los datos de prueba: {accuracy_best_rf:.4f}")

# Matriz de Confusión
conf_matrix_best_rf = confusion_matrix(y_test, y_pred_best_rf)
print("\nMatriz de Confusión para el Mejor Random Forest:")
print(conf_matrix_best_rf)

# Reporte de Clasificación
class_report_best_rf = classification_report(y_test, y_pred_best_rf, target_names=['Normal', 'Ataque'])
print("\nReporte de Clasificación para el Mejor Random Forest:")
print(class_report_best_rf)

print("\n--- Análisis de la Matriz de Confusión del Mejor Random Forest ---")
true_positives_best_rf = conf_matrix_best_rf[1, 1]
true_negatives_best_rf = conf_matrix_best_rf[0, 0]
false_positives_best_rf = conf_matrix_best_rf[0, 1]
false_negatives_best_rf = conf_matrix_best_rf[1, 0]

print(f"Verdaderos Positivos (Ataques detectados correctamente): {true_positives_best_rf}")
print(f"Verdaderos Negativos (Tráfico normal detectado correctamente): {true_negatives_best_rf}")
print(f"Falsos Positivos (Tráfico normal clasificado como Ataque - Falsas Alarmas): {false_positives_best_rf}")
print(f"Falsos Negativos (Ataques no detectados - Peligroso): {false_negatives_best_rf}")

recall_attacks_best_rf = true_positives_best_rf / (true_positives_best_rf + false_negatives_best_rf)
print(f"\nRecall para la detección de ataques con el Mejor Random Forest: {recall_attacks_best_rf:.4f}")

precision_attacks_best_rf = true_positives_best_rf / (true_positives_best_rf + false_positives_best_rf)
print(f"Precisión para la detección de ataques con el Mejor Random Forest: {precision_attacks_best_rf:.4f}")

# Guardar el mejor modelo de Random Forest
joblib.dump(best_rf_model, 'modelo_random_forest_optimizado_ids.pkl')
print("\nMejor Modelo Random Forest guardado como 'modelo_random_forest_optimizado_ids.pkl'")



# --- Paso 11: Implementación y Evaluación del Modelo XGBoost ---

# 11.1. Importar el algoritmo XGBoost
from xgboost import XGBClassifier

# 11.2. Crear una instancia del modelo XGBoost
# objective: Define la función objetivo a optimizar. 'binary:logistic' es para clasificación binaria.
# use_label_encoder: Se ha desaconsejado en versiones recientes, se establece en False.
# eval_metric: Métrica para la evaluación durante el entrenamiento. 'logloss' es común para clasificación.
# n_estimators: Número de "rondas de boosting" o árboles. Similar a n_estimators en Random Forest.
# random_state: Para asegurar la reproducibilidad.
# scale_pos_weight: Importante para el desequilibrio de clases. Se calcula como (count(negative_class) / count(positive_class)).
#                   Esto da más peso a la clase minoritaria (ataques) para mejorar el recall.

# Calculamos el scale_pos_weight manualmente.
# Es la razón de las muestras de la clase mayoritaria (0: Normal) a la clase minoritaria (1: Ataque).
neg_count = y_train.value_counts()[0] # Conteo de la clase 'normal' (0)
pos_count = y_train.value_counts()[1] # Conteo de la clase 'ataque' (1)
scale_pos_weight_value = neg_count / pos_count

print(f"\nCalculando scale_pos_weight: {scale_pos_weight_value:.2f}")

print("\n--- Entrenando el Modelo XGBoost ---")
model_xgb = XGBClassifier(objective='binary:logistic',
                          eval_metric='logloss', # Cambiado de 'error' para ser compatible con versiones recientes
                          n_estimators=200,      # Un buen número de árboles para empezar
                          random_state=42,
                          n_jobs=-1,             # Usa todos los núcleos de CPU
                          scale_pos_weight=scale_pos_weight_value # Ajuste para desequilibrio de clases
                         )

# 11.3. Entrenar el modelo XGBoost
print("Entrenando el modelo XGBoost...")
model_xgb.fit(X_train_scaled, y_train)
print("¡Modelo XGBoost entrenado con éxito!")

# 11.4. Realizar predicciones en el conjunto de prueba con XGBoost
y_pred_xgb = model_xgb.predict(X_test_scaled)

# 11.5. Evaluación del modelo XGBoost
print("\n--- Evaluación Detallada del Modelo XGBoost ---")

# Precisión
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"\nPrecisión del modelo XGBoost en los datos de prueba: {accuracy_xgb:.4f}")

# Matriz de Confusión
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
print("\nMatriz de Confusión para XGBoost:")
print(conf_matrix_xgb)

# Reporte de Clasificación
class_report_xgb = classification_report(y_test, y_pred_xgb, target_names=['Normal', 'Ataque'])
print("\nReporte de Clasificación para XGBoost:")
print(class_report_xgb)

print("\n--- Análisis de la Matriz de Confusión de XGBoost ---")
true_positives_xgb = conf_matrix_xgb[1, 1]
true_negatives_xgb = conf_matrix_xgb[0, 0]
false_positives_xgb = conf_matrix_xgb[0, 1]
false_negatives_xgb = conf_matrix_xgb[1, 0]

print(f"Verdaderos Positivos (Ataques detectados correctamente): {true_positives_xgb}")
print(f"Verdaderos Negativos (Tráfico normal detectado correctamente): {true_negatives_xgb}")
print(f"Falsos Positivos (Tráfico normal clasificado como Ataque - Falsas Alarmas): {false_positives_xgb}")
print(f"Falsos Negativos (Ataques no detectados - Peligroso): {false_negatives_xgb}")

recall_attacks_xgb = true_positives_xgb / (true_positives_xgb + false_negatives_xgb)
print(f"\nRecall para la detección de ataques con XGBoost: {recall_attacks_xgb:.4f}")

precision_attacks_xgb = true_positives_xgb / (true_positives_xgb + false_positives_xgb)
print(f"Precisión para la detección de ataques con XGBoost: {precision_attacks_xgb:.4f}")

# Guardar el modelo XGBoost
joblib.dump(model_xgb, 'modelo_xgboost_ids.pkl')
print("\nModelo XGBoost guardado como 'modelo_xgboost_ids.pkl'")