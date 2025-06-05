# Sistema de Detección de Intrusiones (IDS) con Machine Learning 🚀

---

## 💡 Descripción del Proyecto

Este proyecto es una implementación de un **Sistema de Detección de Intrusiones (IDS)** basado en **Machine Learning** utilizando Python. Su propósito fundamental es analizar el tráfico de red para identificar y clasificar conexiones como **"normales"** o **"ataques"**. Actúa como una capa crucial de seguridad, alertando sobre actividades maliciosas y patrones de comportamiento anómalos.

El enfoque principal de este desarrollo ha sido la optimización de la capacidad del modelo para **detectar el mayor número posible de ataques reales (Recall)**, una métrica de rendimiento de vital importancia en el ámbito de la ciberseguridad, donde un falso negativo (un ataque no detectado) puede tener consecuencias críticas.

---

## 📊 Dataset Utilizado: NSL-KDD

El proyecto se basa en el **Dataset NSL-KDD**, una versión mejorada y depurada del reconocido dataset KDD Cup 99. Este conjunto de datos es un estándar en la investigación de IDS, proporcionando una amplia gama de características de tráfico de red y clasificándolas en tráfico normal y diversas categorías de ataques (como Denegación de Servicio - DoS, Acceso de Usuario a Root - U2R, Acceso Remoto a Local - R2L, y Probing).

Los archivos específicos utilizados son:
* `KDDTrain+.txt` (para el entrenamiento del modelo)
* `KDDTest+.txt` (para la evaluación del rendimiento del modelo)

**Fuente de los Datos:**
Puedes descargar estos archivos directamente desde los siguientes enlaces para replicar el entorno:
* [KDDTrain+.txt](https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt)
* [KDDTest+.txt](https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt)

---

## 🛠️ Metodología y Desarrollo

El proceso de construcción del IDS se estructuró en varias etapas fundamentales:

### 1. Carga y Exploración Inicial de Datos
* Los datasets fueron cargados eficientemente en `DataFrames` de `pandas`.
* Se realizó una inspección preliminar utilizando `df.head()` y `df.info()` para comprender la estructura de los datos, identificar tipos de columnas (numéricas vs. categóricas) y obtener un primer vistazo a la distribución de la variable objetivo (`attack_type`).

### 2. Preprocesamiento de Datos
* **Simplificación de la Variable Objetivo:** La columna `attack_type` se transformó en una etiqueta binaria `is_attack` (donde `0` representa tráfico normal y `1` cualquier tipo de ataque).
* **Codificación One-Hot (One-Hot Encoding):** Las características categóricas (`protocol_type`, `service`, `flag`) fueron convertidas a un formato numérico binario utilizando `pd.get_dummies()`. Se concatenaron los datasets de entrenamiento y prueba antes de este paso para asegurar la consistencia en el número y orden de las columnas resultantes.
* **Escalado de Características:** Todas las características numéricas fueron estandarizadas usando `StandardScaler` de `scikit-learn`. Este paso es crucial para algoritmos sensibles a la escala, asegurando que ninguna característica domine desproporcionadamente debido a su rango de valores.

### 3. Entrenamiento y Evaluación de Modelos de Machine Learning
Se experimentó con diferentes algoritmos de clasificación, evaluando su capacidad para predecir correctamente el tráfico de red, con un énfasis particular en el `Recall` para la clase de ataque.

* **Árbol de Decisión (Decision Tree Classifier):**
    * Un modelo inicial para establecer una línea base de rendimiento.
    * **Recall de Ataques:** Aprox. 66.98%
* **Random Forest Classifier:**
    * Modelo de ensamblaje que mejora la robustez y reduce el sobreajuste.
    * Se probó una versión estándar y otra con `class_weight='balanced'` para abordar el desequilibrio de clases, aunque los resultados iniciales mostraron un compromiso en el recall para este modelo específico.
    * **Recall de Ataques (sin pesos):** Aprox. 62.58%
    * **Recall de Ataques (con pesos):** Aprox. 60.85%
* **XGBoost (Extreme Gradient Boosting Classifier):**
    * Considerado uno de los algoritmos más potentes para datos tabulares. Se configuró con `scale_pos_weight` para optimizar directamente el recall de la clase minoritaria (ataques).

### 4. Optimización de Hiperparámetros (GridSearchCV)
* Se utilizó `GridSearchCV` para realizar una búsqueda exhaustiva de la mejor combinación de hiperparámetros para el `RandomForestClassifier`, con el objetivo de maximizar el `recall` en la clase de ataque mediante validación cruzada.
* Los resultados de esta optimización mostraron los desafíos inherentes al equilibrio entre `Precision` y `Recall` en este dataset.

---

## ✨ Rendimiento del Modelo Final (XGBoost)

El modelo **XGBoost Classifier** ha demostrado ser el más eficaz para nuestro objetivo de IDS, logrando un balance óptimo entre una alta tasa de detección de ataques y un número manejable de falsas alarmas.

**Matriz de Confusión del XGBoost (en el conjunto de prueba):**

[[9435  276]
[4412 8421]]


* **Verdaderos Positivos (VP):** 8421 (Ataques detectados correctamente)
* **Verdaderos Negativos (VN):** 9435 (Tráfico normal detectado correctamente)
* **Falsos Positivos (FP):** 276 (Tráfico normal clasificado erróneamente como Ataque - Falsas Alarmas)
* **Falsos Negativos (FN):** 4412 (Ataques reales clasificados erróneamente como Normal - **¡Ataques no detectados, punto crítico a mejorar!**)

**Métricas de Clasificación Clave:**

* **Recall para la detección de ataques:** `0.6562` (65.62%)
    * Indica que el modelo fue capaz de identificar el 65.62% de todos los ataques reales presentes en el conjunto de prueba. En la detección de intrusiones, un recall alto es de máxima prioridad.
* **Precisión para la detección de ataques:** `0.9683` (96.83%)
    * Significa que cuando el modelo predice que una conexión es un ataque, tiene una fiabilidad del 96.83%. Esto resulta en un número muy bajo de falsas alarmas, lo cual es deseable para la operativa de un IDS.

Este balance entre un **recall robusto** y una **precisión muy alta** posiciona al modelo XGBoost como una solución sólida para este sistema de detección de intrusiones.

---

## 🚀 Cómo Ejecutar el Proyecto

Para poner en marcha este proyecto en tu entorno local:

1.  **Clonar el Repositorio:**
    ```bash
    git clone [https://github.com/tu-usuario/tu-nombre-repositorio.git](https://github.com/tu-usuario/tu-nombre-repositorio.git)
    cd tu-nombre-repositorio
    ```
    *(Asegúrate de reemplazar `tu-usuario` y `tu-nombre-repositorio` con los datos de tu propio repositorio de GitHub.)*

2.  **Descargar los Datasets:**
    Coloca los archivos `KDDTrain+.txt` y `KDDTest+.txt` directamente en la raíz de la carpeta del proyecto. Puedes descargarlos desde los enlaces proporcionados en la sección "Dataset Utilizado".

3.  **Instalar Dependencias:**
    Asegúrate de tener Python 3.x instalado. Luego, instala todas las librerías necesarias ejecutando el siguiente comando en tu terminal, dentro de la carpeta del proyecto:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecutar el Script Principal:**
    Una vez instaladas las dependencias y descargados los datasets, ejecuta el script principal:
    ```bash
    python deteccion_intrusiones.py
    ```
    El script cargará los datos, realizará el preprocesamiento, entrenará los modelos (Árbol de Decisión, Random Forest, XGBoost) y mostrará sus métricas de evaluación en la terminal. Los modelos entrenados también serán guardados como archivos `.pkl`.

---

## 💡 Futuras Mejoras y Expansión

Este proyecto representa una base sólida para un IDS. Para llevarlo a un nivel superior, se podrían considerar las siguientes mejoras:

* **Afinar Hiperparámetros:** Realizar una optimización más exhaustiva de los hiperparámetros de XGBoost utilizando `RandomizedSearchCV` o herramientas avanzadas como `Optuna` para buscar un equilibrio aún mejor entre `Precision` y `Recall`.
* **Manejo Avanzado de Desequilibrio de Clases:** Implementar técnicas como **SMOTE (Synthetic Minority Over-sampling Technique)** o **ADASYN** para generar muestras sintéticas de la clase de ataque y mejorar el entrenamiento.
* **Exploración de Modelos de Deep Learning:** Investigar la aplicación de Redes Neuronales (ej. con TensorFlow/Keras) para la detección de intrusiones, especialmente si los datos son muy complejos o de alta dimensionalidad.
* **Análisis de Importancia de Características:** Utilizar librerías como SHAP o LIME para obtener una mejor interpretabilidad del modelo y entender qué características contribuyen más a sus decisiones.
* **Despliegue y Demo:** Desarrollar una pequeña aplicación web (usando Flask o FastAPI) o una interfaz de usuario simple (con Streamlit) que permita cargar nuevas muestras de tráfico y obtener predicciones en tiempo real utilizando el modelo guardado.
* **Detección Multi-Clase:** Extender el sistema para clasificar no solo "normal" vs. "ataque", sino los diferentes tipos de ataque (DoS, Probing, U2R, R2L).

---