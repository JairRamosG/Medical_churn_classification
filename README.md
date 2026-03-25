# Medical Churn Prediction

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--learn-1.6.1-orange?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-1.6.0-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Stacking-Ensemble-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/AUC--ROC-0.92-success?style=for-the-badge"/>
</div>

<br>

## Descripción del Proyecto

Este proyecto desarrolla un modelo de **Machine Learning** para predecir el **abandono de pacientes (churn)** en servicios médicos. Utilizando un enfoque de **Stacking**, se combinan múltiples algoritmos de clasificación (XGBoost, KNN, Decision Trees y Logistic Regression) con un meta-modelo de regresión logística, logrando una **precisión del 89%** y un **AUC-ROC de 0.92**.

El modelo permite identificar pacientes en riesgo de deserción, facilitando la implementación de estrategias de retención personalizadas y mejorando la continuidad del cuidado médico.

---

## Objetivos

- **Predecir** qué pacientes tienen alta probabilidad de abandonar los servicios médicos.
- **Identificar** factores clave asociados al abandono (demográficos, clínicos, de uso).
- **Implementar** un modelo robusto de stacking para maximizar el rendimiento.
- **Generar** insights accionables para equipos de retención y atención al paciente.

---

## Características Principales

| Característica | Descripción |
|----------------|-------------|
| **Stacking de Modelos** | Combinación de XGBoost, KNN, Decision Trees y Logistic Regression |
| **Meta-Modelo** | Regresión Logística para combinar predicciones de los modelos base |
| **Validación Cruzada** | Evaluación robusta con 5 folds para evitar overfitting |
| **Selección de Características** | RFE y SHAP para identificar variables más relevantes |
| **Balanceo de Clases** | SMOTE para manejar el desbalance (80% retenidos / 20% abandonan) |
| **Métricas de Evaluación** | Precisión, Recall, F1-Score, AUC-ROC, Matriz de Confusión |

---

## Resultados del Mejor Modelo

| Métrica | Valor |
|---------|-------|
| **Precisión (Precision)** | 89.0% |
| **Recall** | 85.2% |
| **F1-Score** | 87.0% |
| **AUC-ROC** | 0.92 |

> *El modelo alcanza un AUC-ROC de 0.92, demostrando una excelente capacidad para discriminar entre pacientes que abandonan y los que permanecen en el servicio.*

---

## Tecnologías Utilizadas

| Área | Tecnologías |
|------|-------------|
| **Lenguaje** | Python 3.11 |
| **Machine Learning** | Scikit-learn, XGBoost, KNN, Decision Trees, Logistic Regression |
| **Ensemble** | StackingClassifier |
| **Balanceo** | SMOTE |
| **Selección de Características** | RFE, SHAP |
| **Visualización** | Matplotlib, Seaborn, Plotly |
| **Procesamiento de Datos** | Pandas, NumPy |

---
## Trabajo a Futuro

Con el objetivo de mejorar el la visualización del modelo de predicción de abandono de pacientes, se plantean las siguientes líneas de trabajo:

---

### Mejoras en el Modelado

- **Arquitecturas de Deep Learning**: Explorar redes neuronales profundas (MLP, TabNet, redes neuronales con atención) para capturar relaciones más complejas en los datos clínicos y de comportamiento.
- **Ensembles Avanzados**: Probar combinaciones con otros algoritmos como LightGBM, CatBoost y Gradient Boosting, utilizando técnicas de voting o stacking con mayor diversidad.
- **AutoML**: Incorporar herramientas como H2O AutoML para explorar automáticamente múltiples configuraciones y seleccionar la mejor combinación.

---

### Datos y Características

- **Feature Store**: Implementar un repositorio centralizado de características para facilitar la reproducibilidad, versionamiento y reutilización en nuevos proyectos.
- **Series Temporales**: Analizar patrones temporales en el comportamiento del paciente (frecuencia de visitas, adherencia a citas) utilizando modelos LSTM o Transformer.

---


