# Bayesian Analysis of Dark Energy Models / Análisis Bayesiano de Modelos de Energía Oscura

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Thesis_Project-orange)

**[🇺🇸 English](#-english-version) | [🇲🇽 Español](#-versión-en-español)**

---

## 🇺🇸 English Version

### 🔭 Project Overview
This repository contains the source code and datasets used for the statistical analysis presented in the Physics Engineering thesis: **"Analysis of Dark Energy Models using Fast Radio Bursts and Cosmic Chronometers"**.

The project implements a Bayesian inference pipeline (MCMC) to constrain cosmological parameters and compare the performance of three theoretical models:
* **$\Lambda$CDM** (Standard Model).
* **CPL** (Chevallier-Polarski-Linder dynamic parametrization).
* **PEDE** (Phenomenologically Emergent Dark Energy).

### 📂 Repository Contents

#### Main Scripts
* **`ajuste_general.py`**: The main engine. Performs MCMC sampling using `emcee`. It calculates the joint Log-Likelihood by integrating the Dispersion Measure ($DM_{IGM}$) and the expansion rate $H(z)$. It outputs Markov chains and model selection criteria (AIC, BIC).
* **`graficas.py`**: Visualization suite. Reads the generated chains and produces high-quality vectorized figures (PDF) including corner plots, convergence traces, and kinematic reconstructions ($q(z)$, $j(z)$).

#### Data
* `localized_FRBs(1).txt`: Catalog of **92 localized Fast Radio Bursts**.
* `HzTable_MM_BC32.txt`: Compilation of **32 Cosmic Chronometers** measurements.

### 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO.git](https://github.com/YOUR_USERNAME/YOUR_REPO.git)
    cd YOUR_REPO
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### ⚙️ Usage

1.  **Configure the Model:**
    Open `ajuste_general.py` and modify the global variables at the top:
    ```python
    MODEL_NAME = "PEDE"   # Options: "LCDM", "CPL", "PEDE"
    RUN_FRBS   = True     # Use FRB data
    RUN_HZ     = True     # Use H(z) data
    ```

2.  **Run the MCMC Analysis:**
    ```bash
    python ajuste_general.py
    ```
    *Results (CSVs and chains) will be saved in the `Ajuste_Modelos/MODEL_NAME/` folder.*

3.  **Generate Plots:**
    ```bash
    python graficas.py
    ```
    *Figures will be saved in `Ajuste_Modelos/MODEL_NAME/GRAFICAS_GLOBALES/`.*

---

## 🇲🇽 Versión en Español

### 🔭 Descripción del Proyecto
Este repositorio contiene el código fuente y los conjuntos de datos utilizados para el análisis estadístico presentado en la tesis de Ingeniería Física: **"Análisis de Modelos de Energía Oscura utilizando Ráfagas Rápidas de Radio y Cronómetros Cósmicos"**.

El proyecto implementa un flujo de trabajo de inferencia bayesiana (MCMC) para restringir parámetros cosmológicos y comparar el desempeño de tres modelos teóricos:
* **$\Lambda$CDM** (Modelo Estándar).
* **CPL** (Parametrización dinámica Chevallier-Polarski-Linder).
* **PEDE** (Energía Oscura Fenomenológicamente Emergente).

### 📂 Contenido del Repositorio

#### Scripts Principales
* **`ajuste_general.py`**: Motor principal. Realiza el muestreo MCMC usando `emcee`. Calcula la Log-Verosimilitud conjunta integrando la Medida de Dispersión ($DM_{IGM}$) y la tasa de expansión $H(z)$. Genera las cadenas de Markov y criterios de selección (AIC, BIC).
* **`graficas.py`**: Suite de visualización. Lee las cadenas generadas y produce gráficos vectorizados de alta calidad (PDF), incluyendo *corner plots*, trazas de convergencia y reconstrucciones cinemáticas ($q(z)$, $j(z)$).

#### Datos
* `localized_FRBs(1).txt`: Catálogo de **92 Ráfagas Rápidas de Radio** localizadas.
* `HzTable_MM_BC32.txt`: Compilación de **32 mediciones de Cronómetros Cósmicos**.

### 🚀 Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/TU_USUARIO/TU_REPO.git](https://github.com/TU_USUARIO/TU_REPO.git)
    cd TU_REPO
    ```

2.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

### ⚙️ Uso

1.  **Configurar el Modelo:**
    Abre `ajuste_general.py` y modifica las variables globales al inicio:
    ```python
    MODEL_NAME = "PEDE"   # Opciones: "LCDM", "CPL", "PEDE"
    RUN_FRBS   = True     # Usar datos de FRBs
    RUN_HZ     = True     # Usar datos de H(z)
    ```

2.  **Ejecutar el Análisis MCMC:**
    ```bash
    python ajuste_general.py
    ```
    *Los resultados (CSVs y cadenas) se guardarán en la carpeta `Ajuste_Modelos/MODEL_NAME/`.*

3.  **Generar Gráficas:**
    ```bash
    python graficas.py
    ```
    *Las figuras se guardarán en `Ajuste_Modelos/MODEL_NAME/GRAFICAS_GLOBALES/`.*

---

### 👨‍💻 Author / Autor

**Miguel Ramón Chávez Toriz** Physics Engineering | Ingeniería Física  
*Undergraduate Thesis / Tesis de Licenciatura* 2025
