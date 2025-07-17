# 🧬 Algoritmo Genético SIC

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Terminado-brightgreen)

Un algoritmo genético versátil para la optimización de problemas **discretos** (como la mochila) y **continuos** (funciones matemáticas), con soporte para diferentes tipos de codificación y visualización de resultados.

---

## 📦 Instalación

1. Clona el repositorio o descarga el archivo `Proyecto_SIC.py`.
2. Asegúrate de tener Python 3.8 o superior.
3. Instala las dependencias necesarias:

```bash
pip install numpy matplotlib
```

---

## 🚀 Ejecución

Desde una terminal o consola:

```bash
python Proyecto_SIC.py
```

El programa se ejecuta por consola y permite configurar paso a paso los parámetros del algoritmo.

---

## 🧠 Funcionalidades

- ✔️ Soporte para tres tipos de codificación:
  - Binaria
  - Entera
  - Decimal

- ⚙️ Configuración del algoritmo:
  - Maximizar o minimizar
  - Selección: ruleta o torneo
  - Cruce: 1 punto o 2 puntos
  - Mutación: normal o intercambio
  - Elitismo configurable
  - Número de generaciones, población, etc.

- 📊 Visualización:
  - Evolución del fitness por generación
  - Gráfica de la función objetivo en codificación decimal

---

## 📌 Ejemplo de uso (problema de la mochila)

Selecciona codificación `binaria` o `entera`, introduce pesos, utilidades y capacidad de la mochila. El algoritmo buscará la mejor combinación de elementos para maximizar (o minimizar) la utilidad respetando la capacidad.

---

## 📌 Ejemplo de uso (función matemática)

Selecciona codificación `decimal`, ingresa una función como:

```python
abs(x - 5)/2 + np.sin(x)
```

Define restricciones opcionales (intervalos inválidos), y observa cómo el algoritmo evoluciona la solución y muestra la gráfica.

---

## 🔧 Principales funciones

- `definir_tamano_poblacion_binario(x)`
- `definir_tamano_poblacion_decimal(lind)`
- `crear_funcion_objetivo_math()`
- `obtener_restricciones()`
- `solicitar_datos_problema(codificacion)`
- `algoritmo_binario(...)`
- `algoritmo_entero(...)`
- `algoritmo_decimal(...)`
- `plot_fitness()`
- `plot_decimal_fitness_y_funcion(...)`

---

## 🧑‍💻 Autores

- Sergio Alejandro Castro Botero  
- Maria Fernanda Gomez Narvaez

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Puedes usarlo y modificarlo libremente.

