# app.py

import streamlit as st
import matplotlib.pyplot as plt
from Proyecto_SIC import algoritmo_binario, definir_tamano_poblacion_binario, mejores_fitness

st.set_page_config(page_title="Algoritmo Genético", layout="centered")

st.title("🔬 Algoritmo Genético - Knapsack")

# ==== Parámetros generales ====
codificacion = st.selectbox("Tipo de codificación", ["binario"])  # Puedes extender a entero/decimal
modo = st.radio("¿Maximizar o Minimizar?", ["max", "min"])
tipo_seleccion = st.selectbox("Tipo de selección", ["ruleta", "torneo"])
tam_torneo = st.slider("Tamaño del torneo", 2, 10, value=3) if tipo_seleccion == "torneo" else 0
tipo_cruce = st.selectbox("Tipo de cruce", ["1punto", "2puntos"])
tipo_mutacion = st.selectbox("Tipo de mutación", ["normal", "intercambio"])
elitismo = st.checkbox("Usar elitismo")
num_elites = st.slider("Número de élites", 1, 5, value=1) if elitismo else 0
Pcruce = st.slider("Probabilidad de cruce", 0.0, 1.0, 0.7)
Pmuta = st.slider("Probabilidad de mutación", 0.0, 1.0, 0.1)
num_generaciones = st.number_input("Número de generaciones", min_value=1, value=20)

# ==== Datos del problema ====
st.subheader("🎒 Datos del problema de la mochila")
x = st.number_input("Número de elementos", min_value=1, value=5)
pesos = st.text_input(f"Ingrese {x} pesos separados por espacio", value="3 5 2 7 1")
utilidades = st.text_input(f"Ingrese {x} utilidades separados por espacio", value="10 20 15 5 7")
capacidad = st.number_input("Capacidad máxima de la mochila", min_value=1, value=10)

# ==== Ejecutar ====
if st.button("Ejecutar algoritmo"):
    try:
        pesos = list(map(int, pesos.split()))
        utilidades = list(map(int, utilidades.split()))
        if len(pesos) != x or len(utilidades) != x:
            st.error("⚠️ Los vectores de pesos y utilidades deben tener longitud igual a x.")
        else:
            # Definir tamaño de población
            n = definir_tamano_poblacion_binario(x)

            # Ejecutar algoritmo
            mejores_fitness.clear()
            algoritmo_binario(n, x, pesos, utilidades, capacidad, tipo_seleccion,
                              elitismo, num_elites, tam_torneo, Pcruce, Pmuta,
                              tipo_cruce, tipo_mutacion, modo, num_generaciones)

            # Mostrar gráfico
            st.success("✅ Ejecución completa")

            fig, ax = plt.subplots()
            ax.plot(mejores_fitness)
            ax.set_xlabel("Generación")
            ax.set_ylabel("Mejor Fitness")
            ax.set_title("Evolución del Fitness")
            st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Error: {e}")
