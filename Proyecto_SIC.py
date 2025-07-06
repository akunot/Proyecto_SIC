# Algoritmo Genético Configurable con Entrada del Usuario y Soporte para Binario / Entero

import random
import numpy as np
import matplotlib.pyplot as plt
import math as m

# ==== VARIABLES GLOBALES ====
mejores_fitness = []
acumulado = None


def solicitar_datos_problema(codificacion):
    print("\nCONFIGURACIÓN DEL PROBLEMA DE LA MOCHILA")
    x = int(input("Número de elementos: "))
    print("Ingrese los pesos de los elementos separados por espacio:")
    pesos = list(map(int, input().split()))
    print("Ingrese las utilidades de los elementos separados por espacio:")
    utilidad = list(map(int, input().split()))
    capacidad_max = int(input("Capacidad máxima de la mochila: "))
    assert len(pesos) == x and len(utilidad) == x, "Datos inconsistentes."

    diccionario_cromosomas = None
    if codificacion == "entero":
        diccionario_cromosomas = {}
        for i in range(x):
            cantidad = int(input(f"Ingrese la cantidad de objetos para x{i+1}: "))
            diccionario_cromosomas[f'x{i+1}'] = int(m.log(cantidad + 1, 2)) if cantidad > 1 else 1

    return x, pesos, utilidad, capacidad_max, diccionario_cromosomas

# ==== ALGORITMO PARA CODIFICACIÓN BINARIA ====
def algoritmo_binario(n, x, pesos, utilidad, capacidad_max, tipo_seleccion, usar_elitismo, num_elites, tam_torneo, Pcruce, Pmuta):
    global mejores_fitness, acumulado
    acumulado = np.zeros(n)

    def generar_poblacion():
        return np.random.randint(0, 2, (n, x))

    def evaluar(pobl):
        fitness = np.zeros(n)
        for i in range(n):
            cantidad = pobl[i]
            peso_total = np.sum(cantidad * pesos)
            utilidad_total = np.sum(cantidad * utilidad)
            if peso_total > capacidad_max:
                fitness[i] = 0
            else:
                fitness[i] = utilidad_total
        return fitness, np.sum(fitness)

    def mutar(ind):
        for i in range(len(ind)):
            if random.random() < Pmuta:
                ind[i] = 1 - ind[i]
        return ind

    def cruzar(p1, p2):
        if random.random() < Pcruce:
            cp = random.randint(1, x - 1)
            h1 = np.concatenate((p1[:cp], p2[cp:]))
            h2 = np.concatenate((p2[:cp], p1[cp:]))
        else:
            h1, h2 = p1.copy(), p2.copy()
        return mutar(h1), mutar(h2)

    def imprime(pobl, fitness, total):
        global mejores_fitness
        acum = 0
        print("\nTabla de Evaluación:")
        for i in range(n):
            prob = fitness[i] / total if total > 0 else 0
            acum += prob
            acumulado[i] = acum
            print(f"{i+1:<3} {str(pobl[i]):<40} {fitness[i]:<10.2f} {prob:<10.3f} {acum:<10.3f}")
        mejores_fitness.append(max(fitness))

    def seleccion_ruleta():
        r = np.random.rand()
        for i in range(n):
            if acumulado[i] > r:
                return poblIt[i]
        return poblIt[-1]

    def seleccion_torneo(fit):
        indices = np.random.choice(n, tam_torneo, replace=False)
        mejor = indices[0]
        for idx in indices[1:]:
            if fit[idx] > fit[mejor]:
                mejor = idx
        return poblIt[mejor]

    def seleccionar_padre(fit):
        return seleccion_ruleta() if tipo_seleccion == "ruleta" else seleccion_torneo(fit)

    poblIt = generar_poblacion()
    fitness, total = evaluar(poblIt)
    imprime(poblIt, fitness, total)

    for generacion in range(20):
        print(f"\n--- Generación {generacion + 1} ---")

        elites = []
        if usar_elitismo:
            elites_idx = np.argsort(fitness)[-num_elites:]
            elites = [poblIt[i].copy() for i in elites_idx]
            elites_fitness = [fitness[i] for i in elites_idx]

        nueva_poblacion = []
        while len(nueva_poblacion) < n:
            p1 = seleccionar_padre(fitness)
            p2 = seleccionar_padre(fitness)
            h1, h2 = cruzar(p1, p2)
            nueva_poblacion.extend([h1, h2])

        poblIt = np.array(nueva_poblacion[:n])
        fitness, total = evaluar(poblIt)

        if usar_elitismo:
            reemplazar_idx = np.argsort(fitness)[:num_elites]
            for i, idx in enumerate(reemplazar_idx):
                poblIt[idx] = elites[i]
                fitness[idx] = elites_fitness[i]

        imprime(poblIt, fitness, total)

# ==== ALGORITMO PARA CODIFICACIÓN ENTERA ====
def algoritmo_entero(n, x, pesos, utilidad, capacidad_max, tipo_seleccion, usar_elitismo, num_elites, tam_torneo, Pcruce, Pmuta, diccionario_cromosomas):
    global mejores_fitness, acumulado
    acumulado = np.zeros(n)

    def generar_poblacion():
        poblacion = []
        for _ in range(n):
            individuo = []
            for j in range(x):
                bits = diccionario_cromosomas[f'x{j+1}']
                valor_max = (2 ** bits) - 1
                valor = random.randint(0, valor_max)
                individuo.append(valor)
            poblacion.append(individuo)
        return np.array(poblacion)

    def evaluar(pobl):
        fitness = np.zeros(n)
        for i in range(n):
            cantidad = pobl[i]
            peso_total = np.sum(cantidad * pesos)
            utilidad_total = np.sum(cantidad * utilidad)
            if peso_total > capacidad_max:
                fitness[i] = 0
            else:
                fitness[i] = utilidad_total
        return fitness, np.sum(fitness)

    def mutar(ind):
        for i in range(len(ind)):
            if random.random() < Pmuta:
                bits = diccionario_cromosomas[f'x{i+1}']
                max_valor = (2 ** bits) - 1
                ind[i] = random.randint(0, max_valor)
        return ind

    def cruzar(p1, p2):
        if random.random() < Pcruce:
            cp = random.randint(1, x - 1)
            h1 = np.concatenate((p1[:cp], p2[cp:]))
            h2 = np.concatenate((p2[:cp], p1[cp:]))
        else:
            h1, h2 = p1.copy(), p2.copy()
        return mutar(h1), mutar(h2)

    def imprime(n, total, fitness, poblIt):
        global mejores_fitness
        print("\nTabla Iteración:\n")
        print(f"{'Individuo':<10} {'Genotipo (binario)':<40} {'Fenotipo (cantidades)':<25} {'Fitness':<10} {'Probabilidad':<15} {'Acumulado':<10}")
        acumula = 0

        for i in range(n):
            probab = fitness[i] / total if total != 0 else 0
            acumula += probab

            decimal_values = [int(val) for val in poblIt[i]]
            genotipo_binario = []
            for j, val in enumerate(decimal_values):
                clave = f'x{j+1}'
                bits = diccionario_cromosomas[clave]
                bin_str = format(int(val), f'0{bits}b')
                genotipo_binario.append(bin_str)

            genotipo_str = ' '.join(genotipo_binario)

            print(f"{i+1:<10} {genotipo_str:<40} {str(decimal_values):<25} {fitness[i]:<10.2f} {probab:<15.3f} {acumula:<10.3f}")
            acumulado[i] = acumula

        print("Suma Z:      ", total)
        mejores_fitness.append(float(max(fitness)))
        return acumulado

    def seleccion_ruleta():
        r = np.random.rand()
        for i in range(n):
            if acumulado[i] > r:
                return poblIt[i]
        return poblIt[-1]

    def seleccion_torneo(fit):
        indices = np.random.choice(n, tam_torneo, replace=False)
        mejor = indices[0]
        for idx in indices[1:]:
            if fit[idx] > fit[mejor]:
                mejor = idx
        return poblIt[mejor]

    def seleccionar_padre(fit):
        return seleccion_ruleta() if tipo_seleccion == "ruleta" else seleccion_torneo(fit)

    poblIt = generar_poblacion()
    fitness, total = evaluar(poblIt)
    imprime(n, total, fitness, poblIt)

    for generacion in range(20):
        print(f"\n--- Generación {generacion + 1} ---")

        elites = []
        if usar_elitismo:
            elites_idx = np.argsort(fitness)[-num_elites:]
            elites = [poblIt[i].copy() for i in elites_idx]
            elites_fitness = [fitness[i] for i in elites_idx]

        nueva_poblacion = []
        while len(nueva_poblacion) < n:
            p1 = seleccionar_padre(fitness)
            p2 = seleccionar_padre(fitness)
            h1, h2 = cruzar(p1, p2)
            nueva_poblacion.extend([h1, h2])

        poblIt = np.array(nueva_poblacion[:n])
        fitness, total = evaluar(poblIt)

        if usar_elitismo:
            reemplazar_idx = np.argsort(fitness)[:num_elites]
            for i, idx in enumerate(reemplazar_idx):
                poblIt[idx] = elites[i]
                fitness[idx] = elites_fitness[i]

        imprime(n, total, fitness, poblIt)

# ==== GRÁFICA ====
def plot_fitness():
    plt.plot(mejores_fitness)
    plt.xlabel('Iteración')
    plt.ylabel('Mejor Fitness')
    plt.title('Evolución del Fitness')
    plt.grid(True)
    plt.show()

# ==== FLUJO PRINCIPAL ====
print("\nCONFIGURACIÓN DEL ALGORITMO GENÉTICO")
tipo_codificacion = input("Tipo de codificación (binario / entero): ").strip().lower()
tipo_seleccion = input("Tipo de selección (ruleta / torneo): ").strip().lower()
usar_elitismo = input("Usar elitismo? (s/n): ").strip().lower() == 's'
num_elites = int(input("Cantidad de individuos élite (si aplica): ")) if usar_elitismo else 0
tam_torneo = int(input("Tamaño del torneo (si aplica): "))
Pcruce = float(input("Probabilidad de cruce (0.0 a 1.0): "))
Pmuta = float(input("Probabilidad de mutación (0.0 a 1.0): "))
n = int(input("Tamaño de la población: "))

x, pesos, utilidad, capacidad_max, diccionario_cromosomas = solicitar_datos_problema(tipo_codificacion)

if tipo_codificacion == "binario":
    algoritmo_binario(n, x, pesos, utilidad, capacidad_max, tipo_seleccion, usar_elitismo, num_elites, tam_torneo, Pcruce, Pmuta)
elif tipo_codificacion == "entero":
    algoritmo_entero(n, x, pesos, utilidad, capacidad_max, tipo_seleccion, usar_elitismo, num_elites, tam_torneo, Pcruce, Pmuta, diccionario_cromosomas)
else:
    print("Codificación no soportada")

plot_fitness()