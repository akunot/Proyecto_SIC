import random
import numpy as np
import math as m
import matplotlib.pyplot as plt

# ==== VARIABLES GLOBALES ====
mejores_fitness = []
acumulado = None

# ==== CONFIGURACIÓN PARA DECIMAL ====
Xmin = 0
Xmax = 12
num_decimales = 1

# ==== SOLICITAR DATOS DEL PROBLEMA DE MOCHILA ====
def solicitar_datos_problema(codificacion):
    if codificacion == "decimal":
        return None, None, None, None, None  # No aplica para decimal
    print("\nCONFIGURACIÓN DEL PROBLEMA DE LA MOCHILA")
    x = int(input("Número de elementos: "))
    pesos = list(map(int, input("Ingrese los pesos separados por espacio: ").split()))
    utilidad = list(map(int, input("Ingrese las utilidades separados por espacio: ").split()))
    capacidad_max = int(input("Capacidad máxima de la mochila: "))
    assert len(pesos) == x and len(utilidad) == x, "Datos inconsistentes."

    diccionario_cromosomas = None
    if codificacion == "entero":
        diccionario_cromosomas = {}
        for i in range(x):
            cantidad = int(input(f"Ingrese cantidad máxima para x{i+1}: "))
            diccionario_cromosomas[f'x{i+1}'] = int(m.log2(cantidad + 1)) if cantidad > 1 else 1

    return x, pesos, utilidad, capacidad_max, diccionario_cromosomas

# ==== ALGORITMO BINARIO ====
def algoritmo_binario(n, x, pesos, utilidad, capacidad_max, tipo_seleccion, usar_elitismo, num_elites, tam_torneo, Pcruce, Pmuta, tipo_cruce, tipo_mutacion):
    global mejores_fitness, acumulado
    acumulado = np.zeros(n)

    def generar_poblacion():
        return np.random.randint(0, 2, (n, x))

    def evaluar(pobl):
        fitness = np.zeros(n)
        for i in range(n):
            peso_total = np.sum(pobl[i] * pesos)
            utilidad_total = np.sum(pobl[i] * utilidad)
            fitness[i] = utilidad_total if peso_total <= capacidad_max else 0
        return fitness, np.sum(fitness)

    def mutar(ind):
        if tipo_mutacion == "intercambio" and len(ind) >= 2 and random.random() < Pmuta:
            i, j = random.sample(range(len(ind)), 2)
            ind[i], ind[j] = ind[j], ind[i]
        else:
            for i in range(len(ind)):
                if random.random() < Pmuta:
                    ind[i] = 1 - ind[i]
        return ind

    def cruzar(p1, p2):
        if random.random() < Pcruce:
            if tipo_cruce == "2puntos":
                cp1 = random.randint(1, x - 2)
                cp2 = random.randint(cp1 + 1, x - 1)
                h1 = np.concatenate((p1[:cp1], p2[cp1:cp2], p1[cp2:]))
                h2 = np.concatenate((p2[:cp1], p1[cp1:cp2], p2[cp2:]))
            else:
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

    def seleccion_padre(fit):
        if tipo_seleccion == "ruleta":
            r = np.random.rand()
            for i in range(n):
                if acumulado[i] > r:
                    return poblIt[i]
            return poblIt[-1]
        else:
            indices = np.random.choice(n, tam_torneo, replace=False)
            return poblIt[max(indices, key=lambda i: fit[i])]

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
            p1 = seleccion_padre(fitness)
            p2 = seleccion_padre(fitness)
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

# ==== ALGORITMO ENTERO ====
def algoritmo_entero(n, x, pesos, utilidad, capacidad_max, tipo_seleccion, usar_elitismo, num_elites, tam_torneo, Pcruce, Pmuta, diccionario_cromosomas, tipo_cruce, tipo_mutacion):
    global mejores_fitness, acumulado
    acumulado = np.zeros(n)

    def generar_poblacion():
        poblacion = []
        for _ in range(n):
            individuo = []
            for j in range(x):
                bits = diccionario_cromosomas[f'x{j+1}']
                valor = random.randint(0, (2**bits) - 1)
                individuo.append(valor)
            poblacion.append(individuo)
        return np.array(poblacion)

    def evaluar(pobl):
        fitness = np.zeros(n)
        for i in range(n):
            peso_total = np.sum(pobl[i] * pesos)
            utilidad_total = np.sum(pobl[i] * utilidad)
            fitness[i] = utilidad_total if peso_total <= capacidad_max else 0
        return fitness, np.sum(fitness)

    def mutar(ind):
        if tipo_mutacion == "intercambio" and len(ind) >= 2 and random.random() < Pmuta:
            i, j = random.sample(range(len(ind)), 2)
            ind[i], ind[j] = ind[j], ind[i]
        else:
            for i in range(len(ind)):
                if random.random() < Pmuta:
                    bits = diccionario_cromosomas[f'x{i+1}']
                    ind[i] = random.randint(0, (2 ** bits) - 1)
        return ind

    def cruzar(p1, p2):
        if random.random() < Pcruce:
            if tipo_cruce == "2puntos":
                cp1 = random.randint(1, x - 2)
                cp2 = random.randint(cp1 + 1, x - 1)
                h1 = np.concatenate((p1[:cp1], p2[cp1:cp2], p1[cp2:]))
                h2 = np.concatenate((p2[:cp1], p1[cp1:cp2], p2[cp2:]))
            else:
                cp = random.randint(1, x - 1)
                h1 = np.concatenate((p1[:cp], p2[cp:]))
                h2 = np.concatenate((p2[:cp], p1[cp:]))
        else:
            h1, h2 = p1.copy(), p2.copy()
        return mutar(h1), mutar(h2)

    def imprime(n, total, fitness, poblIt):
        global mejores_fitness
        print("\nTabla Iteración:")
        acum = 0
        for i in range(n):
            prob = fitness[i] / total if total else 0
            acum += prob
            acumulado[i] = acum
            print(f"{i+1:<3} {str(poblIt[i]):<30} {fitness[i]:<10.2f} {prob:<10.3f} {acum:<10.3f}")
        mejores_fitness.append(max(fitness))

    def seleccion_padre(fit):
        if tipo_seleccion == "ruleta":
            r = np.random.rand()
            for i in range(n):
                if acumulado[i] > r:
                    return poblIt[i]
            return poblIt[-1]
        else:
            indices = np.random.choice(n, tam_torneo, replace=False)
            return poblIt[max(indices, key=lambda i: fit[i])]

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
            p1 = seleccion_padre(fitness)
            p2 = seleccion_padre(fitness)
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

# ==== ALGORITMO DECIMAL ====
def algoritmo_decimal(n, Pcruce, Pmuta, tipo_cruce, tipo_mutacion, usar_elitismo, num_elites, tipo_seleccion, tam_torneo):
    global mejores_fitness, acumulado
    lind = int(round(m.log2((Xmax - Xmin) * 10**num_decimales)))
    acumulado = np.zeros(n)

    def generar_poblacion():
        return np.random.randint(0, 2, (n, lind))

    def evaluar(poblIt):
        fitness = np.zeros(n)
        xi = np.zeros(n)
        total = 0
        for i in range(n):
            val = int("".join(str(b) for b in poblIt[i]), 2)
            xi[i] = Xmin + val * (Xmax - Xmin) / (2**lind - 1)
            if 3 < xi[i] < 4: # Si el valor obtenido es menor o igual a 3 o mayor o igual a 4, se considera que el valor no es válido
                fitness[i] = -np.inf
            else:
                fitness[i] = 32 * xi[i] - 2 * xi[i]**2 # Se calcula la funcion objetivo
                total += 0 if fitness[i] == -np.inf else fitness[i]
        return fitness, total, xi

    def mutar(ind):
        if tipo_mutacion == "intercambio" and random.random() < Pmuta:
            i, j = random.sample(range(len(ind)), 2)
            ind[i], ind[j] = ind[j], ind[i]
        else:
            for i in range(len(ind)):
                if random.random() < Pmuta:
                    ind[i] = 1 - ind[i]
        return ind

    def cruzar(p1, p2):
        if random.random() < Pcruce:
            if tipo_cruce == "2puntos":
                cp1 = random.randint(1, lind - 2)
                cp2 = random.randint(cp1 + 1, lind - 1)
                h1 = np.concatenate((p1[:cp1], p2[cp1:cp2], p1[cp2:]))
                h2 = np.concatenate((p2[:cp1], p1[cp1:cp2], p2[cp2:]))
            else:
                cp = random.randint(1, lind - 1)
                h1 = np.concatenate((p1[:cp], p2[cp:]))
                h2 = np.concatenate((p2[:cp], p1[cp:]))
        else:
            h1, h2 = p1.copy(), p2.copy()
        return mutar(h1), mutar(h2)

    def imprime(poblIt, fitness, xi, total):
        global mejores_fitness
        acum = 0
        for i in range(n):
            prob = 0 if fitness[i] == -np.inf or total == 0 else fitness[i] / total
            acum += prob
            acumulado[i] = acum
            fit_str = f"{fitness[i]:.3f}" if fitness[i] != -np.inf else "X"
            print(f"{i+1:<3} {str(poblIt[i])} xi={xi[i]:.3f} fitness={fit_str} prob={prob:.3f} acum={acum:.3f}")
        mejores_fitness.append(np.max(fitness[fitness != -np.inf]) if np.any(fitness != -np.inf) else 0)


    def seleccion_padre(fit):
        if tipo_seleccion == "ruleta":
            r = np.random.rand()
            for i in range(n):
                if acumulado[i] > r:
                    return poblIt[i]
            return poblIt[-1]
        else:
            indices = np.random.choice(n, tam_torneo, replace=False)
            return poblIt[max(indices, key=lambda i: fit[i])]

    poblIt = generar_poblacion()
    fitness, total, xi = evaluar(poblIt)
    imprime(poblIt, fitness, xi, total)

    for gen in range(20):
        print(f"\n--- Generación {gen + 1} ---")

        elites = []
        if usar_elitismo:
            elites_idx = np.argsort(fitness)[-num_elites:]
            elites = [poblIt[i].copy() for i in elites_idx]

        nueva = []
        while len(nueva) < n:
            p1 = seleccion_padre(fitness)
            p2 = seleccion_padre(fitness)
            h1, h2 = cruzar(p1, p2)
            nueva.extend([h1, h2])

        poblIt = np.array(nueva[:n])

        if usar_elitismo:
            reemplazar_idx = np.argsort(fitness)[:num_elites]
            for i, idx in enumerate(reemplazar_idx):
                poblIt[idx] = elites[i]

        fitness, total, xi = evaluar(poblIt)
        imprime(poblIt, fitness, xi, total)

# ==== PLOT ====
def plot_fitness():
    plt.plot(mejores_fitness)
    plt.xlabel('Iteración')
    plt.ylabel('Mejor Fitness')
    plt.title('Evolución del Fitness')
    plt.grid(True)
    plt.show()

# ==== FLUJO PRINCIPAL ====
print("\nCONFIGURACIÓN DEL ALGORITMO GENÉTICO")
tipo_codificacion = input("Tipo de codificación (binario / entero / decimal): ").strip().lower()
tipo_seleccion = input("Tipo de selección (ruleta / torneo): ").strip().lower()
tipo_cruce = input("Tipo de cruce (1punto / 2puntos): ").strip().lower()
tipo_mutacion = input("Tipo de mutación (normal / intercambio): ").strip().lower()
usar_elitismo = input("¿Usar elitismo? (s/n): ").strip().lower() == 's'
num_elites = int(input("Cantidad de élites (si aplica): ")) if usar_elitismo else 0
tam_torneo = int(input("Tamaño del torneo (si aplica): ")) if tipo_seleccion == "torneo" else 0
Pcruce = float(input("Probabilidad de cruce (0.0 a 1.0): "))
Pmuta = float(input("Probabilidad de mutación (0.0 a 1.0): "))
n = int(input("Tamaño de la población: "))

x, pesos, utilidad, capacidad_max, diccionario_cromosomas = solicitar_datos_problema(tipo_codificacion)

if tipo_codificacion == "binario":
    algoritmo_binario(n, x, pesos, utilidad, capacidad_max, tipo_seleccion, usar_elitismo, num_elites, tam_torneo, Pcruce, Pmuta, tipo_cruce, tipo_mutacion)
elif tipo_codificacion == "entero":
    algoritmo_entero(n, x, pesos, utilidad, capacidad_max, tipo_seleccion, usar_elitismo, num_elites, tam_torneo, Pcruce, Pmuta, diccionario_cromosomas, tipo_cruce, tipo_mutacion)
elif tipo_codificacion == "decimal":
    algoritmo_decimal(n, Pcruce, Pmuta, tipo_cruce, tipo_mutacion, usar_elitismo, num_elites, tipo_seleccion, tam_torneo)
else:
    print("Codificación no soportada.")

plot_fitness()