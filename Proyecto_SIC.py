import random
import numpy as np
import math as m
import matplotlib.pyplot as plt

# ==== VARIABLES GLOBALES ====
mejores_fitness = []
acumulado = None

# ==== CONFIGURACIÓN PARA DECIMAL ====
Xmin = 0
Xmax = 10
num_decimales = 2

# ==== CONFIGURACIÓN DEL TAMAÑO DE LA POBLACIÓN (solo para decimal) ====
def definir_tamano_poblacion_decimal(lind):
    tam_min = max(10, lind * 2)
    tam_max = max(20, lind * 5)
    
    print(f"\n🔢 Sugerencia: usa una población entre {tam_min} y {tam_max} individuos.")
    while True:
        try:
            n = int(input(f"Ingrese el tamaño de la población (entre {tam_min} y {tam_max}): "))
            if tam_min <= n <= tam_max:
                return n
            else:
                print(f"❌ Debe estar entre {tam_min} y {tam_max}.")
        except ValueError:
            print("❌ Ingresa un número válido.")


# ==== CONFIGURACIÓN DEL TAMAÑO DE LA POBLACIÓN (solo para binario) ====
def definir_tamano_poblacion_binario(x):
    lim_inf = max(4, x)
    lim_sup = max(10, x * 2)

    while True:
        try:
            print(f"\nEscoge el tamaño de la población entre {lim_inf} y {lim_sup}:")
            n = int(input("Tamaño de la población: "))
            if lim_inf <= n <= lim_sup:
                return n
            else:
                print(f"❌ El valor debe estar entre {lim_inf} y {lim_sup}.")
        except ValueError:
            print("❌ Ingresa un número válido.")



def solicitar_datos_problema(codificacion):
    if codificacion == "decimal":
        return None, None, None, None, None, None  # No aplica para decimal

    print("\nCONFIGURACIÓN DEL PROBLEMA DE LA MOCHILA")

    # Validar número de elementos
    while True:
        try:
            x = int(input("Número de elementos: "))
            if x > 0:
                break
            print("❌ Debe ser un número mayor que 0.")
        except ValueError:
            print("❌ Ingresa un número entero válido.")

    # Validar pesos
    while True:
        try:
            pesos = list(map(int, input(f"Ingrese {x} pesos separados por espacio: ").split()))
            if len(pesos) == x and all(p >= 0 for p in pesos):
                break
            print(f"❌ Debes ingresar exactamente {x} números enteros no negativos.")
        except ValueError:
            print("❌ Asegúrate de ingresar solo números enteros.")

    # Validar utilidades
    while True:
        try:
            utilidad = list(map(int, input(f"Ingrese {x} utilidades separadas por espacio: ").split()))
            if len(utilidad) == x:
                break
            print(f"❌ Debes ingresar exactamente {x} números.")
        except ValueError:
            print("❌ Asegúrate de ingresar solo números enteros.")

    # Validar capacidad máxima
    while True:
        try:
            capacidad_max = int(input("Capacidad máxima de la mochila: "))
            if capacidad_max > 0:
                break
            print("❌ Debe ser un número entero positivo.")
        except ValueError:
            print("❌ Ingresa un número entero válido.")

    diccionario_cromosomas = None
    n = None

    if codificacion == "entero":
        diccionario_cromosomas = {}
        longitud_total = 0

        for i in range(x):
            while True:
                try:
                    cantidad = int(input(f"Ingrese cantidad máxima para x{i+1} (Ej: si x{i+1} va de 0 a 10, escribe 10): "))
                    if cantidad >= 0:
                        bits = int(m.ceil(m.log2(cantidad + 1))) if cantidad > 0 else 1
                        diccionario_cromosomas[f'x{i+1}'] = bits
                        longitud_total += bits
                        break
                    else:
                        print("❌ Debe ser un número entero mayor o igual a 0.")
                except ValueError:
                    print("❌ Ingresa un número entero válido.")

        # Sugerencia del tamaño de población
        min_pobl = longitud_total * 2
        max_pobl = longitud_total * 5

        while True:
            try:
                print(f"\n🔧 Sugerencia: el tamaño de la población debería estar entre {min_pobl} y {max_pobl} según la complejidad del cromosoma.")
                n = int(input(f"Ingrese el tamaño de la población (entre {min_pobl} y {max_pobl}): "))
                if min_pobl <= n <= max_pobl:
                    break
                else:
                    print("❌ Tamaño fuera del rango sugerido.")
            except ValueError:
                print("❌ Ingresa un número entero válido.")

    return x, pesos, utilidad, capacidad_max, diccionario_cromosomas, n


def algoritmo_binario(n, x, pesos, utilidad, capacidad_max, tipo_seleccion, usar_elitismo, num_elites,
                      tam_torneo, Pcruce, Pmuta, tipo_cruce, tipo_mutacion, maximizar_minimizar):
    global mejores_fitness, acumulado
    acumulado = np.zeros(n)

    def generar_poblacion():
        pobl = np.random.randint(0, 2, (n, x))
        for i in range(n):
            if not pobl[i].any():  # Todos ceros
                num_items = random.randint(1, min(3, x))  # entre 1 y 3 objetos como máximo
                indices = random.sample(range(x), num_items)
                for idx in indices:
                    pobl[i][idx] = 1
        return pobl


    def evaluar(pobl, modo):
        valores = np.zeros(n)
        penalizados = np.zeros(n, dtype=bool)

        for i in range(n):
            peso_total = np.sum(pobl[i] * pesos)
            utilidad_total = np.sum(pobl[i] * utilidad)
            if peso_total == 0:
                penalizados[i] = True
                valores[i] = float('inf') if modo == "min" else 0
            elif peso_total <= capacidad_max:
                valores[i] = utilidad_total
            else:
                penalizados[i] = True
                valores[i] = float('inf') if modo == "min" else 0


        # Convertir valores a fitness
        if modo == "max":
            fitness = valores
        else:
            fitness = np.zeros(n)
            for i in range(n):
                if penalizados[i]:
                    fitness[i] = 0
                else:
                    fitness[i] = 1 / (1 + valores[i])  # menor valor => mayor fitness

        return fitness, valores, np.sum(fitness)

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

    def imprime(pobl, fitness, valores, total, modo, pesos, utilidad):
        global mejores_fitness
        print("\nTabla de Evaluación:")
        print(f"{'Ind':<4} {'Genotipo (Binario)':<25} {'Fitness':<10} {'Utilidad':<10} {'Peso':<10} {'Prob':<10} {'Acum':<10}")

        acum = 0
        acumulado_local = []

        for i in range(len(pobl)):
            prob = fitness[i] / total if total > 0 else 0
            acum += prob
            acumulado_local.append(acum)

            binario_str = ''.join(str(int(b)) for b in pobl[i])
            peso_total = sum(pobl[i][j] * pesos[j] for j in range(len(pobl[i])))
            utilidad_total = sum(pobl[i][j] * utilidad[j] for j in range(len(pobl[i])))

            print(f"{i+1:<4} {binario_str:<25} {fitness[i]:<10.4f} {utilidad_total:<10} {peso_total:<10} {prob:<10.3f} {acum:<10.3f}")

        print("Suma Z:", total)

        if modo == "max":
            mejores_fitness.append(float(max(fitness)))
        else:
            valores_validos = [valores[i] for i in range(len(valores)) if valores[i] != float('inf')]
            if valores_validos:
                mejores_fitness.append(float(min(valores_validos)))
            else:
                mejores_fitness.append(float('inf'))

        return acumulado_local


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

    # Inicializar
    poblIt = generar_poblacion()
    fitness, valores, total = evaluar(poblIt, maximizar_minimizar)
    imprime(poblIt, fitness, valores, total, maximizar_minimizar, pesos, utilidad)

    for generacion in range(20):
        print(f"\n--- Generación {generacion + 1} ---")

        elites = []
        elites_fitness = []
        if usar_elitismo:
            if maximizar_minimizar == "max":
                elites_idx = np.argsort(fitness)[-num_elites:]  # Mejores fitness
            else:  # min
                # Filtrar individuos válidos (valor distinto de inf)
                viables_idx = [i for i in range(n) if valores[i] != float('inf')]
                if len(viables_idx) >= num_elites:
                    # Escoger los de menor valor (mejores en minimización)
                    elites_idx = sorted(viables_idx, key=lambda i: valores[i])[:num_elites]
                else:
                    elites_idx = []  # No se seleccionan élites si no hay suficientes válidos

            elites = [poblIt[i].copy() for i in elites_idx]
            elites_fitness = [fitness[i] for i in elites_idx]

        # === REPRODUCCIÓN ===
        nueva_poblacion = []
        while len(nueva_poblacion) < n:
            p1 = seleccion_padre(fitness)
            p2 = seleccion_padre(fitness)
            h1, h2 = cruzar(p1, p2)
            nueva_poblacion.extend([h1, h2])

        poblIt = np.array(nueva_poblacion[:n])

        # === EVALUACIÓN ===
        fitness, valores, total = evaluar(poblIt, maximizar_minimizar)

        # === REINSERCIÓN DE ÉLITES ===
        if usar_elitismo and elites:
            if maximizar_minimizar == "max":
                reemplazar_idx = np.argsort(fitness)[:num_elites]  # Peores fitness
            else:
                # Solo entre los peores válidos
                viables_idx = [i for i in range(n) if valores[i] != float('inf')]
                if len(viables_idx) >= num_elites:
                    reemplazar_idx = sorted(viables_idx, key=lambda i: valores[i], reverse=True)[:num_elites]
                else:
                    reemplazar_idx = []

            for i, idx in enumerate(reemplazar_idx):
                poblIt[idx] = elites[i]
                fitness[idx] = elites_fitness[i]

            # Re-evaluar tras insertar élites
            fitness, valores, total = evaluar(poblIt, maximizar_minimizar)

        imprime(poblIt, fitness, valores, total, maximizar_minimizar, pesos, utilidad)

def algoritmo_entero(n, x, pesos, utilidad, capacidad_max, tipo_seleccion, usar_elitismo,
                     num_elites, tam_torneo, Pcruce, Pmuta, diccionario_cromosomas,
                     tipo_cruce, tipo_mutacion, maximizar_minimizar):
    global mejores_fitness, acumulado
    acumulado = np.zeros(n)

    def generar_individuo_valido():
        while True:
            individuo = []
            for j in range(x):
                bits = diccionario_cromosomas[f'x{j+1}']
                valor = random.randint(0, (2**bits) - 1)
                individuo.append(valor)
            peso_total = np.sum(np.array(individuo) * pesos)
            if peso_total <= capacidad_max and any(individuo):
                return individuo

    def generar_poblacion():
        poblacion = []
        while len(poblacion) < n:
            poblacion.append(generar_individuo_valido())
        return np.array(poblacion)

    def evaluar(pobl, pesos, utilidad, capacidad_max, modo):
        valores = np.zeros(len(pobl))
        penalizados = np.zeros(len(pobl), dtype=bool)

        for i in range(len(pobl)):
            peso_total = np.sum(pobl[i] * pesos)
            utilidad_total = np.sum(pobl[i] * utilidad)

            if peso_total <= capacidad_max:
                valores[i] = utilidad_total
            else:
                penalizados[i] = True
                valores[i] = float('inf') if modo == "min" else 0

        # Calcular fitness
        if modo == "max":
            fitness = valores.copy()
        else:
            fitness = np.zeros(len(valores))
            for i in range(len(valores)):
                if penalizados[i]:
                    fitness[i] = 0
                else:
                    fitness[i] = 1 / (1 + valores[i])

        total = np.sum(fitness)
        return fitness, valores, total

    def mutar(ind):
        if tipo_mutacion == "intercambio" and len(ind) >= 2 and random.random() < Pmuta:
            i, j = random.sample(range(len(ind)), 2)
            ind[i], ind[j] = ind[j], ind[i]
        else:
            for i in range(len(ind)):
                if random.random() < Pmuta:
                    bits = diccionario_cromosomas[f'x{i+1}']
                    ind[i] = random.randint(1, (2 ** bits) - 1)  # Forzar al menos 1
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

    def imprime(n, total, fitness, poblIt, diccionario_cromosomas, maximizar_minimizar):
        global mejores_fitness

        print("\nTabla de Evaluación:")
        print(f"{'Ind':<4} {'Binario (Genotipo)':<40} {'Decimal (Fenotipo)':<25} {'Fitness':<10} {'Utilidad':<10} {'Peso':<10} {'Prob':<10} {'Acum':<10}")
        
        acumula = 0
        acumulado_local = []

        for i in range(n):
            probab = fitness[i] / total if total != 0 else 0
            acumula += probab
            acumulado_local.append(acumula)

            binario_str = []
            decimal_vals = []

            for j, val in enumerate(poblIt[i]):
                bits = diccionario_cromosomas[f'x{j+1}']
                binario = format(val, f'0{bits}b')
                binario_str.append(binario)
                decimal_vals.append(int(val))

            valor_total = np.sum(poblIt[i] * utilidad)
            peso_total = np.sum(poblIt[i] * pesos)

            print(f"{i+1:<4} {str(binario_str):<40} {str(decimal_vals):<25} {fitness[i]:<10.4f} {valor_total:<10} {peso_total:<10} {probab:<10.3f} {acumula:<10.3f}")

        print("Suma Z: ", total)

        if maximizar_minimizar == "max":
            mejores_fitness.append(float(max(fitness)))
        else:
            mejores_fitness.append(float(min([valores[i] for i in range(len(valores)) if valores[i] != float('inf')])))

        return acumulado_local

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

    # INICIALIZACIÓN
    poblIt = generar_poblacion()
    fitness, valores, total = evaluar(poblIt, pesos, utilidad, capacidad_max, maximizar_minimizar)
    imprime(n, total, fitness, poblIt, diccionario_cromosomas, maximizar_minimizar)

    for generacion in range(20):
        print(f"\n--- Generación {generacion + 1} ---")

        elites, elites_fitness = [], []
        if usar_elitismo:
            if maximizar_minimizar == "max":
                elites_idx = np.argsort(fitness)[-num_elites:]  # Mejores fitness
            else:
                elites_idx = np.argsort(valores)[:num_elites]  # Menores valores = mejores

            elites = [poblIt[i].copy() for i in elites_idx]
            elites_fitness = [fitness[i] for i in elites_idx]

        nueva_poblacion = []
        while len(nueva_poblacion) < n:
            p1 = seleccion_padre(fitness)
            p2 = seleccion_padre(fitness)
            h1, h2 = cruzar(p1, p2)
            nueva_poblacion.extend([h1, h2])

        poblIt = np.array(nueva_poblacion[:n])
        fitness, valores, total = evaluar(poblIt, pesos, utilidad, capacidad_max, maximizar_minimizar)

        if usar_elitismo:
            if maximizar_minimizar == "max":
                reemplazar_idx = np.argsort(fitness)[:num_elites]  # Peores fitness
            else:
                reemplazar_idx = np.argsort(valores)[:num_elites]  # Mayores valores = peores

            for i, idx in enumerate(reemplazar_idx):
                poblIt[idx] = elites[i]
                fitness[idx] = elites_fitness[i]

            # Vuelve a evaluar tras insertar élites
            fitness, valores, total = evaluar(poblIt, pesos, utilidad, capacidad_max, maximizar_minimizar)


        imprime(n, total, fitness, poblIt, diccionario_cromosomas, maximizar_minimizar)

# ==== ALGORITMO DECIMAL ====
def algoritmo_decimal(Pcruce, Pmuta, tipo_cruce, tipo_mutacion, usar_elitismo, num_elites, tipo_seleccion, tam_torneo, maximizar_minimizar):
    global mejores_fitness, acumulado
    lind = int(round(m.log2((Xmax - Xmin) * 10**num_decimales)))
    n = definir_tamano_poblacion_decimal(lind)
    acumulado = np.zeros(n)

    # Configuración de restricciones
    usar_restricciones = True  # Activar o desactivar las restricciones
    intervalos_invalidos = [(3, 4), (7, 8)]  # Puedes modificar o vaciar la lista

    def funcion_objetivo(x):
        return (x - 5)**2 + np.sin(x)  # ✅ Cambia aquí la función fácilmente

    def es_invalido(xi):
        return any(a < xi < b for (a, b) in intervalos_invalidos)

    def generar_poblacion():
        return np.random.randint(0, 2, (n, lind))

    def evaluar(poblIt):
        fitness = np.zeros(n)
        xi = np.zeros(n)
        total = 0
        for i in range(n):
            val = int("".join(str(b) for b in poblIt[i]), 2)
            xi[i] = Xmin + val * (Xmax - Xmin) / (2**lind - 1)

            invalido = usar_restricciones and es_invalido(xi[i])
            if invalido:
                fitness[i] = -np.inf if maximizar_minimizar == "max" else np.inf
            else:
                valor_real = funcion_objetivo(xi[i])
                if maximizar_minimizar == "max":
                    fitness[i] = valor_real
                else:
                    fitness[i] = 1 / (1 + abs(valor_real))
                if np.isfinite(fitness[i]):
                    total += fitness[i]
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
        print("\nTabla de Evaluación:")
        print(f"{'Ind':<4} {'Binario':<35} {'xi':<10} {'Fitness':<10} {'Prob':<10} {'Acum':<10}")
        for i in range(n):
            prob = 0 if not np.isfinite(fitness[i]) or total == 0 else fitness[i] / total
            acum += prob
            acumulado[i] = acum
            fit_str = f"{fitness[i]:.3f}" if np.isfinite(fitness[i]) else "X"
            print(f"{i+1:<4} {''.join(map(str, poblIt[i])):<35} {xi[i]:<10.3f} {fit_str:<10} {prob:<10.3f} {acum:<10.3f}")

        if maximizar_minimizar == "max":
            mejores_fitness.append(np.max(fitness[np.isfinite(fitness)]) if np.any(np.isfinite(fitness)) else 0)
        else:
            viables = [funcion_objetivo(x) for x in xi if not es_invalido(x)]
            mejores_fitness.append(min(viables) if viables else float("inf"))

    def seleccion_padre(fit):
        if tipo_seleccion == "ruleta":
            r = np.random.rand()
            for i in range(n):
                if acumulado[i] > r:
                    return poblIt[i]
            return poblIt[-1]
        else:
            indices = np.random.choice(n, tam_torneo, replace=False)
            return poblIt[max(indices, key=lambda i: fit[i]) if maximizar_minimizar == "max" else min(indices, key=lambda i: fit[i])]

    poblIt = generar_poblacion()
    fitness, total, xi = evaluar(poblIt)
    imprime(poblIt, fitness, xi, total)

    for gen in range(20):
        print(f"\n--- Generación {gen + 1} ---")

        elites = []
        if usar_elitismo:
            elites_idx = np.argsort(fitness)[-num_elites:] if maximizar_minimizar == "max" else np.argsort(fitness)[:num_elites]
            elites = [poblIt[i].copy() for i in elites_idx]

        nueva = []
        while len(nueva) < n:
            p1 = seleccion_padre(fitness)
            p2 = seleccion_padre(fitness)
            h1, h2 = cruzar(p1, p2)
            nueva.extend([h1, h2])

        poblIt = np.array(nueva[:n])

        if usar_elitismo and elites:
            reemplazar_idx = np.argsort(fitness)[:num_elites] if maximizar_minimizar == "max" else np.argsort(fitness)[-num_elites:]
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

print("\nCONFIGURACIÓN DEL ALGORITMO GENÉTICO")

# Tipo de codificación
while True:
    tipo_codificacion = input("Tipo de codificación (binario / entero / decimal): ").strip().lower()
    if tipo_codificacion in ["binario", "entero", "decimal"]:
        break
    print("❌ Opción no válida. Elige: binario, entero o decimal.")

# Modo de optimización
while True:
    maximizar_minimizar = input("¿Maximizar o minimizar? (max/min): ").strip().lower()
    if maximizar_minimizar in ["max", "min"]:
        break
    print("❌ Escribe 'max' para maximizar o 'min' para minimizar.")

# Tipo de selección
while True:
    tipo_seleccion = input("Tipo de selección (ruleta / torneo): ").strip().lower()
    if tipo_seleccion in ["ruleta", "torneo"]:
        break
    print("❌ Selección no válida. Usa: ruleta o torneo.")

# Tamaño del torneo (solo si aplica)
tam_torneo = 0
if tipo_seleccion == "torneo":
    while True:
        try:
            tam_torneo = int(input("Tamaño del torneo (entero > 1): "))
            if tam_torneo > 1:
                break
            print("❌ El tamaño debe ser mayor que 1.")
        except ValueError:
            print("❌ Ingrese un número entero válido.")

# Tipo de cruce
while True:
    tipo_cruce = input("Tipo de cruce (1punto / 2puntos): ").strip().lower()
    if tipo_cruce in ["1punto", "2puntos"]:
        break
    print("❌ Cruce no válido. Usa: 1punto o 2puntos.")

# Tipo de mutación
while True:
    tipo_mutacion = input("Tipo de mutación (normal / intercambio): ").strip().lower()
    if tipo_mutacion in ["normal", "intercambio"]:
        break
    print("❌ Mutación no válida. Usa: normal o intercambio.")

# Elitismo
while True:
    elitismo_input = input("¿Usar elitismo? (s/n): ").strip().lower()
    if elitismo_input in ["s", "n"]:
        usar_elitismo = elitismo_input == 's'
        break
    print("❌ Opción inválida. Escribe 's' para sí o 'n' para no.")

# Número de élites (si aplica)
num_elites = 0
if usar_elitismo:
    while True:
        try:
            num_elites = int(input("Cantidad de élites (entero >= 1): "))
            if num_elites >= 1:
                break
            print("❌ Debe ser al menos 1.")
        except ValueError:
            print("❌ Ingresa un número entero válido.")

# Probabilidad de cruce
while True:
    try:
        Pcruce = float(input("Probabilidad de cruce (0.0 a 1.0): "))
        if 0.0 <= Pcruce <= 1.0:
            break
        print("❌ Debe estar entre 0.0 y 1.0.")
    except ValueError:
        print("❌ Ingresa un número válido.")

# Probabilidad de mutación
while True:
    try:
        Pmuta = float(input("Probabilidad de mutación (0.0 a 1.0): "))
        if 0.0 <= Pmuta <= 1.0:
            break
        print("❌ Debe estar entre 0.0 y 1.0.")
    except ValueError:
        print("❌ Ingresa un número válido.")

# ==== DATOS DEL PROBLEMA ====
x, pesos, utilidad, capacidad_max, diccionario_cromosomas, n = solicitar_datos_problema(tipo_codificacion)

# Si es binario, definir tamaño de población aquí
if tipo_codificacion == "binario":
    n = definir_tamano_poblacion_binario(x)


if tipo_codificacion == "binario":
    algoritmo_binario(n, x, pesos, utilidad, capacidad_max, tipo_seleccion, usar_elitismo, num_elites, tam_torneo, Pcruce, Pmuta, tipo_cruce, tipo_mutacion, maximizar_minimizar)
elif tipo_codificacion == "entero":
    algoritmo_entero(n, x, pesos, utilidad, capacidad_max, tipo_seleccion, usar_elitismo, num_elites, tam_torneo, Pcruce, Pmuta, diccionario_cromosomas, tipo_cruce, tipo_mutacion, maximizar_minimizar)
elif tipo_codificacion == "decimal":
    algoritmo_decimal(Pcruce, Pmuta, tipo_cruce, tipo_mutacion, usar_elitismo, num_elites, tipo_seleccion, tam_torneo, maximizar_minimizar)
else:
    print("Codificación no soportada.")

plot_fitness()