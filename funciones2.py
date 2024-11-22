import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd

def hallar_max(vector):
    """
    Función: Retorna el índice del valor máximo del vector

    Parametros de entrada
    ----------
    vector : np.array

    Retorna
    -------
    integer
    
    """
    maximo = (0, 0)
    
    for i in range(vector.shape[0]):
        if maximo[1] <= abs(vector[i]):
            maximo = (i, abs(vector[i]))
    
    return maximo[0]

def calcularLU(A):
    """
    Función: Retorna la factorización LU de A con pivoteo parcial, 
    y la P que revierte sus permutaciones, tal que A = PLU.
    
    Parametros de entrada
    ----------
    A : np.array (matriz cuadrada).

    Retorna
    -------
    L, U, P : tuple (np.array, np.array, np.array). 

    """
    A = A.astype('float64')
    n = A.shape[0]
    L, U = np.identity(n), np.identity(n)
    
    #LU es una matriz que va conteniendo a L debajo de la diagonal
    #y a U en la y sobre la diagonal a medida que se van construyendo
    LU = A.copy()
    orden = np.array(range(n))
    
    for k in range(n-1):
        #hallamos el elemento más grande
        pos_de_permutacion = k + hallar_max(LU[k:,k])
        
        #permutamos la fila - y lo guardamos
        original = LU[k,:].copy()
        LU[k,:], LU[pos_de_permutacion, :] = LU[pos_de_permutacion, :], original
        orden[k], orden[pos_de_permutacion] = orden[pos_de_permutacion], orden[k]
        
        #hallamos Tau, nuestro vector de eliminación gaussiana
        Tau = np.zeros(n)
        for i in range(k, n):
            Tau[i] = LU[i, k]/LU[k,k]
        
        #armamos el paso k de U
        for i in range(k+1, n):
            LU[i, k:] = LU[i,k:] - Tau[i] * LU[k, k:]
        
        #agregamos Tau, armando L
        LU[k+1:, k] = Tau.reshape((n, 1))[k+1:, 0]
        
    
    #dividimos LU en L y U
    for k in range(n):
        L[k+1:, k] = LU[k+1:, k]
        U[k, k:] = LU[k, k:]
    
    #creamos P (inversa de matriz de permutación) utlizando nuestro vector "orden"
    P = np.zeros((n,n))
    for k in range(n):
        P[orden[k], k] = 1    
   
    #Luego, A = P x L x U
    
    return L, U, P

def inversaLU(L, U, P):
    """
    Función: Retorna la inversa de A (la cual resulta de multiplicar P x L x U)
    
    Parametros de entrada
    ----------
    L : np.array (lower triangular)
    U : np.array (upper triangular)
    P : np.array (inversa de matriz de permutación)

    Retorna
    -------
    Inversa: np.array (matriz cuadrada)
    
    """
    # para hallar la inversa de A necesitamos hallar la Inversa de L, U y P
    # luego, la inversa de A será igual a Inversa de U x Inversa de L x Inversa de P
    # pensamos a la ecuación U x Inv_U = I como:
    # U x (primera columna de Inv_U (siendo en principio una incognita)) = 
    # (Primer canónico),
    # U x (segunda columna de Inv_U) = (Segundo canónico)
    # y así sucesivamente (nos van a quedar tantas ecuaciones como cantidad de 
    # columnas de U).
    # De esta forma hallamos las columnas de la inversa de U individualmente
    # y luego las combinamos en una única matriz, formando finalmente la matriz
    # inversa de U.
    # Repitiendo la misma idea, hallamos la inversa de L. 
    
    n = L.shape[0]
    Inv_U = np.identity(n)
    Inv_L = np.identity(n)
    
    for k in range(n):
        e_k = np.zeros(n)
        e_k[k] = 1
        Inv_U[:, k] = np.linalg.solve(U, e_k)
        Inv_L[:, k] = np.linalg.solve(L, e_k)


    #P, al ser una matriz de permutación, es fácil de invertir.
    Inv_P = np.transpose(P)

    Inversa = Inv_U @ Inv_L @ Inv_P
    
    return Inversa


def inv(A):
    """
    Función: Retorna la inversa del A, calculada mediante la factorización LU
    
    Parametros de entrada
    ----------
    A : np.array (matriz cuadrada)

    Retorna
    -------
    np.array (matriz cuadrada)

    """
    return inversaLU(*calcularLU(A))




#Acá empiezan las funciones del TP2

def metodo_potencia(A, numero_de_iteraciones=250):
    """
    Función: Retorna el mayor autovalor y el autovector asociado de la matriz A
    
    Parametros de entrada
    ----------
    A : np.array (matriz cuadrada)
    numero_de_iteraciones : potencia a la que elevo A antes de multiplicarla por el vector inicial

    Retorna
    -------
    El autovalor y su autovector asociado

    """
    n = A.shape[0]
    
    #tomo un vector normal random
    v = np.random.rand(n)   
    v = v / np.linalg.norm(v)  
    
    for i in range(numero_de_iteraciones):
        #lo multiplico y normalizo
        v_siguiente = np.dot(A, v)
        v_siguiente = v_siguiente / np.linalg.norm(v_siguiente)
        v = v_siguiente

    #obtenemos el autovalor
    autovalor = np.dot(v_siguiente, np.dot(A, v_siguiente))     
    return autovalor, v_siguiente

def matriz_ip_del_pais_1():
    """
    Función: Obtener la matriz insumo-producto del país 1

    Retorna
    -------
    La matriz insumo-producto del país 1

    """

    #importamos el archivo con los datos
    df = pd.read_excel("./matrizlatina2011_compressed_0.xlsx", sheet_name='LAC_IOT_2011')

    #obtenemos la matriz parcial con los datos relevantes, filtrando con pandas
    COL_COL = df[df['Country_iso3'] == 'COL'][df.columns[pd.Series(
        df.columns.values).str.startswith("COL")]]
    
    #obtenemos los outputs, la producción por país
    output_COL = np.array(df[df['Country_iso3'] == 'COL']['Output'])

    #ahora que tenemos los datos, queremos construir la matriz insumo-producto

    #convertimos los outputs a matrices
    P_COL = np.diag(output_COL)

    #invertimos las matrices producto
    Inv_P_COL = inv(P_COL)

    #creamos la matriz de coeficientes técnicos
    A_CC = np.array(COL_COL@Inv_P_COL)
    
    return A_CC

def matriz_ip_del_pais_2():
    """
    Función: Obtener la matriz insumo-producto del país 2

    Retorna
    -------
    La matriz insumo-producto del país 2

    """

    #importamos el archivo con los datos
    df = pd.read_excel("./matrizlatina2011_compressed_0.xlsx", sheet_name='LAC_IOT_2011')

    #obtenemos la matriz parcial con los datos relevantes, filtrando con pandas
    MEX_MEX = df[df['Country_iso3'] == 'MEX'][df.columns[pd.Series(
    df.columns.values).str.startswith("MEX")]]
    
    #obtenemos los outputs, la producción por país
    output_MEX = np.array(df[df['Country_iso3'] == 'MEX']['Output'])

    #ahora que tenemos los datos, queremos construir la matriz insumo-producto

    #convertimos los outputs a matrices
    P_MEX = np.diag(output_MEX)

    #invertimos las matrices producto
    Inv_P_MEX = inv(P_MEX)

    #creamos la matriz de coeficientes técnicos
    A_MM = np.array(MEX_MEX@Inv_P_MEX)
    
    return A_MM


def metodoPotenciaHotelling(A):
    """
    Función: Retorna el mayor autovector asociado de la matriz A
    
    Parametros de entrada
    ----------
    A : np.array (matriz cuadrada)
    
    Retorna
    -------
    El autovector asociado al mayor autovalor

    """
    n = A.shape[0]
    
    #tomo un vector normal random
    v = np.random.rand(n)   
    v = v / np.linalg.norm(v)  
    v_siguiente = np.dot(A, v)
    
    while(np.linalg.norm(v_siguiente - v) > 1 - 0.999999): #con un epsilon arbitrario
        #lo multiplico y normalizo
        v = v_siguiente
        v_siguiente = np.dot(A, v)
        v_siguiente = v_siguiente / np.linalg.norm(v_siguiente)    
            
    return v_siguiente


def metodoMonteCarlo(A, numero_de_iteraciones, iteraciones_metodo_potencia=50): #TODO
    """
    Función: Retorna la lista de resultados de ejecutar el método de la potencia numero_de_iteraciones veces
    
    Parametros de entrada
    ----------
    A : np.array (matriz cuadrada)
    numero_de_iteraciones : cantidad de veces que se ejecuta el método de Montecarlo
    iteraciones_metodo_potencia : iteraciones del método de la potencia
    
    Retorna
    -------
    Lista de autovalores calculados

    """
    n = A.shape[0]
    autovalores = []  
    for i in range(numero_de_iteraciones):
        v = np.random.rand(n)   
        v = v / np.linalg.norm(v)  
        for i in range(iteraciones_metodo_potencia):
            v_siguiente = np.dot(A, v)
            v_siguiente = v_siguiente / np.linalg.norm(v_siguiente)
        autovalor = np.dot(v_siguiente, np.dot(A, v_siguiente)) 
        autovalores.append(autovalor)
    return autovalores

def calcular_normas_potencia(A, potencia):
    """
    Función: Retorna la lista de normas de las sumas de potencias parciales
    
    Parametros de entrada
    ----------
    A : np.array (matriz cuadrada)
    potencia : cantidad de potencias a calcular
    
    Retorna
    -------
    Lista de normas de las sumas parciales de potencias

    """
    n = A.shape[0]
    I = np.eye(n)
    suma_parcial = I.copy()
    normas = [np.linalg.norm(suma_parcial, 2)]
    potencia_A = A.copy()
    for i in range(potencia):
        suma_parcial = suma_parcial + potencia_A
        normas.append(np.linalg.norm(suma_parcial, 2))
        potencia_A = A @ potencia_A
    return normas



def graficar_consigna_2 (A1,A2):  
    """
    Función: Grafica el efecto de la potencia sobre las normas de las matrices
  
    Parametros de entrada
    ----------
    A1
    A2
    """
  
    #calculamos los vectores a1 y a2, las normas de las potencias
    N = 250
    a1, a2 = np.zeros(N), np.zeros(N)

    potencia_A1 = A1.copy()
    potencia_A2 = A2.copy()

    for i in range(N):
        a1[i] = np.linalg.norm(potencia_A1, ord=2)
        a2[i] = np.linalg.norm(potencia_A2, ord=2)

        potencia_A1 = A1 @ potencia_A1
        potencia_A2 = A2 @ potencia_A2

    #graficamos los resultados
    plt.plot(a1, label="A1")
    plt.plot(a2, label="A2")
    plt.legend()
    plt.xlabel("Potencia a la que fue elevada")
    plt.ylabel("Norma")
    plt.title("Efecto de la potencia sobre la norma de la matriz")
    plt.show()
    
    
    
def graficar_consigna_3(autovalores_A1, promedio_A1, autovalores_A2, promedio_A2):
    """
    Función: Grafica los autovalores de las matrices calculados varias veces, siguiendo el método de Monte Carlo
    
    Parametros de entrada
    ----------
    autovalores_A1
    promedio_A1
    autovalores_A2
    promedio_A2

    """
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(autovalores_A1, marker='o', linestyle='-', color='b')
    plt.axhline(promedio_A1, color='r', linestyle='--', label='Promedio')
    plt.title('Autovalores de A1 (Método de Monte Carlo)')
    plt.xlabel('Iteraciones')
    plt.ylabel('Autovalor Aproximado')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid()
    
    
    plt.subplot(1, 2, 2)
    plt.plot(autovalores_A2, marker='o', linestyle='-', color='g')
    plt.axhline(promedio_A2, color='r', linestyle='--', label='Promedio')
    plt.title('Autovalores de A2 (Método de Monte Carlo)')
    plt.xlabel('Iteraciones')
    plt.ylabel('Autovalor Aproximado')
    plt.ylim(0, 1.1)
    
    plt.legend()
    plt.grid()
    
    
    plt.tight_layout()
    plt.show()
    
    
    
def graficar_consigna_4(normas_A1_n10, normas_A1_n100, normas_A2_n10, normas_A2_n100):
    """
    Función: Grafica las normas de las sumas parciales de A^n
    
    Parametros de entrada
    ----------
    normas_A1_n10 
    normas_A1_n100 
    normas_A2_n10 
    normas_A2_n100 

    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(101), normas_A1_n100, label="n=100")
    plt.plot(range(11), normas_A1_n10, label="n=10")
    plt.title("Convergencia de la serie parcial para A1")
    plt.xlabel("Número de términos (n)")
    plt.ylabel("Norma 2 de Suma Parcial")
    plt.ylim(-5,110)
    plt.legend()
    plt.scatter([10], [normas_A1_n10[-1]], marker="o", c="r", linewidths=4, zorder=3)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(101), normas_A2_n100, label="n=100")
    plt.plot(range(11), normas_A2_n10, label="n=10")
    plt.title("Convergencia de la serie parcial para A2")
    plt.xlabel("Número de términos (n)")
    plt.ylabel("Norma 2 de Suma Parcial")
    plt.ylim(-5,110)
    plt.legend()
    plt.scatter([10], [normas_A2_n10[-1]], marker="o", c="r", linewidths=4, zorder=3)

    
    plt.tight_layout()
    plt.show()

def error_de_suma_de_potencias(A2, n=100):
    """
    Función: Retorna el error en cada paso de la suma de potencias
    
    Parametros de entrada
    ----------
    A2 : np.array (matriz cuadrada)
    n : cantidad de sumas de potencias

    """    
    Leontief = inv(np.identity(5)- A2)
    error = []
    A = np.identity(5)
    suma_de_As = A
    error.append(np.linalg.norm(suma_de_As - Leontief, 2))
    for i in range(n):
        A = A @ A2
        suma_de_As += A
        error.append(np.linalg.norm(suma_de_As - Leontief, 2))

    return error


def graficar_consigna_4d(error_A2):
    """
    Función: Grafica la diferencia entre las sumas parciales y la matriz de Leontief
    
    Parametros de entrada
    ----------
    error_A2

    """    
    plt.plot(range(101), error_A2)
    plt.title("Diferencia entre la matriz de Leontief y la suma parcial para A2")
    plt.xlabel("Número de términos (n) de la suma parcial")
    plt.ylabel("Diferencia")
    
    plt.show()


def graficar_consigna_11_1(reduccionArr):
    """
    Función: grafica las distancias al origen de coordenadas de cada punto de la ACP en dos dimensiones e identificar el sector   
    más lejano y el más cercano al origen
    
    Parametros de entrada
    ----------
    reduccionArr
    
    Retorna
    -------
    sector_cercano
    sector_lejano
    """

    #tengo la matriz Arr reducida. calculo las distancias al origen
    cant_puntos = len(reduccionArr)
    distancias = np.zeros(cant_puntos)
    for i in range(cant_puntos):
        distancias[i] = np.linalg.norm(reduccionArr[i])

    plt.bar(range(1, cant_puntos+1), height=distancias, width=0.8)
    plt.title("Distancia de los distintos sectores al origen")
    plt.xlabel("Sectores")
    plt.ylabel("Distancias")

    #hallamos el sector más cercano al origen
    plt.bar([np.argmin(distancias)+1], height=min(distancias), width=0.8, color="red", label="Cercano")
    sector_cercano = np.argmin(distancias)

    #hallamos el sector más lejano al origen
    plt.bar([np.argmax(distancias)+1], height=max(distancias), width=0.8, color="orange", label="Lejano")
    sector_lejano = np.argmax(distancias)

    #lo mostramos
    plt.legend()
    plt.show()

    print(f'El sector con la menor distancia al origen es el sector {np.argmin(distancias)+1}'+
            f' con distancia {min(distancias)}')
    print(f'El sector con la mayor distancia al origen es el sector {np.argmax(distancias)+1}'+
            f' con distancia {max(distancias)}')
    
    return sector_cercano, sector_lejano
    
def graficar_consigna_11_2 (Arr, H, sector_cercano, sector_lejano): 
    """
    Función: Grafica la producción en Arr y H del sector mas lejano y mas cercano al origen
    
    Parametros de entrada
    ----------
    Arr
    H
    sector_cercano
    sector_lejano

    """
    cant_puntos = Arr.shape[0]

    #graficamos las filas de los sectores más cercanos y lejanos en la matriz original
    fig, ax1 = plt.subplots()
    ax1.plot(range(1, cant_puntos+1), Arr[sector_cercano], marker="o", c="g", label=f"Sector {sector_cercano+1}")
    ax1.plot(range(1, cant_puntos+1), Arr[sector_lejano], marker="o", label=f"Sector {sector_lejano+1}")
    ax1.set_title("Insumos consumidos por sector en la matriz Arr")
    ax1.set_xlabel("Sectores")
    ax1.set_ylabel("Consumo")
    ax1.legend()

    #lo mismo en la H
    fig, ax3 = plt.subplots()
    ax3.plot(range(1, cant_puntos+1), H[sector_cercano], marker="o", c="g", label=f"Sector {sector_cercano+1}")
    ax3.plot(range(1, cant_puntos+1), H[sector_lejano], marker="o", label=f"Sector {sector_lejano+1}")
    ax3.set_title("Insumos consumidos por sector en la matriz H")
    ax3.set_xlabel("Sectores")
    ax3.set_ylabel("Consumo")
    ax3.legend()
    
    plt.show()
















