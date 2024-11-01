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
































