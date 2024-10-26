import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

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
