from numpy import array
import math

def sigmoid(lista):
    L = lista
    for x in range(len(lista)): 
        for y in range(len(lista[0])):
            L[x][y] = 1/(1 + math.exp(-lista[x][y]))

    return L

def dsigmoid(lista):
    L = lista
    for x in range(len(lista)): 
        for y in range(len(lista[x])):
            L[x][y] = lista[x][y] * (1 - lista[x][y])
    return L

def transpose(lista):
    x = len(lista)
    y = len(lista[0])
    
    arr = array(lista).reshape(y,x)
    L = array(lista).reshape(y,x)

    for col,lis in enumerate(arr):
        for row,cell in enumerate(lis):
            L[col][row] = lista[row][col]

    return L

def somaMatrixial(A, B):
    L = A
    for x in range(len(A)):
        for y in range(len(A[0])):
            L[x][y] = A[x][y]+B[x][y]
    return L

def subtracaoMatrixial(A, B):

    L = A
    for x in range(len(A)):
        for y in range(len(B)):
            L[x][y] = A[x][y]-B[x][y]
    return L

def multiplicacaoMatrixial(A, B):
    C = []
    Cx = len(A)
    Cy = len(B[0])

    for x in range(Cx):
        cel= []
        for y in range (Cy): 
            num = 0
            for j in range(len(B)):
                num += A[x][j] * B[j][y]
            cel.append(num)
        C.append(cel) 
    
    return C
            
