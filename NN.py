from random import randint
from numpy import multiply as hadamart
from numpy import multiply as multiply_Scalar
from matrix import *


class NeuralNetwork():
    def __init__(self,ISize,HSize,OSize):

        self.ISize = ISize
        self.HSize = HSize
        self.OSize = OSize

        #camadas

        self.Imput  = []
        self.Hidden  = []
        self.Output = []

        #bias

        self.biasIH = []
        self.biasHO = []

        #pesos

        self.PesoIH = []
        self.PesoHO = []

        self.DefinirPesos()

        self.learnig_rate = 0.3

    def DefinirPesos(self):
        #define os pesos iniciais de forma aleatoria

        for x in range (self.HSize):
            cel = []
            for y in range(self.ISize):
                cel.append(randint(-10,10)/10)
            self.PesoIH.append(cel)


        for x in range (self.OSize):
            cel = []
            for y in range(self.HSize):
                cel.append(randint(-10,10)/10)
            self.PesoHO.append(cel)
    
    def DefinirBias(self):
        #define os bias iniciais de forma aleatoria

        for x in range (0,2):
            cel = []
            for y in range(self.HSize):
                cel.append(randint(-10,10)/10)
            self.biasIH.append(cel)


        for x in range (0,2):
            cel = []
            for y in range(self.OSize):
                cel.append(randint(0,10))
            self.biasHO.append(cel)
    
    def treino(self,entrada,saida):

        #Feedforward

        self.Imput = entrada
        
        #Input -> hiden

        self.Hidden = multiplicacaoMatrixial(self.PesoIH,self.Imput)
        self.Hidden = sigmoid(self.Hidden)
        
        #hiden -> output

        self.Output = multiplicacaoMatrixial(self.PesoHO,self.Hidden)
        #self.Output = somaMatrixial(self.Output,self.biasHO)
        self.Output = sigmoid(self.Output)



        #Backpropagation 

        expected = saida
        
        
        Output_error = subtracaoMatrixial(expected,self.Output)
        d_Output = dsigmoid(self.Output)
        
        hidden_T = transpose(self.Hidden)

        gradient_O = hadamart(Output_error,d_Output)
        gradient_O = multiply_Scalar(gradient_O, self.learnig_rate)

        #self.biasHO = somaMatrixial(self.biasHO,gradient_O)

        pesos_HO_deltas = multiplicacaoMatrixial(gradient_O,hidden_T)
        self.PesoHO = somaMatrixial(self.PesoHO,pesos_HO_deltas)



        pesos_HO_T = transpose(self.PesoHO)
        hidden_error = multiplicacaoMatrixial(pesos_HO_T,Output_error)
        d_Hidden = dsigmoid(self.Hidden)

        input_T = transpose(self.Imput)

        gradient_H = hadamart(d_Hidden, hidden_error)
        gradient_H = multiply_Scalar(gradient_H, self.learnig_rate)

        #self.biasIH = somaMatrixial(self.biasIH,gradient_H)

        pesos_IH_deltas = multiplicacaoMatrixial(gradient_H,input_T)
        self.PesoIH = somaMatrixial(self.PesoIH,pesos_IH_deltas)

    
    def predict(self,entrada):

        #Feedfoward:

        self.Imput = entrada
        
        #Input -> hiden

        self.Hidden = multiplicacaoMatrixial(self.PesoIH,self.Imput)
       # self.Hidden = somaMatrixial(self.Hidden,self.biasIH)
        self.Hidden = sigmoid(self.Hidden)
        
        #hiden -> output

        self.Output = multiplicacaoMatrixial(self.PesoHO,self.Hidden)
        #self.Output = somaMatrixial(self.Output,self.biasHO)
        self.Output = sigmoid(self.Output)

        print(self.Output)


       





        
