import NN

RedeNeural = NN.NeuralNetwork(2,4,1)

for loss in range(1000000):
    entradas = [[[1],[1]],[[1],[0]],[[0],[1]],[[0],[0]]]
    saidas = [[[0]],[[1]],[[1]],[[0]]]

    index = loss%4

    if loss%100 == 0:
        RedeNeural.predict([[1],[1]])

    
    RedeNeural.treino(entradas[index],saidas[index])

RedeNeural.predict([[1],[1]])
RedeNeural.predict([[0],[0]])
RedeNeural.predict([[1],[0]])
RedeNeural.predict([[0],[1]])

print(RedeNeural.PesoIH)
print(RedeNeural.PesoHO) #0.0003359