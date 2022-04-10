import math
import random
import library_qiskit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

sys.path.insert(1, '/home/kubuntulegion/github/Librerias/Heuristicas')
import algoritmo_genetico


identificador =
iteraciones =
limite_fitness =
parada = 3
condicion_parada = {0:"iteraciones",1:"fitness",2:"iteraciones_and_fitness",3:"iteraciones_or_fitness"}
semilla =
size =
elitismo =
inmortalidad = False
seleccion = 2
tipo_seleccion = {0:"Ruleta",1:"Rank",2:"Torneo"}
objetivo = False #True: maximizar, False: minimizar
tipo_objetivo = {False:"Minimizar",True:"Maximizar"}
verbose = False
path_resultados = "/home/kubuntulegion/github/TFG/resultados_algoritmo_genetico/"
ciclos_estancamiento = 3
diferencia_estancamiento = 0.01

tamaño_maximo =
n_qbits =

datos_auxiliares = [tamaño_maximo, n_qbits]

def RX(circuito, qbit):
    circuito.x(qbit[0])

def RY(circuito, qbit):
    circuito.y(qbit[0])

def RZ(circuito, qbit):
    circuito.z(qbit[0])

def H(circuito, qbit):
    circuito.h(qbit[0])

def I(circuito, qbit):
    pass

def CX(circuito, qbit):
    circuito.cx(qbit[0],qbit[1])

funcionesPuertas = [RX,RY,RZ,H,I,CX]

class Puerta:
  tipo = None #1: Rx, 2: Ry, 3: Rz, 4: H, 5: Unidad, 6: Cnot
  qbits = None #lista de 1 o 2 num que representan pos de qbits

  def __init__(self,tipo, qbits):
    self.tipo = tipo
    self.qbits = qbits

def genoma2Circuito(genoma, n_qbits):
    circuito = QuantumCircuit(n_qbits)
    for i in range(len(genoma)):
        for j in range(len(genoma[i])):
            funcionesPuertas[genoma[i][j].tipo](circuito, genoma[i][j].qbits)
    return circuito

def getmatriz(self):
    pass

def funcion_str_genoma(self_):  #Lista
    #Se imprime el circuito
    return str(genoma2Circuito(self_.genoma, self_.datos_auxiliares[1]))

def funcion_generar_individuo_aleatorio(self_):  #Lista
    genoma = []
    tamaño = random.randint(1, self_.datos_auxiliares[0]+1) #tamaño del individuo
    for _ in range(tamaño):
        columna = []
        qbitList = range(self_.datos_auxiliares[1]) # lista de todos los qbits disponibles
        while len(qbitList) > 0:
            puerta = random.randint(1, 7)
            if puerta == 6 and len(qbitList) > 1:
                control = qbitList.pop(random.randrange(len(qbitList)))
                target = qbitList.pop(random.randrange(len(qbitList)))
                columna.append(Puerta(puerta,[control,target]))
            elif puerta < 6:
                qbit = qbitList.pop(random.randrange(len(qbitList)))
                columna.append(Puerta(puerta,[qbit]))
        genoma.append(columna)

    return genoma

def funcion_fitness(self_):
    return math.sin(self_.genoma)

def funcion_mutar(self_):  #Lista
    for i in range(len(self_.genoma)):
        if random.random() < self_.ratio_mutacion:
            columna = []
            qbitList = range(self_.datos_auxiliares[1]) # lista de todos los qbits disponibles
            while len(qbitList) > 0:
                puerta = random.randint(1, 7)
                if puerta == 6 and len(qbitList) > 1:
                    control = qbitList.pop(random.randrange(len(qbitList)))
                    target = qbitList.pop(random.randrange(len(qbitList)))
                    columna.append(Puerta(puerta,[control,target]))
                elif puerta < 6:
                    qbit = qbitList.pop(random.randrange(len(qbitList)))
                    columna.append(Puerta(puerta,[qbit]))
            self_.genoma[i] = columna

def funcion_cruzar_genomas(individuos):
    return [(individuos[0]+individuos[1])/2, (2*objetivo-1)*math.sqrt(abs(individuos[0]*individuos[1]))]


print(identificador)
AG = algoritmo_genetico.AlgoritmoGenetico(identificador, iteraciones, limite_fitness, condicion_parada[parada], semilla, size, elitismo, ratio_mutacion,
           inmortalidad, tipo_seleccion[seleccion], tipo_objetivo[objetivo], datos_auxiliares, funcion_str_genoma, funcion_generar_individuo_aleatorio,
           funcion_fitness, funcion_mutar, funcion_cruzar_genomas, verbose, path_resultados, ciclos_estancamiento, diferencia_estancamiento)
AG.ejecutar()
print(AG.historial_mejor_fitness)
print()
print(AG.historial_mejor_individuo)
print()
print(AG.ganador)
print()
AG.plot_historico_fitness()
