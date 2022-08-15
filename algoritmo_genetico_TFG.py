###############################################################################
#librerias
###############################################################################
import math
import random
import library_qiskit

import numpy as np
import matplotlib.pyplot as plt


from qiskit import IBMQ
from qiskit import *
from qiskit.circuit import Parameter
from qiskit.circuit import ParameterVector
from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

import algoritmo_genetico

###############################################################################
#configuracion
###############################################################################

TOKEN = None

#IBMQ.delete_account()
IBMQ.save_account(TOKEN)
provider = IBMQ.load_account()
#backend = provider.get_backend('ibmq_qasm_simulator')
backend = provider.get_backend('ibmq_qasm_simulator')
seed = 12345
algorithm_globals.random_seed = seed

identificador = None
iteraciones = None
limite_fitness = None
parada = 3
condicion_parada = {0:"iteraciones",1:"fitness",2:"iteraciones_and_fitness",3:"iteraciones_or_fitness"}
semilla = None
size = None
elitismo = None
ratio_mutacion = None 
ratio_cruce = None
inmortalidad = False
seleccion = 2
tipo_seleccion = {0:"Ruleta",1:"Rank",2:"Torneo"}
objetivo = False #True: maximizar, False: minimizar
tipo_objetivo = {False:"Minimizar",True:"Maximizar"}
verbose = False
path_resultados = "/home/kubuntulegion/github/TFG/resultados_algoritmo_genetico/"
ciclos_estancamiento = 3
diferencia_estancamiento = 0.01
datos_auxiliares = None

tamaño_maximo = None
n_qbits = None

###############################################################################
#Datasets
###############################################################################

#Adhoc
def getAdhoc(dimension = 2):
    train_features, train_labels, test_features, test_labels, adhoc_total = ad_hoc_data(
        training_size=20,
        test_size=5,
        n=dimension,
        gap=0.3,
        plot_data=False,
        one_hot=False,
        include_sample_total=True,
    )
    plt.figure(figsize=(5, 5))
    plt.ylim(0, 2 * np.pi)
    plt.xlim(0, 2 * np.pi)
    plt.imshow(
        np.asmatrix(adhoc_total).T,
        interpolation="nearest",
        origin="lower",
        cmap="RdBu",
        extent=[0, 2 * np.pi, 0, 2 * np.pi],
    )

    plt.scatter(
        train_features[np.where(train_labels[:] == 0), 0],
        train_features[np.where(train_labels[:] == 0), 1],
        marker="s",
        facecolors="w",
        edgecolors="b",
        label="A train",
    )
    plt.scatter(
        train_features[np.where(train_labels[:] == 1), 0],
        train_features[np.where(train_labels[:] == 1), 1],
        marker="o",
        facecolors="w",
        edgecolors="r",
        label="B train",
    )
    plt.scatter(
        test_features[np.where(test_labels[:] == 0), 0],
        test_features[np.where(test_labels[:] == 0), 1],
        marker="s",
        facecolors="b",
        edgecolors="w",
        label="A test",
    )
    plt.scatter(
        test_features[np.where(test_labels[:] == 1), 0],
        test_features[np.where(test_labels[:] == 1), 1],
        marker="o",
        facecolors="r",
        edgecolors="w",
        label="B test",
    )

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.title("Ad hoc dataset for classification")

    #plt.show()
    return train_features, train_labels, test_features, test_labels, adhoc_total, plt

#Tarjetas
#Salud
#Kernel covariante



###############################################################################
#Kernel coding
###############################################################################
#ACABAR
def getFeatureMap():
  pass

#Circuito cuantico base del kernel
adhoc_feature_map = ZZFeatureMap(feature_dimension=adhoc_dimension, reps=2, entanglement="linear") #SUSTITUIR ESTA PARTE
#Ordenador/Simulador cuantico
adhoc_backend = QuantumInstance(BasicAer.get_backend("qasm_simulator"), shots=1024, seed_simulator=seed, seed_transpiler=seed)
#Kernel cuantico
#adhoc_kernel = QuantumKernel(feature_map=adhoc_feature_map, quantum_instance=adhoc_backend)
adhoc_kernel = QuantumKernel(feature_map=getFeatureMap(), quantum_instance=adhoc_backend)

def getKernelClasico(datos, lenMatriz):
  K = np.zeros((lenMatriz,lenMatriz)) # lenMatriz by lenMatriz matrix
  for i in range(lenMatriz):
      for j in range(lenMatriz):
          K[i,j] = (1 + np.dot(datos[i,:],datos[j,:]))**2


###############################################################################
#QSVM coding
###############################################################################

#para otros datos ejecuatar results = qsvc.run(quantum_instance)
def getQSVC(kernel, train_features, train_labels, test_features, test_labels):
  #Se genera la QSVC dado su kernel
  qsvc = QSVC(quantum_kernel=kernel)
  #Se entrena
  qsvc.fit(train_features, train_labels)
  #Se mide su eficiencia en datos nuevos
  qsvc_score = qsvc.score(test_features, test_labels)
  return qsvc, qsvc_score


###############################################################################
#Quantum gates functions
###############################################################################

def RX(args):
    #args [circuito, qbit[a,b], valor]
    args[0].rx(args[2],args[1][0])

def RY(args):
    #args [circuito, qbit[a,b], valor]
    args[0].ry(args[2],args[1][0])

def RZ(args):
    #args [circuito, qbit[a,b], valor]
    args[0].rz(args[2],args[1][0])

def H(args):
    #args [circuito, qbit[a,b], valor]
    args[0].h(args[1][0])

def I(args):
    #args [circuito, qbit[a,b], valor]
    args[0].id(args[1][0])

def CX(args):
    #args [circuito, qbit[a,b], valor]
    args[0].cx(args[1][0],args[1][0])

funcionesPuertas = [RX,RY,RZ,H,I,CX]

class Puerta:
  tipo = None #1: Rx, 2: Ry, 3: Rz, 4: H, 5: Unidad, 6: Cnot
  qbits = None #lista de 1 o 2 num que representan pos de qbits
  valor = None #rotacion en las R. Puede ser un valor directo de np.pi*n o un parametro

  def __init__(self,tipo, qbits, valor):
    self.tipo = tipo
    self.qbits = qbits
    self.valor = valor


###############################################################################
#Convierte la codificación que simboliza el circuito cuantico al circuito en sí
###############################################################################

def genoma2Circuito(genoma, n_qbits):
    circuito = QuantumCircuit(n_qbits)
    for i in range(len(genoma)):
        for j in range(len(genoma[i])):
            args = [circuito, genoma[i][j].qbits, genoma[i][j].valor]
            funcionesPuertas[genoma[i][j].tipo](args)
    return circuito


###############################################################################
#QSVM coding
###############################################################################

def funcion_str_genoma(self_):
    return str(genoma2Circuito(self_.genoma, self_.datos_auxiliares[1]))


###############################################################################
#QSVM coding
###############################################################################

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
                valor = None
                if puerta <= 3:
                    valor = 2*np.pi*random.random()
                columna.append(Puerta(puerta,[qbit], valor))
        genoma.append(columna)
    return genoma


###############################################################################
#Llama a la qsvm y evalua el resultado
###############################################################################
#ACABAR
def funcion_fitness(self_):
    return getQSVC(kernel, train_features, train_labels, test_features, test_labels)


###############################################################################
#Muta una columna del genoma de forma aleatoria
###############################################################################

def funcion_mutar(self_):  
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
                    valor = None
                    if puerta <= 3:
                        valor = 2*np.pi*random.random()
                    columna.append(Puerta(puerta,[qbit], valor))
            self_.genoma[i] = columna


###############################################################################
#Cruza dos dos individuos intercambiando de manera aleatoria sus columnas
###############################################################################

def funcion_cruzar_genomas(ratio_cruce, individuos):
    #Intercambiar las columnas de forma aleatoria entre dos individuos
    #Seleccionar el grande y el pequenio
    if len(individuos[0]) >= len(individuos[1]):
        grande = individuos[0]
        pequenio = individuos[1]
    else:
        grande = individuos[1]
        pequenio = individuos[0]

    for i in range(len(pequenio)):
        if random.random() < ratio_cruce:
            columna = grande[i]
            grande[i] = pequenio[i]
            pequenio[i] = columna

    return [pequenio, grande]


###############################################################################
#GA running
###############################################################################
#ACABAR
print(identificador)
AG = algoritmo_genetico.AlgoritmoGenetico(identificador, iteraciones, limite_fitness, condicion_parada[parada], semilla, 
        size, elitismo, ratio_mutacion, ratio_cruce, inmortalidad, tipo_seleccion[seleccion], tipo_objetivo[objetivo], datos_auxiliares, 
        funcion_str_genoma, funcion_generar_individuo_aleatorio, funcion_fitness, funcion_mutar, funcion_cruzar_genomas, 
        verbose, path_resultados, ciclos_estancamiento, diferencia_estancamiento)
AG.ejecutar()
print(AG.historial_mejor_fitness)
print()
print(AG.historial_mejor_individuo)
print()
print(AG.ganador)
print()
AG.plot_historico_fitness()
