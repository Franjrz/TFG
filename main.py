































import sys
sys.path.insert(0, '..') # Add qiskit_runtime directory to the path

#api token 4e01dc0c56e6f9941836f52ea4255e797119debe6afab3b25bcb0388e285f3977c743bd3d1c25ebcc12991e8332daf8436a69d8bbb4b7212713a367a5eaea23d
from qiskit import IBMQ
IBMQ.load_account()
provider = IBMQ.get_provider(project='qiskit-runtime') # Change this to your provider.
backend = provider.get_backend('ibmq_montreal')

import numpy as np
import random import randint
from qiskit import *

import pandas as pd

df = pd.read_csv('../qiskit_runtime/qka/aux_file/dataset_graph7.csv',sep=',', header=None) # alterative problem: dataset_graph10.csv
data = df.values

print(df.head(4))

import numpy as np

# choose number of training and test samples per class:
num_train = 10
num_test = 10

# extract training and test sets and sort them by class label
train = data[:2*num_train, :]
test = data[2*num_train:2*(num_train+num_test), :]

ind=np.argsort(train[:,-1])
x_train = train[ind][:,:-1]
y_train = train[ind][:,-1]

ind=np.argsort(test[:,-1])
x_test = test[ind][:,:-1]
y_test = test[ind][:,-1]

from qiskit_runtime.qka import FeatureMap

d = np.shape(data)[1]-1                                         # feature dimension is twice the qubit number

em = [[0,2],[3,4],[2,5],[1,4],[2,3],[4,6]]                      # we'll match this to the 7-qubit graph
# em = [[0,1],[2,3],[4,5],[6,7],[8,9],[1,2],[3,4],[5,6],[7,8]]  # we'll match this to the 10-qubit graph

fm = FeatureMap(feature_dimension=d, entangler_map=em)          # define the feature map
initial_point = [0.1]                                           # set the initial parameter for the feature map

from qiskit.tools.visualization import circuit_drawer
circuit_drawer(fm.construct_circuit(x=x_train[0], parameters=initial_point),
               output='text', fold=200)

C = 1                                                           # SVM soft-margin penalty
maxiters = 10                                                   # number of SPSA iterations

initial_layout = [10, 11, 12, 13, 14, 15, 16]                   # see figure above for the 7-qubit graph
# initial_layout = [9, 8, 11, 14, 16, 19, 22, 25, 24, 23]       # see figure above for the 10-qubit graph

print(provider.runtime.program('quantum-kernel-alignment'))

def interim_result_callback(job_id, interim_result):
    print(f"interim result: {interim_result}\n")

program_inputs = {
    'feature_map': fm,
    'data': x_train,
    'labels': y_train,
    'initial_kernel_parameters': initial_point,
    'maxiters': maxiters,
    'C': C,
    'initial_layout': initial_layout
}

options = {'backend_name': backend.name()}

job = provider.runtime.run(program_id="quantum-kernel-alignment",
                              options=options,
                              inputs=program_inputs,
                              callback=interim_result_callback,
                              )

print(job.job_id())
result = job.result()

print(f"aligned_kernel_parameters: {result['aligned_kernel_parameters']}")

from matplotlib import pyplot as plt
from pylab import cm
plt.rcParams['font.size'] = 20
plt.imshow(result['aligned_kernel_matrix']-np.identity(2*num_train), cmap=cm.get_cmap('bwr', 20))
plt.show()

from qiskit_runtime.qka import KernelMatrix
from sklearn.svm import SVC
from sklearn import metrics

# train the SVM with the aligned kernel matrix:

kernel_aligned = result['aligned_kernel_matrix']
model = SVC(C=C, kernel='precomputed')
model.fit(X=kernel_aligned, y=y_train)

# test the SVM on new data:

km = KernelMatrix(feature_map=fm, backend=backend, initial_layout=initial_layout)
kernel_test = km.construct_kernel_matrix(x1_vec=x_test, x2_vec=x_train, parameters=result['aligned_kernel_parameters'])
labels_test = model.predict(X=kernel_test)
accuracy_test = metrics.balanced_accuracy_score(y_true=y_test, y_pred=labels_test)
print(f"accuracy test: {accuracy_test}")




class AlgoritmoGenetico:
  matriz_objetivo = None
  error_max = None
  poblacion = None
  nueva_poblacion = None
  elite = None
  porcentaje_elite = None
  tamaño_poblacion = None
  porcentaje_mutacion = None
  seed = None
  n_qbits = None

  def __init__(self, matriz_objetivo, error_max, porcentaje_elite, tamaño_poblacion, porcentaje_mutacion, seed, tamaño_individuo, n_qbits):
    self.matriz_objetivo = matriz_objetivo
    self.error_max = error_max
    self.porcentaje_elite = porcentaje_elite
    self.tamaño_poblacion = tamaño_poblacion
    self.porcentaje_mutacion = porcentaje_mutacion
    self.seed = seed
    self.tamaño_individuo = tamaño_individuo
    self.n_qbits = n_qbits

  def evolucion(self): # Ejecuta la evolucion de los individuos y devuelve el más optimo con un error por debajo del umbral de error_max
    random.seed(self.seed)
    self.generacion_poblacion()
    if not self.calculo_fit():
      while True:
        self.cribado_poblacion()
        self.cruce()
        self.mutacion()
        self.poblacion = self.elite + self.nueva_poblacion
        if self.calculo_fit():
            break

    fitness = min(self.poblacion[1])
    ganador = self.poblacion[0][self.poblacion[1].index(fitness)]
    return ganador, fitness


  def generacion_poblacion(self): #Devuelve una lista de lista [objetos CircuitoCuantico aleatorio] y lista [fitness(num random)]
    poblacion = []
    fitness = []
    for _ in range(self.tamaño_poblacion):
      poblacion.append(Individuo(self.tamaño_individuo, None, False, self.n_qbits))
      fitness.append(100000) #el fitness es tan alto para que no lo tome como ganador todavía
    self.poblacion.append(poblacion)
    self.poblacion.append(fitness)

  def generar_circuito(self, individuo):
      circuito = QuantumCircuit(self.n_qbits)

     """
     tipo = None #1: Rx, 2: Ry, 3: Rz, 4: H, 5: Unidad, 6: Cnot
     qbits = None #lista de 1 o 2 num que representan pos de qbits
     """
      for columna in individuo.adn:
          for puerta in columna:


      return circuito

  def calculo_fit(self): #Devuelve True si el error del mejor individuo supera a error_max y False en caso contrario. Evalua el error de cada individuo y lo almacena en fitness
    pass

  def cribado_poblacion(self): #Reduce la poblacion a los porcentaje_elite por ciento del original mediante torneo
    ganadores = [[],[]]
    max_ganadores = ceil(self.porcentaje_elite * len(self.poblacion[0]))
    contador = 0
    max_fitness = max(self.poblacion[1])
    min_fitness = min(self.poblacion[1])
    cogido = [False] * len(self.poblacion[1])
    while max_ganadores > contador:
      if contador = len(self.poblacion[1]):
        contador = 0

      if random.uniform(min_fitness, max_fitness) > self.poblacion[1][contador] and cogido[contador] == False:
        cogido[contador] = True
        ganadores[0].append(self.poblacion[0][contador])
        ganadores[1].append(self.poblacion[1][contador])

      contador += 1

    self.elite = ganadores

  def cruce(self): #De forma aleatoria cruza individuos hasta volver a completar la poblacion original
    nueva_poblacion = [[],[]]
    contador = 0
    while len(self.poblacion[1])-len(self.elite[1]) > 0:
      if contador == len(self.elite[1]):
        contador = 0

      corte = random.randint(1, self.tamaño_individuo-1)
      if corte < len(self.elite[0][contador]): #Se divide en 2
        if corte < len(self.elite[0][contador]) - 1:
          nueva_poblacion[0].append(self.elite[0][contador][corte:])
          nueva_poblacion[1].append(100000)
          nueva_poblacion[0].append(self.elite[0][contador][:corte])
          nueva_poblacion[1].append(100000)
      else: #se fusionan 2



      contador += 1



  def mutacion(self): #De forma aleatoria muta la estructura de cada individuo
    for i in range(len(self.nueva_poblacion[1])):
      for j in range(len(self.nueva_poblacion[0][i])):
        if random.uniform(0, 1) < self.porcentaje_mutacion:
          qbitList = range(self.n_qbits)
          columna = []
          while len(qbitList) > 0:
            puerta = random.randint(1, 7)
            if puerta == 6 and len(qbitList) > 1:
              control = qbitList.pop(random.randrange(len(qbitList)))
              target = qbitList.pop(random.randrange(len(qbitList)))
              columna.append(Puerta(puerta,[control,target]))
            elif puerta < 6:
              qbit = qbitList.pop(random.randrange(len(qbitList)))
              columna.append(Puerta(puerta,[qbit]))
          self.nueva_poblacion[0][i][j] = columna
