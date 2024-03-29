###############################################################################
#librerias
###############################################################################
import random
import math
#import library_qiskit

import numpy as np
import matplotlib.pyplot as plt


from qiskit import IBMQ
from qiskit import *
from qiskit.circuit import Parameter
from qiskit import BasicAer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_machine_learning.datasets import ad_hoc_data

import algoritmo_genetico
import copy



###############################################################################
#Datasets
###############################################################################

IDdataset = 0
porcentajeTrain = [0.8, None, None]
dimension_ad_hoc = 3
extra = [[dimension_ad_hoc,25],None, None]


#Adhoc
def ad_hoc(porcentajeTrain, extra):
    train_features, train_labels, test_features, test_labels, adhoc_total = ad_hoc_data(
        training_size=20,
        test_size=5,
        n=extra[0],
        gap=0.3,
        plot_data=False,
        one_hot=False,
        include_sample_total=True,
    )
    return train_features, train_labels, test_features, test_labels

#ACABAR
#Tarjetas
def tarjetas(porcentajeTrain, extra):
    #return train_features, train_labels, test_features, test_labels
    pass

#ACABAR
#Cancer
def cancer(porcentajeTrain, extra):
    #return train_features, train_labels, test_features, test_labels
    pass

get_dataset = [ad_hoc, tarjetas, cancer]

train_features, train_labels, test_features, test_labels = get_dataset[IDdataset](porcentajeTrain[IDdataset], extra[IDdataset])

###############################################################################
#configuracion
###############################################################################


identificador = ["AD HOC", "Tarjetas", "Cáncer"]
iteraciones = [2, None, None]
limite_fitness = [0.8, None, None]
parada = [0, None, None]
condicion_parada = {0:"iteraciones",1:"fitness",2:"iteraciones_and_fitness",3:"iteraciones_or_fitness"}
semilla = [12345, None, None]
#Tamaño poblacion
size = [5, None, None]
elitismo = [0.2, None, None]
ratio_mutacion = [0.5, None, None]
ratio_cruce = [1, None, None]
inmortalidad = [True, True, True]
seleccion = [2, 2, 2]
tipo_seleccion = {0:"Ruleta",1:"Rank",2:"Torneo"}
objetivo = [True, True, True] #True: maximizar, False: minimizar
tipo_objetivo = {False:"Minimizar",True:"Maximizar"}
verbose = [True, True, True]
path_resultados = ["/content/drive/MyDrive/TFG", "/content/drive/MyDrive/TFG", "/content/drive/MyDrive/TFG"]
ciclos_estancamiento = [5, None, None]
diferencia_estancamiento = [0, None, None]
longitud_maxima = [10,None,None]
n_qubits = [3,None,None]
puertas = [0,1,2,3,4,5]
pcnot = [0.5, 0, 0]
presto = []
for i in pcnot:
    presto.append((1-i)/5)
ppuertas = [[],[],[]]
for i in range(len(n_qubits)):
    for j in range(len(puertas)):
        if j < 5:
            ppuertas[i].append(presto[i])
        else:
            ppuertas[i].append(pcnot[i])
#[longitud_maxima, n_qubits, puertas[0,1,2,3,4,5], probabilidad puertas]
datos_auxiliares = [[],[],[]]
for i in range(len(n_qubits)):
    datos_auxiliares[i] = [longitud_maxima[i], n_qubits[i], puertas, ppuertas[i]]
backend = Aer.get_backend('qasm_simulator')
shots = 1024
semilla_cuantica = 12345
datos_auxiliares_poblacion = [[backend, shots, semilla_cuantica, train_features, train_labels, test_features, test_labels]]



def autenticar(token, backend, seed):
    #IBMQ.delete_account()
    IBMQ.save_account(token)
    provider = IBMQ.load_account()
    backend = provider.get_backend(backend)
    algorithm_globals.random_seed = seed
    return provider, backend

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
    args[0].cx(args[1][0],args[1][1])

funcionesPuertas = [RX,RY,RZ,H,I,CX]
nombresPuertas = ["Rx","Ry","Rz","H","I","Cx"]

class Puerta:
  tipo = None #1: Rx, 2: Ry, 3: Rz, 4: H, 5: Unidad, 6: Cnot
  qbits = None #lista de 1 o 2 num que representan pos de qbits
  valor = None #rotacion en las R. Puede ser un valor directo de np.pi*n o un parametro

  def __init__(self,tipo, qbits, valor):
    self.tipo = tipo
    self.qbits = qbits
    self.valor = valor
  def __str__(self):
    resultado = "Tipo: " + nombresPuertas[self.tipo] + "   Qubits involucrados: "
    for i in range(len(self.qbits)):
      resultado += "q_" + str(self.qbits[i]) + " "
    resultado += " Valor: " + str(self.valor)
    return resultado

class Genoma:
  matriz = None #Matriz con todas las puertas
  columnaspp = None #Diccionario con clave columna y valor lista de qbits de las puertas con parametros en esa columna

  def __init__(self,matriz, columnaspp):
    self.matriz = matriz
    self.columnaspp = columnaspp


###############################################################################
#Convierte la codificación que simboliza el circuito cuantico al circuito en sí
###############################################################################

def genoma2Circuito(genoma, n_qbits):
    #Se crea un objeto circuito cuantico
    circuito = QuantumCircuit(n_qbits)
    #Se recorren las columnas de constante
    for i in range(len(genoma.matriz)):
        #Se recorre toda la columna
        for j in range(len(genoma.matriz[i])):
            #Se generan los parámetros para crear la puerta real
            args = [circuito, genoma.matriz[i][j].qbits, genoma.matriz[i][j].valor]
            #Se crea la puerta real
            funcionesPuertas[genoma.matriz[i][j].tipo](args)
    #Cuando esta todo el circuito generado se devuelve
    return circuito


###############################################################################
#Convierte el circuito a un string entendible por el ser humano
###############################################################################

def funcion_str_genoma(self_):
    return genoma2Circuito(self_.genoma, self_.datos_auxiliares[1])


###############################################################################
#Genera un individuo aleatorio en funcion de una semilla
###############################################################################

def generar_puerta_parametrica(tipo, qbit, letra):
    #alfabeto griego para los parametros
    alfabetoGriego = ["α","β","γ","δ","ε","ζ","η","θ","ι","κ","λ","μ","ν","ξ","ο","π","ρ","σ","τ","υ","φ","χ","ψ","ω"]      
    #Si no se ha completado una vuelta al alfabeto se escribe solo la letra
    if letra < len(alfabetoGriego):
      name = alfabetoGriego[letra]
      letra += 1
    else:
      #Si se ha completado una vuelta al alfabeto se escribe la letra y la vuelta
      name = alfabetoGriego[letra%len(alfabetoGriego)] + str(letra//len(alfabetoGriego))
      letra += 1
    #Se genera el parametro
    parametro = Parameter(name)
    #Se genera la puerta y se introduce en el diccionario
    puerta = Puerta(tipo,qbit,parametro)
    return puerta, letra
  
def funcion_generar_individuo_aleatorio(self_):
    #Tamaño del individuo
    tamanio = random.randint(1, self_.datos_auxiliares[0]) 
    #puntero para letra griega
    letra = 0
    #Se crea la matriz genoma de puertas
    genoma = []
    for i in range(tamanio):
      genoma.append([])      
    #Se almacenan los qubits y las columnas de las puertas parametricas en qbitspp clave columna valor lista de qbits
    qbitspp = {}
    #Se genera la posicion de columna de cada puerta parametrica
    #En caso del tamaño ser 1 solo se crean puertas parametricas
    if tamanio == 1:
      #Todas las puertas están en la columna 0
      qbitspp[0] = list(range(self_.datos_auxiliares[1]))
      #Se recorre cada qubit
      for i in range(self_.datos_auxiliares[1]):
          #Se decide que puerta R generar
          puerta = random.randint(0, 2)
          #Solo hay una columna
          columna = 0
          #Se genera la puerta y se actualiza letra
          puerta, letra = generar_puerta_parametrica(puerta, [i], letra)
          #Se pone en su lugar
          genoma[columna].append(puerta)
    else:
      #Se recorre cada qubit
      for i in range(self_.datos_auxiliares[1]):
          #Se decide que puerta R generar
          puerta = random.randint(0, 2)
          #Se decide en que columna colocarla
          columna = random.randint(0, tamanio-1)
          #Y se almacena en qbitspp
          if columna not in qbitspp.keys():
            qbitspp[columna] = [i]
          else:
            qbitspp[columna].append(i)
          #Se genera la puerta y se actualiza letra
          puerta, letra = generar_puerta_parametrica(puerta, [i], letra)
          #Se pone en su lugar
          genoma[columna].append(puerta)

      for i in range(tamanio):
          #Se crea la lista de qubits
          qbitList = list(range(self_.datos_auxiliares[1]))
          #Se eliminan los ocupados en esa columna por puertas parametricas
          if i in qbitspp.keys():
            qbitList = list(set(qbitList) - set(qbitspp[i]))
          #Y se baraja
          random.shuffle(qbitList)
          #Para recorrerla se crea el contador q
          q = 0
          #Se recorren todos los qubits
          while len(qbitList) > q:
              #Se decide que puerta generar
              puerta = np.random.choice(self_.datos_auxiliares[2], 1, p = self_.datos_auxiliares[3])
              puerta = puerta[0]
              if puerta == 5 and len(qbitList)-3 >= q:
                  #Se seleccionan los valores de control y target
                  control = qbitList[q]
                  target = qbitList[q+1]
                  #Se actualiza el contador
                  q += 2
                  #Y se añade a la columna
                  genoma[i].append(Puerta(puerta,[control,target],None))
              #Si es cualquier otra puerta
              elif puerta < 5:
                  #Se selecciona el valor de qbit
                  qbit = qbitList[q]
                  #Se actualiza el contador
                  q += 1
                  #El valor de la rotacion es nulo
                  valor = None
                  #Pero si la puerta es una rotacion
                  if puerta <= 3:
                      #Se genera un valor aleatorio para la rotacion
                      valor = 2*np.pi*random.random()
                  #Y se añade a la columna
                  genoma[i].append(Puerta(puerta,[qbit], valor))
    #Se genera el nuevo objeto genoma con toda la informacion
    genoma = Genoma(genoma, qbitspp)
    #Y se devuelve
    return genoma


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

class CustomFeatureMap():
    """Mapping data with a custom feature map."""
    
    def __init__(self, feature_dimension, circuit):
        self._support_parameterized_circuit = False
        self._feature_dimension = feature_dimension
        self.num_parameters = self.num_qubits = self._feature_dimension = feature_dimension
        self._circuit = circuit
        #Guardarlo despues
        self.parameters = circuit.parameters
    
    def assign_parameters(self, x):
        return self._circuit.assign_parameters(x)
        
def setParametrosQSVM(backend, circuito, shots, seed_simulator, seed_transpiler, train_features, train_labels, test_features, test_labels):
  circuito = CustomFeatureMap(circuito.datos_auxiliares[1], genoma2Circuito(circuito.genoma, circuito.datos_auxiliares[1]))
  quantumInstance = QuantumInstance(backend, shots=shots, seed_simulator=seed_simulator, seed_transpiler=seed_transpiler)
  kernel = QuantumKernel(feature_map=circuito, quantum_instance=quantumInstance)
  qsvc, qsvc_score = getQSVC(kernel, train_features, train_labels, test_features, test_labels)
  return qsvc_score

def getKernelClasico(datos, lenMatriz):
  K = np.zeros((lenMatriz,lenMatriz))
  for i in range(lenMatriz):
      for j in range(lenMatriz):
          K[i,j] = (1 + np.dot(datos[i,:],datos[j,:]))**2

###############################################################################
#Llama a la qsvm y evalua el resultado
###############################################################################

def funcion_fitness(self_, datos_auxiliares_poblacion):
    #datos_auxiliares_poblacion [backend, shots, seed, train_features, train_labels, test_features, test_labels]
    return setParametrosQSVM(datos_auxiliares_poblacion[0], self_, 
                             datos_auxiliares_poblacion[1], 
                             datos_auxiliares_poblacion[2], 
                             datos_auxiliares_poblacion[2], 
                             datos_auxiliares_poblacion[3], 
                             datos_auxiliares_poblacion[4], 
                             datos_auxiliares_poblacion[5], 
                             datos_auxiliares_poblacion[6])


###############################################################################
#Muta una columna del genoma de forma aleatoria
###############################################################################

def eliminar_qubits_ocupados(num_qubits, columnapp):    
  #Se crea la lista de qubits
  qbitList = list(set(range(num_qubits)) - set(columnapp))
  return qbitList

def funcion_mutar(self_):
  #Si el individuo tiene tamaño 1 entonces solo tiene pertas parametricas por lo que no puede mutar
  if len(self_.genoma.matriz) > 1:
    #Se recorre cada columna
    for i in range(len(self_.genoma.matriz)):
      #Si el numero obtenido es menor que el ratio de mutacion 
      if random.random() <= self_.ratio_mutacion:        
        #Se crea la nueva columna
        columnaNueva = []
        #Se crea la lista de qubits libres
        if i in self_.genoma.columnaspp.keys():
          qbitList = eliminar_qubits_ocupados(self_.datos_auxiliares[1], self_.genoma.columnaspp[i])
          #En caso de que no haya qubits libres se pasa a la siguiente columna
          if len(qbitList) == 0:
            continue
        else:
          qbitList = list(range(self_.datos_auxiliares[1]))
        #Si todos los qubits estan libres no hay puertas parametricas en la columna
        if len(qbitList) != self_.datos_auxiliares[1]:
          #Todos las puertas paramétricas se copian de la antigua a la nueva
          for j in self_.genoma.matriz[i]:
            if isinstance(j.valor,Parameter):
              columnaNueva.append(j)
        #Se sustituye una lista por otra
        self_.genoma.matriz[i] = columnaNueva
        #Se baraja la lista de qbits libres
        random.shuffle(qbitList)
        #Para recorrerla se crea el contador q
        q = 0
        #Se recorren todos los qubits
        while len(qbitList) > q:
            #Se decide que puerta generar
            puerta = np.random.choice(self_.datos_auxiliares[2], 1, p = self_.datos_auxiliares[3])
            puerta = puerta[0]
            if puerta == 5 and len(qbitList)-3 >= q:
                #Se seleccionan los valores de control y target
                control = qbitList[q]
                target = qbitList[q+1]
                #Se actualiza el contador
                q += 2
                #Y se añade a la columna
                self_.genoma.matriz[i].append(Puerta(puerta,[control,target],None))
            #Si es cualquier otra puerta
            elif puerta < 5:
                #Se selecciona el valor de qbit
                qbit = qbitList[q]
                #Se actualiza el contador
                q += 1
                #El valor de la rotacion es nulo
                valor = None
                #Pero si la puerta es una rotacion
                if puerta <= 3:
                    #Se genera un valor aleatorio para la rotacion
                    valor = 2*np.pi*random.random()
                #Y se añade a la columna
                self_.genoma.matriz[i].append(Puerta(puerta,[qbit], valor))
  return self_.genoma


###############################################################################
#Cruza dos dos individuos intercambiando de manera aleatoria sus columnas
###############################################################################

def sacar_puertas(col):
  #lista auxiliar para guardar las puertas a intercambiar 
  cola = []
  #Se comprueba puerta a puerta que no sean parametricas
  cc = 0
  while cc < len(col):
    #print(col[cc])
    if not isinstance(col[cc].valor,Parameter):
      #Se añade a la auxiliar
      cola.append(col[cc])
      #Se elimina de la original
      col.remove(col[cc])
      cc -= 1
    cc += 1
  #Se devuelve solo la auxiliar
  return cola

def meter_puertas(col, cola):
  #Se meten todas las puertas
  col.extend(cola)

def intercambiar_qbits(colap, colag):
  #Qubits en los que estan las puertas a intercambiar
  colaap = []
  colaag = []
  #Se extraen todos
  for i in range(len(colap)):
    colaap.extend(colap[i].qbits)
  for i in range(len(colag)):
    colaag.extend(colag[i].qbits)
  #Se barajan ambas listas
  random.shuffle(colaap)
  random.shuffle(colaag)
  q1 = 0
  q2 = 0
  #Se intercambian los qbits
  for i in range(len(colap)):
    if len(colap[i].qbits) == 1:
      colap[i].qbits = [colaag[q1]]
      q1 += 1
    else:
      colap[i].qbits = [colaag[q1], colaag[q1+1]]
      q1 += 2

  for i in range(len(colag)):
    if len(colag[i].qbits) == 1:
      colag[i].qbits = [colaap[q2]]
      q2 += 1
    else:
      colag[i].qbits = [colaap[q2], colaap[q2+1]]
      q2 += 2

  return colap, colag

def intercambiar_columnas(colp, colg):
  #puertas a intercambiar
  colpa = sacar_puertas(colp)
  colga = sacar_puertas(colg)
  #Se intercambian los qbits
  colap, colag = intercambiar_qbits(colpa, colga)
  #Se meten en la columna contraria
  meter_puertas(colp, colga)
  meter_puertas(colg, colpa)

def funcion_cruzar_genomas(ratio_cruce, individuos):    
  #Intercambiar las columnas de forma aleatoria entre dos individuos
  #Seleccionar el grande y el pequenio
  if len(individuos[0].matriz) >= len(individuos[1].matriz):
    grande = copy.deepcopy(individuos[0])
    pequenio = copy.deepcopy(individuos[1])
  else:
    grande = copy.deepcopy(individuos[1])
    pequenio = copy.deepcopy(individuos[0])

  for i in range(len(pequenio.matriz)):
    #Si el numero obtenido es menor al ratio de cruce
    if random.random() <= ratio_cruce:
      #Se obtienen los qbits ocupados de cada individuo
      if i in pequenio.columnaspp.keys():
        qbitsOcupadosP = pequenio.columnaspp[i]
      else:
        qbitsOcupadosP = []
      if i in grande.columnaspp.keys():
        qbitsOcupadosG = grande.columnaspp[i]
      else:
        qbitsOcupadosG = []
      #Si tienen la misma dimension se intercambian columnas:
      if len(qbitsOcupadosP) < len(pequenio.matriz[0]) and len(qbitsOcupadosP) == len(qbitsOcupadosG):
          intercambiar_columnas(pequenio.matriz[i], grande.matriz[i])
  return [pequenio, grande]


###############################################################################
#GA running
###############################################################################

AG = algoritmo_genetico.AlgoritmoGenetico(identificador[IDdataset], iteraciones[IDdataset], limite_fitness[IDdataset], 
        condicion_parada[parada[IDdataset]], semilla[IDdataset], size[IDdataset], elitismo[IDdataset], ratio_mutacion[IDdataset], 
        ratio_cruce[IDdataset], inmortalidad[IDdataset], tipo_seleccion[seleccion[IDdataset]], tipo_objetivo[objetivo[IDdataset]], 
        datos_auxiliares[IDdataset], datos_auxiliares_poblacion[IDdataset], funcion_str_genoma, funcion_generar_individuo_aleatorio, 
        funcion_fitness, funcion_mutar, funcion_cruzar_genomas, verbose[IDdataset], path_resultados[IDdataset], 
        ciclos_estancamiento[IDdataset], diferencia_estancamiento[IDdataset])
AG.ejecutar()
print(AG.historial_mejor_fitness)
print()
print(AG.historial_mejor_individuo)
print()
print(AG.ganador)
print()
AG.plot_historico_fitness()
#ACABAR
#Guardar en un archivo el mejor
