import numpy as np
import random import randint

class Puerta:
  tipo = None #1: Rx, 2: Ry, 3: Rz, 4: H, 5: Unidad, 6: Cnot
  qbits = None #lista de 1 o 2 num que representan pos de qbits

  def __init__(self,tipo, qbits):
    self.tipo = tipo
    self.qbits = qbits

class Individuo:
  adn = None #lista de listas(columnas) que contienen las puertas codificadas

  def __init__(self, tamaño_maximo, adn, adn_o_random, qbits):
    if adn_o_random: #si solo se quiere generar un individuo dado su adn basta con gardar su adn
      self.adn = adn
    else: #si no se genera de forma aleatoria su tamaño [1,tamaño_maximo] y se generan de forma aleatoria las puertas de las que se compone
      self.adn = []
      tamaño = random.randint(1, tamaño_maximo+1) #tamaño del individuo
      for _ in rango(tamaño):
        qbitList = range(tamaño)
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
        self.adn.append(columna)

  def adn2Circuito(self):
    pass

  def getmatriz(self):
    pass


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
      self.cribado_poblacion()
      self.cruce()
      self.mutacion()
      self.poblacion = self.elite + self.nueva_poblacion

      while(self.calculo_fit()):
        self.cribado_poblacion()
        self.cruce()
        self.mutacion()
        self.poblacion = self.elite + self.nueva_poblacion

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
    parada = False
    while ¡Parada: len(self.poblacion[1])-len(self.elite[1]) > contador:
      if contador = len(self.elite[1]):
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
