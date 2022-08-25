###############################################################################
#Uso de la libreria
###############################################################################
#import optimizacion

import numpy as np
import time
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import random
from pathlib import Path

class Individuo:
    """
    Esta clase sirve de interfaz que debe cumplir un individuo para ser valido

    Atributos
    ----------
    identificador : `int`
        numero que indentifica de forma univoca a cada individuo

    genoma : `personalizado`
        contiene toda la informacion que permite definir y testar al individuo
        para obtener su fitness

    fitness : `float`
        numero que simboliza la idoneidad del individuo ante un problema concreto

    semilla_generar : `int`
        numero entero que usa como semilla para generar numeros pseudoaleatorios
        que sirvan para generar un individuo nuevo

    semilla_mutar : `int`
        numero entero que usa como semilla para generar numeros pseudoaleatorios
        que sirvan para mutar a un individuo existente

    ratio_mutacion : `float`
        probabilidad de que un evento relacionado con la mutacion suceda

    funcion_str_genoma : `funcion`
        funcion personalizada para generar una representacion legible por un
        humano del genoma del individuo

    funcion_generar_individuo_aleatorio : `funcion`
        funcion personalizada para generar un individuo pseudoaleatorio usando
        una semilla

    funcion_fitness : `funcion`
        funcion objetivo fitness personalizada del problema

    funcion_mutar : `funcion`
        funcion personalizada para mutar de manera aleatoria al individuo

    """
    identificador = None
    genoma = None
    fitness = None
    semilla_generar = None
    semilla_mutar = None
    ratio_mutacion = None
    funcion_str_genoma = None
    funcion_generar_individuo_aleatorio = None
    funcion_fitness = None
    funcion_mutar = None

    def __init__(self, identificador, datos, datos_auxiliares, semilla_generar, semilla_mutar, ratio_mutacion, funcion_str_genoma,
                 funcion_generar_individuo_aleatorio, funcion_fitness, funcion_mutar):
        self.fitness = None
        self.identificador = identificador
        self.datos_auxiliares = datos_auxiliares
        self.semilla_mutar = semilla_mutar
        self.semilla_generar = semilla_generar
        self.ratio_mutacion = ratio_mutacion
        self.funcion_str_genoma = funcion_str_genoma
        self.funcion_generar_individuo_aleatorio = funcion_generar_individuo_aleatorio
        self.funcion_fitness = funcion_fitness
        self.funcion_mutar = funcion_mutar
        if datos == None:
            self.genoma = self._generar_individuo_aleatorio()
        else:
            self.genoma = datos

    def __str__(self):
        string = "    INDIVIDUO " + str(self.identificador)
        string += "\n" + str(self._str_genoma())
        string += "\nFitness " + str(self.fitness)

        return string

    def _str_genoma(self):
        """
        Metodo que genera un string que es legible por un humano y describe al
        genoma del individuo
        """
        return self.funcion_str_genoma(self)

    def _generar_individuo_aleatorio(self):
        """
        Metodo que genera un individuo de forma aleatoria mediante una semilla
        """
        random.seed(int(self.semilla_generar))
        return self.funcion_generar_individuo_aleatorio(self)

    def calcular_fitness(self, datos_auxiliares_poblacion):
        """
        Metodo que calcula el fitness del individuo
        """
        self.fitness = self.funcion_fitness(self, datos_auxiliares_poblacion)

    def _mutar(self):
        """
        Metodo que muta al individuo
        """
        random.seed(int(self.semilla_mutar))
        self.funcion_mutar(self)


class Poblacion:
    """
    Esta clase sirve de interfaz que debe cumplir una poblacion para ser valida

    Atributos
    ----------
    poblacion : `dict{identificador:Individuo}`
        diccionario donde se almacena la informacion de los individuos

    poblacion_fitness : `list[identificador,fitness]`
        lista con informacion redundante de los individuos para agilizar el
        calculo del fitness

    siguiente_id : `int`
        numero que indica el identificador del siguiente individuo que se cree

    size : `int`
        numero que indentifica el tamanio de la poblacion

    elitismo : `float`
        porcentaje mas cualificados de individuos que debe sobrevivir en cada
        iteracion para reproducirse

    ratio_cruce : `float`
        probabilidad de que un evento relacionado con el cruce suceda

    inmortalidad : `bool`
        flag booleana que indica si los individuos elite son inmortales y por
        lo tanto pasan de una generacion a otra mientras sigan siendo elite

    objetivo : `string`
        string que puede ser "Maximizar" o "Minimizar" que indica hacia donde
        optimiza el algoritmo

    tipo_objetivo : `dict{string:bool}`
        diccionario que sirve para simplificar calculos logicos relacionados
        con el objetivo

    seleccion : `string`
        string que puede ser "Ruleta", "Rank" o "Torneo" que indica el tipo de
        seleccion que se hace sobre la poblacion

    tipo_seleccion : `dict{string:int}`
        diccionario que sirve para simplificar calculos logicos relacionados
        con la seleccion

    funciones_seleccion : `tuple(function)`
        tupla de funciones que sirve para simplificar calculos logicos relacionados
        con el objetivo

    funcion_str_genoma : `funcion`
        funcion personalizada para generar una representacion legible por un
        humano del genoma del individuo

    funcion_generar_individuo_aleatorio : `funcion`
        funcion personalizada para generar un individuo pseudoaleatorio usando
        una semilla

    funcion_fitness : `funcion`
        funcion objetivo fitness personalizada del problema

    funcion_mutar : `funcion`
        funcion personalizada para mutar de manera aleatoria al individuo

    funcion_cruzar_genomas : `funcion`
        funcion personalizada para cruzar (generalmente dos) genomas de individuos
        produciendo (generalmente dos) genomas de individuos nuevos
    """
    poblacion={}
    poblacion_fitness = None
    siguiente_id=0
    size = None
    elitismo = None
    ratio_cruce = None
    ratio_mutacion = None
    inmortalidad = True
    objetivo = None
    tipo_objetivo = {"Maximizar":True,"Minimizar":False}
    seleccion = None
    tipo_seleccion = {"Ruleta":0,"Rank":1,"Torneo":2}
    funciones_seleccion = None
    funcion_str_genoma = None
    funcion_generar_individuo_aleatorio = None
    funcion_fitness = None
    funcion_mutar = None
    funcion_cruzar_genomas = None

    def __init__(self, semilla, size, elitismo, ratio_mutacion, ratio_cruce, inmortalidad, tipo_seleccion, objetivo, datos_auxiliares,
                 datos_auxiliares_poblacion, funcion_str_genoma, funcion_generar_individuo_aleatorio, funcion_fitness, funcion_mutar, funcion_cruzar_genomas):
        self.poblacion={}
        self.poblacion_fitness = None

        self.size = size
        self.siguiente_id = self.size
        self.elitismo = elitismo
        self.ratio_mutacion = ratio_mutacion
        self.ratio_cruce = ratio_cruce
        self.inmortalidad = inmortalidad
        self.objetivo = self.tipo_objetivo[objetivo]
        self.datos_auxiliares = datos_auxiliares
        self.datos_auxiliares_poblacion = datos_auxiliares_poblacion
        self.seleccion = self.tipo_seleccion[tipo_seleccion]
        self.funciones_seleccion = (self._seleccion_ruleta,self._seleccion_rank,self._seleccion_torneo)
        self.funcion_str_genoma = funcion_str_genoma
        self.funcion_generar_individuo_aleatorio = funcion_generar_individuo_aleatorio
        self.funcion_fitness = funcion_fitness
        self.funcion_mutar = funcion_mutar
        self.funcion_cruzar_genomas = funcion_cruzar_genomas
        np.random.seed(semilla)
        semillas_generar=np.random.randint(100*self.size, size = self.size)
        semillas_mutar=np.random.randint(100*self.size, size = self.size)
        self.poblacion_fitness=np.asarray([np.arange(self.size, dtype = np.longlong),np.zeros(self.size)])
        for i in range(self.size):
            self.poblacion[i]=Individuo(i, None, datos_auxiliares, semillas_generar[i], semillas_mutar[i], ratio_mutacion, funcion_str_genoma,
                                        funcion_generar_individuo_aleatorio, funcion_fitness, funcion_mutar)
            #self.poblacion_fitness[1,i] = self.poblacion[i].genoma
            self.poblacion_fitness[1,i] = i

    def __str__(self):
        string = "    POBLACION"
        string += "\n\nSize: " + str(self.size)
        string += "\nElitismo: " + str(self.elitismo)
        string += "\nInmortalidad: " + str(self.inmortalidad)
        string += "\nObjetivo: " + str(self.objetivo)
        string += "\nSeleccion: " + str(self.seleccion) + "\n"
        for i in range(self.size):
            string += "\n" + str(self.poblacion[self.poblacion_fitness[0,i]]) + "\n"

        string += "\n\n"

        return string[:-1]

    def _cruzar(self, individuos):
        """
        Metodo que genera individuos nuevos (generalmente dos) a partir del genoma de los padres

        Parametros
        ----------
        individuos : `list[Individuo.genoma]`
            lista de generalmente de los genomas de dos individuos a reproducir
        """
        genomas = self.funcion_cruzar_genomas(self.ratio_cruce, individuos)
        semillas_mutar=np.random.randint(100*self.size, size = 2)
        nuevos_individuos = [Individuo(self.siguiente_id, genomas[0], self.datos_auxiliares, None, semillas_mutar[0], self.ratio_mutacion, self.funcion_str_genoma,
                                        self.funcion_generar_individuo_aleatorio, self.funcion_fitness, self.funcion_mutar),
                             Individuo(self.siguiente_id+1, genomas[1], self.datos_auxiliares, None, semillas_mutar[1], self.ratio_mutacion, self.funcion_str_genoma,
                                        self.funcion_generar_individuo_aleatorio, self.funcion_fitness, self.funcion_mutar)]
        self.siguiente_id += 2
        return nuevos_individuos

    def _evaluar_poblacion(self):
        """
        Metodo donde se calcula el fitness de toda la poblacion
        """
        for i in range(len(self.poblacion_fitness[1])):
            self.poblacion[int(self.poblacion_fitness[0,i])].calcular_fitness(self.datos_auxiliares_poblacion)
            self.poblacion_fitness[1,i] = self.poblacion[self.poblacion_fitness[0,i]].fitness


    def _seleccion_ruleta(self):
        """
        Metodo donde se selecciona a la elite mediante ruleta. Se devuelven solo los identificadores
        de los individuos por eficiencia
        """
        elite = np.random.choice(
                                    a       = self.poblacion_fitness[0],
                                    size    = int(np.ceil(self.size*self.elitismo)),
                                    p       = self.poblacion_fitness[1]/np.sum(self.poblacion_fitness[1]),
                                    replace = True
                                ).astype(np.int_)
        return elite

    def _seleccion_rank(self):
        """
        Metodo donde se selecciona a la elite mediante rank. Se devuelven solo los identificadores
        de los individuos por eficiencia
        """
        probabilidad = 1 / (np.argsort(self.poblacion_fitness[1]) + 2) #Para evitar que el mejor tenga probabilidad de 1/0 se empieza en 1/2 y luego se divide entre el total ya que es serie armónica
        probabilidad /= sum(probabilidad)
        elite = np.random.choice(
                                    a       = self.poblacion_fitness[0],
                                    size    = int(np.ceil(self.size*self.elitismo)),
                                    p       = probabilidad,
                                    replace = True
                                ).astype(np.int_)
        return elite

    def _seleccion_torneo(self):
        """
        Metodo donde se selecciona a la elite mediante torneo. Se devuelven solo los identificadores
        de los individuos por eficiencia
        """
        elite = np.zeros(int(np.ceil(self.size*self.elitismo)), dtype=np.int_)
        for i in range(len(elite)):
            candidatos = np.random.choice(
                                    a       = self.poblacion_fitness[0],
                                    size    = 4,
                                    replace = False
                                         )
            elite[i] = candidatos[0]
            fitness_elite = self.poblacion[int(candidatos[0])].fitness
            for j in range(1,4):
                if self.poblacion[int(candidatos[j])].fitness > fitness_elite:
                    elite[i] = candidatos[j]
                    fitness_elite = self.poblacion[int(candidatos[j])].fitness
        return elite

    def _seleccion_elite(self):
        """
        Metodo donde se selecciona a la elite mediante el metodo de seleccion especificado
        """
        return self.funciones_seleccion[self.seleccion]()

    def _actualizar_poblacion(self):
        """
        Metodo donde de actualiza la poblacion. Primero se selecciona la elite mediante el
        metodo especificado. Posteriormente dependiendo del valor de la inmortalidad se
        vuelca la elite en la nueva poblacion o no. Finalmente se rellena con individuos
        nuevos fruto del cruce de otros dos
        """
        nueva_poblacion = {}
        nueva_poblacion_fitness = np.zeros((2,self.size))
        elite = list(set(self._seleccion_elite()))

        if self.inmortalidad:
            for i in range(len(elite)):
                nueva_poblacion[elite[i]] = self.poblacion[elite[i]]
                nueva_poblacion_fitness[0,i] = elite[i]
                nueva_poblacion_fitness[1,i] = self.poblacion[elite[i]].fitness

        for i in range(len(elite)*self.inmortalidad, self.size, 2):
            if len(elite) == 1:
                padres = np.asarray([elite[0],elite[0]])
            else:
                padres = np.random.choice(
                                        a       = elite,
                                        size    = 2,
                                        replace = False
                                        )
            padres = [self.poblacion[padres[0]].genoma, self.poblacion[padres[1]].genoma]
            hijos = self._cruzar(padres)

            hijos[0]._mutar()
            hijos[1]._mutar()

            nueva_poblacion[hijos[0].identificador] = hijos[0]
            nueva_poblacion_fitness[0,i] = hijos[0].identificador
            nueva_poblacion_fitness[1,i] = 0

            if i != self.size - 1:
                nueva_poblacion[hijos[1].identificador] = hijos[1]
                nueva_poblacion_fitness[0,i+1] = hijos[1].identificador
                nueva_poblacion_fitness[1,i+1] = 0
            else:
                self.siguiente_id -= 1
        self.poblacion = nueva_poblacion
        self.poblacion_fitness = nueva_poblacion_fitness


class AlgoritmoGenetico:
    """
    Clase que orquesta los metodos de una instancia de Poblacion para simular un
    algoritmo genetico sobre ella

    Atributos
    ----------
    identificador : `string`
        identificador univoco de la instancia del algoritmo genetico que puede llevar
        informacion del tipo de poblacion/individuo que se esta usando o cualquier detalle
        diferenciador entre instancias. Debe ser en formato snake (palabras unidas por
        barrabajas)

    poblacion : `PoblacionGenerico`
        objeto PoblacionGenerico donde se almacena todo lo referente a la poblacion actual

    iteracion_actual : `int`
        numero entero que indica la iteracion actual

    iteraciones : `int`
        numero entero que indica el numero de iteraciones

    fitness_actual : `float`
        numero que indica la idoneidad del mejor individuo ante el problema que
        trata de optimizar el algoritmo genetico

    limite_fitness : `float`
        numero que indica la idoneidad del mejor individuo ante el problema que
        trata de optimizar el algoritmo genetico a partir de la cual se considera resuelto
        el problema

    condicion_parada : `string`
        string con posibles valores "iteraciones", "fitness", "iteraciones_and_fitness" o
        "iteraciones_or_fitness" que sirve de key de la funcion correspondiente en el
        diccionario tipo_condicion_parada

    tipo_condicion_parada : `dict{string:function}`
        Diccionario de funciones que hace mas eficiente la toma de decicion de la parada
        del algoritmo

    historial_mejor_fitness : `list[float]`
        lista ordenada por iteraciones de los mejores fitness de cada iteracion

    historial_mejor_individuo : `list[Individuo]`
        lista ordenada por iteraciones de los mejores individuos de cada iteracion

    start_time : `float`
        numero flotante que hace las veces de marcador temporal de cuando empieza el
        algoritmo

    end_time : `float`
        numero flotante que hace las veces de marcador temporal de cuando acaba el
        algoritmo

    total_ime : `float`
        numero flotante fruto de la resta de los 2 anteriores que indica el tiempo
        transcurrido durante la ejecucion del algoritmo

    ejecutado : `bool`
        booleano que hace las veces de flag para comprobar si el algoritmo se ha
        ejecutado al menos una vez

    objetivo : `string`
        string que puede ser "Maximizar" o "Minimizar" que indica hacia donde
        optimiza el algoritmo

    tipo_objetivo : `dict{string:bool}`
        diccionario que sirve para simplificar calculos logicos relacionados
        con el objetivo
    """

    identificador = None
    poblacion = None
    iteracion_actual = 0
    iteraciones = None
    fitness_actual = None
    limite_fitness = None
    condicion_parada = None
    tipo_condicion_parada = None
    historial_mejor_fitness = []
    historial_mejor_individuo = []
    ganador = None
    start_time = None
    end_time = None
    total_ime = None
    ejecutado = False
    objetivo = None
    tipo_objetivo = {"Maximizar":True,"Minimizar":False}
    verbose = None

    def __init__(self, identificador, iteraciones, limite_fitness, condicion_parada, semilla, size, elitismo, ratio_mutacion, ratio_cruce, 
                 inmortalidad, tipo_seleccion, objetivo, datos_auxiliares, datos_auxiliares_poblacion, funcion_str_genoma, funcion_generar_individuo_aleatorio,
                 funcion_fitness, funcion_mutar, funcion_cruzar_genomas, verbose, path_resultados, ciclos_estancamiento, diferencia_estancamiento):
        self.poblacion = Poblacion(semilla, size, elitismo, ratio_mutacion, ratio_cruce, inmortalidad, tipo_seleccion, objetivo, datos_auxiliares,
                 datos_auxiliares_poblacion, funcion_str_genoma, funcion_generar_individuo_aleatorio, funcion_fitness, funcion_mutar,
                 funcion_cruzar_genomas)
        self.identificador = identificador
        self.iteraciones = iteraciones
        self.limite_fitness = limite_fitness
        self.condicion_parada = condicion_parada
        self.tipo_condicion_parada = {"iteraciones":self._condicion_parada_iteraciones,"fitness":self._condicion_parada_fitness,
                             "iteraciones_and_fitness":self._condicion_parada_iteraciones_and_fitness,
                             "iteraciones_or_fitness":self._condicion_parada_iteraciones_or_fitness}
        self.objetivo = self.tipo_objetivo[objetivo]
        self.verbose = verbose

        self.iteracion_actual = 0
        self.fitness_actual = None
        self.historial_mejor_fitness = []
        self.historial_mejor_individuo = []
        self.ganador = None
        self.ejecutado = False
        self.path_resultados = path_resultados
        self.ciclos_estancamiento = ciclos_estancamiento
        self.diferencia_estancamiento = diferencia_estancamiento


    def __str__(self):
        string = ""
        if self.ejecutado:
            string = "    ALGORITMO GENeTICO"
            string += "\n\nIdentificador: " + str(self.identificador)
            string += "\nEstado: Ejecutado"
            string += "\nIteraciones limite: " + str(self.iteraciones)
            string += "\nIteraciones finales: " + str(len(self.historial_mejor_fitness))
            string += "\nFitness limite: " + str(self.limite_fitness)
            string += "\nFitness final: " + str(self.historial_mejor_fitness[-1])
            string += "\nCondicion de parada: " + str(self.condicion_parada)
            string += "\nMejor indididuo: \n\n" + str(self.historial_mejor_individuo[-1])
        else:
            string = "\n  ALGORITMO GENeTICO"
            string += "\n\nIdentificador: " + str(self.identificador)
            string += "\nEstado: No Ejecutado"
            string += "\nIteraciones limite: " + str(self.iteraciones)
            string += "\nFitness limite: " + str(self.limite_fitness)
            string += "\nCondicion de parada: " + str(self.condicion_parada)
        return string

    def _actualizar_registro(self):
        """
        Metodo que actualiza los registros de la instancia aniadiendo los resultados
        de la nueva iteracion e incrementandola
        """
        self.iteracion_actual += 1
        if self.ganador == None:
            self.fitness_actual = self.objetivo*np.max(self.poblacion.poblacion_fitness[1])+abs(self.objetivo-1)*np.min(self.poblacion.poblacion_fitness[1])
            self.ganador = self.poblacion.poblacion[self.poblacion.poblacion_fitness[0,np.where(self.poblacion.poblacion_fitness[1]==self.fitness_actual)[0][0]]]
        else:
            fitness_antiguo = self.fitness_actual
            self.fitness_actual = abs(self.objetivo-1)*min(np.min(self.poblacion.poblacion_fitness[1]), self.fitness_actual)+self.objetivo*max(np.max(self.poblacion.poblacion_fitness[1]), self.fitness_actual)
            condicion = (fitness_antiguo > self.fitness_actual and not self.objetivo) or (fitness_antiguo < self.fitness_actual and self.objetivo)
            if condicion:
                self.ganador = self.poblacion.poblacion[self.poblacion.poblacion_fitness[0,np.where(self.poblacion.poblacion_fitness[1]==self.fitness_actual)[0][0]]]
        self.historial_mejor_individuo.append(self.ganador)
        self.historial_mejor_fitness.append(self.fitness_actual)


    def _condicion_parada_iteraciones(self):
        """
        Metodo que computa si se debe parar la ejecucion del algoritmo con el
        flag iteraciones
        """
        """
        condicion = (self.iteracion_actual > self.ciclos_estancamiento) and \
                    (self.diferencia_estancamiento >= abs(self.historial_mejor_fitness[-(self.ciclos_estancamiento)]-self.historial_mejor_fitness[-1])) and \
                    (self.iteracion_actual >= self.iteraciones)
        """
        condicion = self.iteracion_actual >= self.iteraciones
        if self.verbose:
            print("Comprobando iteraciones\n    Iteracion actual: " + str(self.iteracion_actual) + " Total de iteraciones: " + str(self.iteraciones))
        return condicion

    def _condicion_parada_fitness(self):
        """
        Metodo que computa si se debe parar la ejecucion del algoritmo con el
        flag fitness
        """
        condicion = (self.fitness_actual < self.limite_fitness and not self.objetivo) or (self.fitness_actual > self.limite_fitness and self.objetivo)
        if self.verbose:
            print("Comprobando fitness\n    Fitness actual: " + str(self.fitness_actual) + " Fitness limite: " + str(f'{self.limite_fitness:.20f}'))
        return condicion

    def _condicion_parada_iteraciones_and_fitness(self):
        """
        Metodo que computa si se debe parar la ejecucion del algoritmo con el
        flag iteraciones_and_fitness
        """
        condicion = self._condicion_parada_iteraciones() and self._condicion_parada_fitness()
        return condicion

    def _condicion_parada_iteraciones_or_fitness(self):
        """
        Metodo que computa si se debe parar la ejecucion del algoritmo con el
        flag iteraciones_or_fitness
        """
        condicion = self._condicion_parada_iteraciones() or self._condicion_parada_fitness()
        return condicion

    def plot_historico_fitness(self):
        """
        Metodo que genera una imagen  de 8x8 pulgadas con el grafico del historico de fitness
        """
        if self.ejecutado:
            plt.rcParams["figure.figsize"] = [8, 8]
            plt.rcParams["figure.autolayout"] = True

            print("Iteraciones")
            print(np.asarray(range(len(self.historial_mejor_fitness))))
            print("Mejor fitness")
            print(np.asarray(self.historial_mejor_fitness))

            df = pd.DataFrame({"Iteraciones": np.asarray(range(len(self.historial_mejor_fitness))),
                               "Mejor fitness": np.asarray(self.historial_mejor_fitness)})

            plot = sns.lineplot(data=df, x="Iteraciones", y="Mejor fitness")
            plot.figure.savefig(str(Path.joinpath(Path(self.path_resultados), Path(self.identificador + ".png"))))

    def ejecutar(self):###AniADIR PRINTS QUE INDIQUEN ESTADO ACTUAL CON LA AYUDA DE LA LIBRERIA ADMINISTRADOR PANTALLA
        """
        Metodo que ejecuta el algoritmo genetico como tal
        """
        if self.verbose:
            print("Comenzando ejecucion de algoritmo genético " + str(self.identificador))

        self.ejecutado = True

        if self.verbose:
            print("Midiendo tiempo")

        self.start_time = time.time()

        if self.verbose:
            print("Evaluando poblacion")

        self.poblacion._evaluar_poblacion()
        #print(self.poblacion)
        if self.verbose:
            print("Actualizando registro")

        self._actualizar_registro()

        while not self.tipo_condicion_parada[self.condicion_parada]():
            if self.verbose:
                start_time_iteracion = time.time()
                print("Iteracion: " + str(self.iteracion_actual) + "/" + str(self.iteraciones))
            if self.verbose:
                print("Actualizando poblacion")
            self.poblacion._actualizar_poblacion()
            if self.verbose:
                print("Evaluando poblacion")
            self.poblacion._evaluar_poblacion()
            #print(self.poblacion)
            if self.verbose:
                print("Actualizando registro")
            self._actualizar_registro()
            if self.verbose:
                end_time_iteracion = time.time()
                total_time_iteracion = end_time_iteracion - start_time_iteracion
                print("Iteracion ejecutada en " + str(total_time_iteracion) + " segundos\n")


        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time

        if self.verbose:
            print("Tiempo total " + str(self.total_time) + " segundos\n\n\n")