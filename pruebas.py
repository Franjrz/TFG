import sys
sys.path.insert(0, '..') # Add qiskit_runtime directory to the path

from qiskit import IBMQ
TOKEN = "4e01dc0c56e6f9941836f52ea4255e797119debe6afab3b25bcb0388e285f3977c743bd3d1c25ebcc12991e8332daf8436a69d8bbb4b7212713a367a5eaea23d"

#IBMQ.delete_account()
IBMQ.save_account(TOKEN)
provider = IBMQ.load_account()
#backend = provider.get_backend('ibmq_qasm_simulator')
backend = provider.get_backend('ibmq_qasm_simulator')

import numpy as np
from random import randint
from qiskit import *
from qiskit.tools.visualization import circuit_drawer
from matplotlib import pyplot as plt
from pylab import cm
from sklearn.svm import SVC
from sklearn import metrics
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/TFG/datasetPrueba.csv',sep=',', header=None) # alterative problem: dataset_graph10.csv
data = df.values

print(df.head(4))

# choose number of training and test samples per class:
num_train = 4
num_test = 4

# extract training and test sets and sort them by class label
train = data[:2*num_train, :]
test = data[2*num_train:2*(num_train+num_test), :]

ind=np.argsort(train[:,-1])
x_train = train[ind][:,:-1]
y_train = train[ind][:,-1]

ind=np.argsort(test[:,-1])
x_test = test[ind][:,:-1]
y_test = test[ind][:,-1]

d = np.shape(data)[1]-1                                         # feature dimension is twice the qubit number

em = [[0,2],[3,4],[2,5],[1,4],[2,3],[4,6]]                      # we'll match this to the 7-qubit graph
# em = [[0,1],[2,3],[4,5],[6,7],[8,9],[1,2],[3,4],[5,6],[7,8]]  # we'll match this to the 10-qubit graph

fm = FeatureMap(feature_dimension=d, entangler_map=em)          # define the feature map
initial_point = [0.1]                                           # set the initial parameter for the feature map

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


plt.rcParams['font.size'] = 20
plt.imshow(result['aligned_kernel_matrix']-np.identity(2*num_train), cmap=cm.get_cmap('bwr', 20))
plt.show()


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
