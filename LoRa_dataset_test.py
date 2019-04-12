# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 23:55:23 2019

@author: tiarl
"""

import scipy
from scipy import stats
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

filenames = glob('gr-lora-samples-master/*.cfile')
dict_vectors = {}
all_vectors = np.array([])
for filename in filenames:
    vector = np.array(scipy.fromfile(open(filename), dtype=scipy.uint8))
    all_vectors = np.concatenate([all_vectors, vector])
    dict_vectors[filename.split('\\')[1][:-6]] = vector
    print("-->", filename.split('\\')[1][:-6])
    print("    %21s" % "Numero de amostras = ", vector.size)
    print("    %21s" % "Média = ", vector.mean())
    print("    %21s" % "Desvio Padrão = ", vector.std())
    print("    %21s" % "Mínimo = ", vector.min())
    print("    %21s" % "Máximo = ", vector.max())
    print("    %21s" % "Moda   = ", stats.mode(vector).mode[0])
    
    plt.plot(vector, 'bo--')
    plt.show()
    
print("")
print("")
print("Todos em sequência:")
    
plt.figure(figsize=(10,2))
plt.plot(all_vectors, 'b-')
plt.show()

print("")
print("")
print("Correlação entre os dados do Dataset:")

df = pd.DataFrame(dict_vectors)

correlations = np.array(df.corr())

plt.pcolor(correlations, cmap='RdBu')
plt.show()