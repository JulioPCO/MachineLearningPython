# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 16:58:15 2020

@author: julio
"""

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection

rede = FeedForwardNetwork()
camadaEntrada = LinearLayer(2)
camadaOculta = SigmoidLayer(3)
camadaSaida = SigmoidLayer(1)
bias1 = BiasUnit()
bias2 = BiasUnit()

rede.addModule(camadaEntrada)
rede.addModule(camadaOculta)
rede.addModule(camadaSaida)
rede.addModule(bias1)
rede.addModule(bias2)

entradaOculta = FullConnection(camadaEntrada,camadaOculta)
entradaSaida = FullConnection(camadaOculta,camadaSaida)
biasOculta = FullConnection(bias1, camadaOculta)
biasSaida = FullConnection(bias2,camadaSaida)

rede.sortModules()

print(rede)
print(entradaOculta.params)
print(entradaSaida.params)
print(biasOculta.params)
print(biasSaida.params)