#! /usr/bin/python
# -*- coding: iso-8859-1 -*-

import sys
import glob
from numpy import *
from scipy.interpolate import interp1d
from matplotlib.pyplot import *
from curvature import *

lwidth = 2.
fsize = 10
N = 20
# s = vetor que armazena diferentes valores de desvio padrao sigma da gaussiana
# utilizada para janela de suavizacao 
sigma_range = logspace(0.2,1.5,N)
s = sigma_range

# Instancializa objeto
c1 = curvature(sys.argv[1],sigma_range,smooth="Gaussian",order = 1)
c2 = curvature(sys.argv[1],sigma_range,smooth="Average",order = 1)
c3 = curvature(sys.argv[1],sigma_range,smooth="Average",order = 2)
#c4 = curvature(sys.argv[1],sigma_range,smooth="Average",order = 20)
figure(1)

subplot(221)
plot(c1.z.real,c1.z.imag,lw=lwidth)
xlabel('x',fontsize=fsize)
ylabel('y',fontsize=fsize)

for i in arange(N):
 plot(c1.rcontours[i].real,c1.rcontours[i].imag,lw=lwidth)
xlabel('x',fontsize=fsize)
ylabel('y',fontsize=fsize)
title('Gaussiana',fontsize=fsize)

subplot(222)

plot(c2.z.real,c2.z.imag,lw=lwidth)
xlabel('x',fontsize=fsize)
ylabel('y',fontsize=fsize)

for i in arange(N):
 plot(c2.rcontours[i].real,c2.rcontours[i].imag,lw=lwidth)
xlabel('x',fontsize=fsize)
ylabel('y',fontsize=fsize)
title('Media Movel 1st order',fontsize=fsize)

subplot(223)

plot(c3.z.real,c3.z.imag,lw=lwidth)
xlabel('x',fontsize=fsize)
ylabel('y',fontsize=fsize)

for i in arange(N):
 plot(c3.rcontours[i].real,c3.rcontours[i].imag,lw=lwidth)
xlabel('x',fontsize=fsize)
ylabel('y',fontsize=fsize)
title('Media Movel 2nd order',fontsize=fsize)

show()
