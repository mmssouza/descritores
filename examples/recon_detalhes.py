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
N = 5
# s = vetor que armazena diferentes valores de desvio padrao sigma da gaussiana
# utilizada para janela de suavizacao 
sigma_range = logspace(-0.05,0.7,N)
s = sigma_range

# Instancializa objeto
c1 = curvature(sys.argv[1],sigma_range,smooth="Gaussian",order = 1)
c2 = curvature(sys.argv[1],sigma_range,smooth="Average",order = 1)
c3 = curvature(sys.argv[1],sigma_range,smooth="Average",order = 10)
c4 = curvature(sys.argv[1],sigma_range,smooth="Average",order = 20)
figure(1)

subplot(221)
#plot(c1.z.real,c1.z.imag,lw=lwidth)
xlabel('x',fontsize=fsize)
ylabel('y',fontsize=fsize)
caux = c1.z - c1.rcontours[0]
for i in arange(1,N):
 c2[ = c1.rcontours[i]-c1.rcontours[i-1]
 plot(caux.real,caux.imag,lw=lwidth)
 
 
xlabel('x',fontsize=fsize)
ylabel('y',fontsize=fsize)
title('coeficientes de detalhes',fontsize=fsize)

subplot(222)

plot(c2.z.real,c2.z.imag,lw=lwidth)
xlabel('x',fontsize=fsize)
ylabel('y',fontsize=fsize)

for i in arange(N):
 plot(c2.rcontours[i].real,c2.rcontours[i].imag,lw=lwidth)
xlabel('x',fontsize=fsize)
ylabel('y',fontsize=fsize)
title('3rd order',fontsize=fsize)

subplot(223)

plot(c3.z.real,c3.z.imag,lw=lwidth)
xlabel('x',fontsize=fsize)
ylabel('y',fontsize=fsize)

for i in arange(N):
 plot(c3.rcontours[i].real,c3.rcontours[i].imag,lw=lwidth)
xlabel('x',fontsize=fsize)
ylabel('y',fontsize=fsize)
title('5th order',fontsize=fsize)

subplot(224)

plot(c4.z.real,c4.z.imag,lw=lwidth)
xlabel('x',fontsize=fsize)
ylabel('y',fontsize=fsize)

for i in arange(N):
 plot(c4.rcontours[i].real,c4.rcontours[i].imag,lw=lwidth)
xlabel('x',fontsize=fsize)
ylabel('y',fontsize=fsize)
title('7th order',fontsize=fsize)
show()
