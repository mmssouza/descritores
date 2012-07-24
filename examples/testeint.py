#!/usr/bin/python
# -*- coding: iso-8859-1 -*-
import sys
import numpy as np
import scipy.stats
from matplotlib.pyplot import *

sys.path.append("./")
from descritores import integralinvariant
 
for aux in sys.argv[1:]:
 y = integralinvariant(aux,0.05,5,500)
 yfft = np.fft.fft(y()-np.mean(y()))
 plot(10*np.log10(abs(1+np.fft.fftshift(yfft*yfft.conjugate()))))
show()

