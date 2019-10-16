# -*- coding: iso-8859-1 -*-
# descritores : m�dulo que implementa o c�lculo de assinaturas e descritores de imagens

import numpy as np
import cv2
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist,squareform
from math import sqrt,acos
#from oct2py import Oct2Py
import atexit

#oc = Oct2Py('/usr/bin/octave-cli')
#atexit.register(oc.exit)

class contour_base:
 '''Represents an binary image contour as a complex discrete signal.
   Some calculation methods are provided to compute contour 1st derivative, 2nd derivatives and perimeter.
   The signal variable (self.c) is represented as a single dimensional ndarray of complex.
   This class is interable and callable so, interation over objects results in sequential access to each signal variable element. Furthermore, calling the object as a function yields as return value te signal variable.

 '''

 def __init__(self,fn,nc = 256,method = 'cv'):
  self.__i = 0
  if method == 'octave':
   pass
   #if type(fn) is str:
 #   im = oc.imread(fn)
 #   s = oc.extract_longest_cont(im,nc)
 #   self.c = np.array([complex(i[0],i[1]) for i in s])
#   elif type(fn) is ndarray:
#    self.c = fn
  else:
   if type(fn) is str:
    im = cv2.imread(fn,cv2.IMREAD_GRAYSCALE )
    image, s, hierarchy = cv2.findContours(im,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    self.c = np.array([complex(i[0][1],i[0][0]) for i in s[0]])
   elif (type(fn) is np.ndarray):
    self.c = fn
   elif (type(fn) is cv2.iplimage):
    image, s, hierarchy = cv2.findContours(fn,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    #s = cv2.FindContours(fn,cv2.CreateMemStorage(),cv2.CV_RETR_LIST,cv2.CV_CHAIN_APPROX_NONE)
    self.c = np.array([complex(i[0][1],i[0][0]) for i in s[0]])
  N = self.c.size
  self.freq = np.fft.fftfreq(N,1./float(N))

  self.ftc = np.fft.fft(self.c)

  if isinstance(self,contour_base):
   self.calc_derivatives()

 def calc_derivatives(self):
   ftcd = np.complex(0,1) * 2 * np.pi * self.freq * self.ftc
   ftcdd = - (2 * np.pi * self.freq)**2 * self.ftc
   self.cd = np.fft.ifft(ftcd)
   self.cdd = np.fft.ifft(ftcdd)

 def first_deriv(self):
  '''Return the contour signal 1st derivative'''
  return self.cd

 def second_deriv(self):
  '''Return the contour signal 2nd derivative'''
  return self.cdd

 def perimeter(self):
  '''Calculate and return the contour perimeter'''
  return (2*np.pi*np.sum(np.abs(self.cd))/float(self.cd.size))

 def __iter__(self): return self

 def next(self):

  if self.__i > self.c.size-1:
   self.__i = 0
   raise StopIteration
  else:
   self.__i += 1
   return self.c[self.__i-1]

 def __call__(self): return self.c

class contour(contour_base):
  '''Like contour_base except that, prior to derive a complex signal representation, smooths the image contour using a Gaussian kernel. The kernel parameter (gaussian standard deviation) is the second constructor parameter. See also contour_base.'''

  # Gaussian smoothing function
  def __G(self,s):
    return (1/(s*(2*np.pi)**0.5))*np.exp(-self.freq**2/(2*s**2))

  def __init__(self,fn,sigma=None,nc = 256,method = 'cv'):
   contour_base.__init__(self,fn,nc = nc,method = method)
   if sigma is not None:
    E = np.sum(self.ftc * self.ftc.conjugate())
    self.ftc = self.ftc * self.__G(sigma)
    Eg  = np.sum(self.ftc * self.ftc.conjugate())
    k = sqrt(abs(E/Eg))
    self.c = np.fft.ifft(self.ftc)*k
    self.calc_derivatives()
    self.cd = self.cd * k
    self.cdd = self.cdd * k


# classe curvatura : calcula a curvatura de um contorno para v�rios n�veis de suaviza��o
# Par�metros do Construtor:   def __init__(self,fn = None,sigma_range = np.linspace(2,30,10))
#  fn : Pode ser o nome de um arquivo de imagem (string) que contenha uma forma bin�ria ou um vetor (ndarray) de valores das
# coordenadas do contorno de uma forma (representa��o complexa x+j.y).
# No primeiro caso os contornos s�o extra�dos atrav�s da fun��o cv.FindContours() da biblioteca Opencv
#  sigma_range :  vetor (ndarray) que cont�m os valores que ser�o utilizados como desvio padr�o para o FPB Gaussiana.
# que filtra os contorno antes do c�lculo da curvatura.
 #  when zero no filtering is applied to contour

class curvatura:
  '''For a given binary image calculates and yields a family of curvature signals represented in a two dimensional ndarray structure; each row corresponds to the curvature signal derived from the smoothed contour for a certain smooth level.'''

  def __Calcula_Curvograma(self,fn,nc = 256,method = 'cv'):
   if type(fn) is contour:
    z = fn
   else:
    z = contour(fn,nc = nc,method = method)
   caux = [contour(z(),s) for s in self.sigmas]
   caux.append(z)
   self.contours = np.array(caux)
   self.t = np.linspace(0,1,z().size)
   self.curvs = np.ndarray((self.sigmas.size+1,self.t.size),dtype = "float")

   for c,i in zip(self.contours,np.arange(self.contours.size)):
    # Calcula curvatura para varias escalas de suaviza��o do contorno da forma
     curv = c.first_deriv() * np.conjugate(c.second_deriv())
     curv = - curv.imag
     curv = curv/(np.abs(c.first_deriv())**3)
     # Array bidimensional curvs = Curvature Function k(sigma,t)
     self.curvs[i] = curv

  # Contructor
  def __init__(self,fn = None,sigma_range = np.linspace(2,30,20),nc = 256,method = 'cv'):
   # Extrai contorno da imagem
   self.sigmas = sigma_range
   self.__Calcula_Curvograma(fn,nc = nc,method = method)

 # Function to compute curvature
 # It is called into class constructor
  def __call__(self,idx = 0,t= None):
    if t is None:
     __curv = self.curvs[idx]
    elif (type(t) is np.ndarray):
     __curv = interp1d(self.t,y = self.curvs[idx],kind='quadratic')
     return(__curv(t))
    else:
      __curv = self.curvs[idx]

    return(__curv)

class bendenergy:
 ''' For a given binary image, computes the multiscale contour curvature bend energy descriptor'''

 def __init__(self,fn,scale,nc = 256,method = 'cv'):
  self.__i = 0
  k = curvatura(fn,scale[::-1],nc = nc,method = method)
  # p = perimetro do contorno nao suavisado
  p = k.contours[-1].perimeter()
  self.phi  = np.array([(p**2)*np.mean(k(i)**2) for i in np.arange(0,scale.size)])

 def __call__(self): return self.phi

 def __iter__(self): return self

 def next(self):

   if self.__i > self.phi.size-1:
    self.__i = 0
    raise StopIteration
   else:
    self.__i += 1
    return self.phi[self.__i-1]

