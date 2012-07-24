#!/usr/bin/python
# -*- coding: iso-8859-1 -*-
import sys
import os
import cStringIO
from os.path import *
from re import *
from numpy import *

sys.path.append("./")

from descritores import curvatura

# Esta função é chamada pelo método os.path.walk()
# Coloca em duas listas (arg[0], arg[1], arg[2]) 
# os nomes dos arquivos das figuras que calcularemos a energia de dobramento
# , o nome para o arquivo de armazenamento das saídas e o nome do arquivo
# do grafico.
# Qualdo o método os.path.walk() executa, este se encarrega de visitar
# cada diretório abaixo da raíz fornecida chamando esta função. 
# Antes de chamar visit() os.path.walk() passa como parâmetro, em d,
# o nome do diretório visitado e em fl uma lista de arquivos
# contidos neste mesmo diretório 
def visit(arg,d,fl):
  for f in fl:
   aux = join(d,f)   
   if isfile(aux):
     r = compile(sys.argv[2]+"$")
     if r.search(aux):
       # arquivo .png (imagem para calculo da curvatura)
       arg[0].append(aux)
      # nome do arquivo .dat para saída
       arg[1].append(r.sub("dat",aux))
       arg[2].append(r.sub("ps",aux))

lista_de_arquivos = [[],[],[]]
# Obtém listas dos nomes dos arquivos a partir da 
# raiz fornecida ao script
if sys.argv[1]: 
 walk(sys.argv[1],visit,lista_de_arquivos)
else: os.exit(-1)

sigma= linspace(2,50,10)
for im_file,out_file in zip(lista_de_arquivos[0],lista_de_arquivos[1]):
  #print im_file,"\t",out_file,"\t",plt_file,"\n"
  output = cStringIO.StringIO()
  fout = open(out_file,"w")
  k = curvatura(im_file,sigma)

  for a in k.curvs:
   for b in a:
    output.write("{0: < 5,.3f} ".format(a))
   output.write("\n")

  fout.write(output.getvalue())
  output.close()
  fout.close()
#  a = loadtxt(out_file)
#  plot(log(1/sigma[::-1]),a,".") 
#  savefig(plt_file)
#  clf()   
