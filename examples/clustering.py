#!/usr/bin/python
# -*- coding: iso-8859-1 -*-
import sys
import os
import cStringIO
import numpy as np
import math
from matplotlib.pyplot import figure,plot,savefig,clf
from os.path import walk,join,isfile
from re import compile
sys.path.append("./")
from descritores import curvatura,bendenergy
from scipy.spatial.distance import pdist,squareform

# parametros

# caminho : aponta para local das figuras e dados
caminho = sys.argv[1];

# extensao : extensao do arquivo de imagem a processar
extensao = sys.argv[2];

# passo :etapa de execucao do script
# passo = 0 : Geracao dos descritores
# passo = 1 : Calculo de distancias e geracao de resultados 
passo = sys.argv[3];

# sigma_min : menor valor do fator de escala
# Quanto menor menor será o valor da frequencia de corte inferior
# do filtro a ser aplicado na curvatura 
sigma_min = -0.1

# sigma_max : maior valor do fator de escala
# Quanto maior maior sera o valor da frequencia de corte superior do filtro 
# a ser aplicado na curvatura 
sigma_max = 2.3

# sigma_n : numero de pontos para a curvatura multiescala
sigma_n = 2000 

# distancia : medida de dissimilaridade a ser empregada 
distancias = ['braycurtis','canberra','chebyshev','cityblock','correlation',
              'cosine','dice','euclidean','hamming','jaccard',
              'kulsinski','mahalanobis','matching','minkowski',
              'rogerstanimoto','russelrao','seuclidean','sokalmichener',
              'sokalsneath','sqeuclidean','yule']

distancia = 'canberra'
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
     r = compile(extensao+"$")
     if r.search(aux):
       # arquivo .png (imagem para calculo da curvatura)
       arg[0].append(aux)
      # nome do arquivo .dat para saída
       arg[1].append(r.sub("txt",aux))
       arg[2].append(r.sub("ps",aux))

lista_de_arquivos = [[],[],[]]
# Obtém listas dos nomes dos arquivos a partir da 
# raiz fornecida ao script
if caminho: 
 walk(caminho,visit,lista_de_arquivos)
else: os.exit(-1)

if passo == '0':
 # Le arquivo de entrada e gera descritores
 # plota graficos e gera arquivos de saida 
 sigma= np.logspace(sigma_min,sigma_max,sigma_n)
 figure(1)
 for im_file,out_file,plt_file in zip(lista_de_arquivos[0],lista_de_arquivos[1],lista_de_arquivos[2]):
#   print im_file,"\t",out_file,"\t",plt_file,"\n"
   output = cStringIO.StringIO()
   fout = open(out_file,"w")
   k = curvatura(im_file,sigma)
   nmbe  = bendenergy(k)
   for a in nmbe.phi[::-1]:
    output.write("{0: < 5,.3f} ".format(math.log(a)))
    output.write("\n")
   fout.write(output.getvalue())
   output.close()
   fout.close()
   a = np.loadtxt(out_file)
   plot(np.log(1/sigma[::-1]),a,".") 
   savefig(plt_file)
   clf()  

elif passo == '1':

# dicionario que associa cada figura a classe que esta pertence
 classe_dic = dict(zip(('./bunny04.png','./mgen1bp.png','./easterncottontail.png','./mountaincottontailrot.png','./mgen2fp.png','./marshrabbit.png','./mgen2ap.png','./fgen2fp.png','./fgen1fp.png','./fgen2dp.png','./fgen5cp.png','./fgen1ep.png','./pygmyrabbit.png','./herrings.png','./mullets.png','./skyhawkocc1.png','./fgen3bp.png','./swamprabbit.png','./skyhawk.png','./swamprabbitocc2.png','./bonefishes.png','./fgen1ap.png','./phantomocc1.png','./bonefishesocc1.png','./fish30.png','./fish14.png','./fish23.png','./cow1.png','./fox1.png','./fish28.png','./fgen1bp.png','./whalesharks.png','./tool38.png','./dude5.png','./dude0.png','./tool04bent1.png','./dude11.png','./dude12.png','./tool07.png','./handdeform2.png','./dude1.png','./tool09.png','./dogfishsharks.png','./phantom.png','./dude4.png','./tool04.png','./handdeform.png','./tool27.png','./dude10.png','./dude2.png','./tool12.png','./tool44.png','./tool22.png','./tool17.png','./f16occ1.png','./dude6.png','./dude7.png','./dog2.png','./dude8.png','./f16.png','./kk0731.png','./f15.png','./harrierocc1.png','./harrierocc2.png','./harrier.png','./harrierocc3.png','./desertcottontail.png','./swordfishes.png','./dog3.png','./kk0739.png','./hand2occ3.png','./cat2.png','./kk0732.png','./handbent2.png','./handbent1.png','./hand2occ2.png','./hand3.png','./kk0741.png','./kk0737.png','./kk0738.png','./kk0736.png','./cat1.png','./hand90.png','./hand.png','./kk0728.png','./mountaincottontailocc2.png','./kk0740.png','./hand2.png','./kk0729.png','./hand2occ1.png','./kk0735.png','./calf1.png','./mountaincottontail.png','./calf2.png','./cow2.png','./tool08.png','./mountaincottontailocc1.png','./donkey1.png','./dog1.png'),(2,4,2,2,4,2,4,4,4,4,4,4,2,1,1,3,4,2,3,2,1,4,3,1,1,1,1,8,8,1,4,1,5,7,7,5,7,7,5,6,7,5,1,3,7,5,6,5,7,7,5,5,5,5,3,7,7,8,7,3,9,3,3,3,3,3,2,1,8,9,6,8,9,6,6,6,6,9,9,9,9,8,6,6,9,2,9,6,9,6,9,8,2,8,8,5,2,8,8)))

 output = cStringIO.StringIO()
 output.write("\ndistancia  {0}\n".format(distancia))
 output.write("sigma_min  {0: < 2,.3f}\nsigma_max {1: < 2,.3f}\nsigma_n {2: < d}\n\n".format(sigma_min,sigma_max,sigma_n))

# Monta matriz de observacoes (mt).
# Cada linha corresponde ao vetor de caracterisicas de uma das imagens 
# da classe.
# O vetor de características, para cada imagem da base de dados,
# foi obtido no passo '0' deste script e os resultados armazenadas na
# lista_de_arquivos[1]
 l = []

 for out_file in lista_de_arquivos[1]:
  l.append(np.loadtxt(out_file)[0:4])

 mt = np.vstack(tuple(l))
 
# Calcula matriz de distancias de todos a todos a partir da matriz mt 
 md = squareform(pdist(mt,distancia))
 
# Processa cada linha da md para estabelecer rank de recuperacao para 
# cada uma das formas
 for i in np.arange(md.shape[0]):

  # Obtem nome das figuras em ordem crescente de distancia
  # na forma de array
  idx = np.argsort(md[i])
  if (idx[0] != i):
   k = (idx == i).nonzero()  
   t = idx[0]
   idx[0] = idx[k[0][0]]
   idx[k[0][0]] = t
   
  aux = np.array(lista_de_arquivos[0])[idx]

  # pega classe a qual o primeiro padrao pertence
  output.write(aux[0])
  classe_padrao = classe_dic[aux[0]]
  # estamos interessados apenas nos 10 subsequentes resultados
  retrs = np.array(aux[1:11])
  # Contador para contabilizar desempenho
  corretos = 0
  # Avalia e despeja resultados na saida
  for tmp,k in zip(retrs,idx[1:11]):
   #output.write(" ({0},{1},{2: < .12f});".format(tmp,classe_dic[tmp],md[i][k]))
   if (classe_padrao == classe_dic[tmp]):
    corretos = corretos + 1

  output.write(" {0: < 2,.0f}\n".format(float(100.*corretos/10.)))
 
 sys.stdout.write(output.getvalue())  

 output.close()
