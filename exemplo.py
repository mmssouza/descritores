
import descritores as desc
import pylab

entrada = "mullets.png"
tas = desc.TAS(entrada)
pylab.plot(tas.sig)
pylab.show()

