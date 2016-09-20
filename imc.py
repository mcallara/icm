import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

class matriz(object):
	@staticmethod
	def Energy(xi,xj,yi,h,eta,beta):
		E=h*xi-eta*xi*yi-sum(beta*xj*xi)
		return E

	def __init__(self,X,Y):
		#Y son las variables que observamos
		#X son las hidden variables
		self.X=X
		self.Y=Y
		self.directions={'U':np.array([-1,0]),'D':np.array([1,0]),'R':np.array([0,-1]),'L':np.array([0,1])}

	def GetDirValue(self,point,direction):
		#Con numpy podemos acceder, por ejemplo, a (-1,0) por eso tenemos que evitar acceder a ese valor.
		if np.all(np.array(point)+self.directions[direction]>=0):
			try:
				return self.X[tuple(np.array(point)+self.directions[direction])]
			except IndexError:
				return np.nan
		else:
			return np.nan
	
	def Get4DirValues(self,point):
		return np.array([self.GetDirValue(point,direction) for direction in self.directions])

	def LocalEnergy(self,point):
		"point es un tupla, y direction una letra que indica la direccion"
		xi=self.X[point]
		yi=self.Y[point]
		xj=self.Get4DirValues(point)
		xj=xj[~np.isnan(xj)]
		Eactual=self.Energy(xi,xj,yi,h,eta,beta)
		xi=xi*-1
		Emod=self.Energy(xi,xj,yi,h,eta,beta)
		return Eactual,Emod

	def Flip(self,point):
		self.X[point]=self.X[point]*-1

	def ShowMat(self):
		plt.imshow(self.X)
		plt.show()

	def Optimize(self):
		for i in range(self.X.shape[0]):
			for j in range(self.X.shape[1]):
				Eactual,Emod=self.LocalEnergy((i,j))
				#print("Punto {0} con valor Actual: {1} y valor Modificiado: {2}".format((i,j),Eactual,Emod))
				if Emod<Eactual:
					self.Flip((i,j))
					print("Se cambia valor en ({0},{1}), con Energia:{2} y pasa a tener:{3}".format(i,j,Eactual,Emod))

#Cargando la matriz
X2=misc.imread('img2.png')
X2=X2.astype(int)
X2[X2==0]=-1
X2[X2==255]=1
imagen=matriz(X2,X2)

h=0
eta=2.1
beta=1.2

print("Parametros: h={0}, eta={1}, beta={2}".format(h,eta,beta))

#Muestro la imagen ruidosa
imagen.ShowMat()

#Optimizamos
imagen.Optimize()

#Muestro Resultado
imagen.ShowMat()
