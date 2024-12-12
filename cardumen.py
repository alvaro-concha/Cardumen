import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from errno import EEXIST
from os import makedirs,path

N = 128
M = 400
maxVel = 100
maxDist = 100
nIter = 100

class R2(np.ndarray):
	"""
	Attributes
	----------
	inputArray: array like
		Coordinates in 2D space
	"""
	def __new__(cls,inputArray):
		obj = np.asarray(inputArray).view(cls)
		return obj

class Pez:
	"""
	Attributes
	----------
	r: R2
		Position of the Pez in 2D space
	v: R2
		Velocity of the Pez in 2D space
	"""
	def __init__(self,position,velocity):
		self.r = position
		self.v = velocity

class Cardumen:
	"""
	Attributes
	----------
	_N: int
		Size of Cardumen, i.e., number of composing Pez
	_M: float
		Size of the 2D square space where Cardumen lives
	_maxVel: float
		Maximum speed for Pez
	_maxDist: float
		Maximum distance for Pez nearest neighbours
	peces: list
		List of Pez in Cardumen
	t: int
		Time step counter
	rc: R2
		Position of the center of the Cardumen in 2D space
	vc: R2
		Velocity of the center of the Cardumen in 2D space
	Methods
	-------
	initialize(M,maxVel,maxDist):
		Initializes Pez in Cardumen in an MxM space
	doCenter():
		Computes the position and velocity of the center of the Cardumen
	evolutionRule1(r):
		Pez in Cardumen drifts towards its center
	evolutionRule2(r):
		Pez in Cardumen avoids collisions with other Pez
	evolutionRule3(v):
		Velocity of Pez in Cardumen approaches the velocity of its center
	doStep():
		Updates state of the Cardumen in a single time step
	printCardumen(save,plot):
		Plots the positions and velocities of the Pez in Cardumen
		If save=True saves the plot as a .png image
		If plot=True shows the plot
	"""
	def __init__(self,N):
		self._N = N

	def initialize(self,M,maxVel,maxDist):
		self._M = M
		self._maxVel = maxVel
		self._maxDist = maxDist
		peces = []
		np.random.seed(42)
		for i in range(self._N):
			r = R2(np.random.uniform(low=0,high=M,size=2))
			vNorm = np.random.uniform(low=0,high=maxVel)
			vAngle = np.random.uniform(low=0,high=2*np.pi)
			v = vNorm*R2([np.cos(vAngle), np.sin(vAngle)])
			peces.append(Pez(r,v))
		self.peces = peces
		self.t = 0

	def doCenter(self):
		rc, vc = R2([0,0]), R2([0,0])
		for pez in self.peces:
			rc = rc + pez.r
			vc = vc + pez.v
		rc /= self._N
		vc /= self._N
		self.rc, self.vc = rc, vc

	def evolutionRule1(self,r):
		return (self.rc-r)*0.125

	def evolutionRule2(self,r):
		deltaVel = 0
		for pez in self.peces:
			diff =  r - pez.r
			dist = np.linalg.norm(diff)
			if (dist != 0) & (dist < self._maxDist):
				deltaVel += diff/dist
		return deltaVel

	def evolutionRule3(self,v):
		return (self.vc-v)*0.125

	def doStep(self):
		self.doCenter()
		for pez in self.peces:
			deltaVel = self.evolutionRule1(pez.r) + self.evolutionRule2(pez.r) + self.evolutionRule3(pez.v)
			pez.v += deltaVel
			vNorm = np.linalg.norm(pez.v)
			if vNorm > self._maxVel:
				pez.v = pez.v*self._maxVel/vNorm
			pez.r += pez.v
		self.t += 1

	def printCardumen(self,save=True,plot=False):
		cmap = cm.get_cmap('jet')
		plt.figure(figsize=(5,5))
		for pez in self.peces:
			plt.quiver(
				pez.r[0],
				pez.r[1],
				pez.v[0],
				pez.v[1],
				edgecolor='k',
				facecolor=cmap(np.arctan2(pez.v[1],pez.v[0])/np.pi*0.5+0.5),
				linewidth=.5,
				alpha=.5
				)
		plt.plot([0,self._M,self._M,0,0],[0,0,self._M,self._M,0],linestyle="dashed",c="grey",alpha=0.5)
		plt.xticks([])
		plt.yticks([])
		plt.xlim(-0.5*self._M,1.5*self._M)
		plt.ylim(-0.5*self._M,1.5*self._M)
		if save:
			try:
				makedirs("cardumen")
			except OSError as exc:
				if exc.errno == EEXIST and path.isdir("cardumen"):
					pass
				else: raise
			plt.savefig(f"cardumen/estado{self.t:03d}.png",bbox_inches="tight")
		if plot:
			plt.show()
		plt.close()

def run_cardumen():
	"""
	Run simulation of the school of fish
	"""
	c = Cardumen(N)
	c.initialize(M,maxVel,maxDist)
	for i in range(nIter):
		print(f"Iteration number: {i}")
		c.printCardumen(True,False)
		c.doStep()
		

run_cardumen()
