"""
Created on Mon Dec 20 12:53:56 2021
@author: khali
"""

import matplotlib.pyplot as plt
from ypstruct import structure
import ga
from cost_function import sphere


# Problem Definition 
problem = structure()
problem.costfunc = sphere
problem.nvar = 5
problem.varmin = -10
problem.varmax = 10

# GA Parameters
params = structure()
params.maxit = 100
params.npop = 50
params.beta = 1
params.pc = 1
params.gamma = 0.1
params.mu = 0.1
params.sigma = 0.1


# Run GA
out = ga.process( problem, params)

# Results
#plt.plot(out.bestcost)
print( out.bestsol )
plt.semilogy(out.bestcost)
plt.xlim(0, params.maxit)
plt.xlabel('Iterations')
plt.ylabel('Best cost')
plt.title('Genetic Algorithm (GA) ' )
plt.grid(True)
plt.show()














