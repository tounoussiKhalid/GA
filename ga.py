"""
Created on Mon Dec 20 12:59:29 2021
@author: khali
"""
from ypstruct import structure
import numpy as np


def process( problem, params):
    
    # Problem Information
    costfunc = problem.costfunc
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax
    
    # Parameters 
    maxit = params.maxit
    npop = params.npop # number of population
    beta = params.beta
    pc = params.pc # propotion of childreen in main population
    nc   = int( np.round(pc * npop/2 ) *2 ) # number of childreen 
    gamma = params.gamma
    mu = params.mu
    sigma = params.sigma

    
    # individual
    individual = structure()
    individual.position = None
    individual.cost = None
    
    # Track Best Solution ever found
    bestsol = individual.deepcopy()
    bestsol.cost = np.inf
    
    # Initialize population
    pop = individual.repeat(npop)
    for i in range( 0, npop ):
        pop[i].position = np.random.uniform(varmin, varmax, nvar)
        pop[i].cost = costfunc( pop[i].position )
        if pop[i].cost < bestsol.cost:
            bestsol = pop[i].deepcopy()
        
    # to store best cost during iterations
    bestcost = np.empty(maxit)
    
    # Main Loop
    for it in range(maxit ):
        
        costs = np.array( [x.cost for x in pop])
        avg_cost = np.mean(costs)
        if avg_cost != 0:
            costs= costs/avg_cost
        probs = np.exp(-beta*costs)     
        
        popc = []
        for k in range(nc//2):
            
            # Select parents
            ## Random Selection
            #q = np.random.permutation(npop)
            #p1 = pop[q[0]]
            #p2 = pop[q[1]]
            
            ## Roulette Selection
            p1 = pop[roulette_wheel_selection(probs)]
            p2 = pop[roulette_wheel_selection(probs)]

            
            # Perform Crossover
            c1, c2 = crossover( p1, p2, gamma)
            
            # Perform Mutation
            c1 = mutate(c1 , mu, sigma)
            c2 = mutate(c2 , mu, sigma)
            
            # Apply Bounds
            apply_bound(c1, varmin, varmax)
            apply_bound(c2, varmin, varmax)
            
            # Evaluation First offspring
            c1.cost = costfunc( c1.position )
            if c1.cost < bestsol.cost:
                bestsol = c1.deepcopy()
                
            # Evaluation Second offspring
            c1.cost = costfunc( c1.position )
            if c1.cost < bestsol.cost:
                bestsol = c1.deepcopy()

            # Add Offsprings to popc
            popc.append(c1)
            popc.append(c2)
        
        # Merge, Sort and Select
        pop += popc
        pop = sorted( pop, key= lambda x: x.cost )
        pop = pop[0 : npop]

        #Store best cost
        bestcost[it] = bestsol.cost
            
        # display iteration information
        print( 'Iteration= {} .Best Cost={}. '.format( it, bestcost[it] ))


    # Output

    out = structure()
    out.pop = pop
    out.bestsol = bestsol
    out.bestcost = bestcost
    return out

# uniform crossover
# x1 =(x11, x12, ..., x1n)
# x2 = (x21, x22, ..., x2n)
# alpha = (aplha1, alpha2, ..., alphan)
# aplha[i] inside [ 0 , 1] or generally [ -gamma, 1 + gamma]
# y1i = aplha(i) * x1i + ( 1-alpha(i) )*x2i
# y2i = aplha(i) * x2i + ( 1-alpha(i) )*x1i
def crossover(p1, p2, gamma = 0.1):
    c1 = p1.deepcopy()
    c2 = p2.deepcopy()
    alpha = np.random.uniform( -gamma, 1 + gamma, *c1.position.shape)
    c1.position = alpha*p1.position + (1-alpha)*p2.position
    c2.position = alpha*p2.position + (1-alpha)*p1.position
    return c1,c2

# mu : mutation rate( percentage), sigma : standard deviation
def mutate( x, mu, sigma):
    y = x.deepcopy()
    flag = np.random.rand(*x.position.shape) <= mu
    ind = np.argwhere(flag)
    y.position[ind] += sigma * np.random.randn(*ind.shape)
    return y
    
def apply_bound( x, varmin, varmax ):
    x.position = np.maximum(x.position, varmin)
    x.position = np.minimum(x.position, varmax)

def roulette_wheel_selection( p ):
    c = np.cumsum(p)
    r = sum(p)*np.random.rand()
    ind = np.argwhere( r <= c)
    return ind[0][0]
    
  


# some tests 
"""
print( np.random.permutation(5))
s = np.zeros((1,5))
print(s)
t = np.random.rand(1,5)
flag = ( t<= 0.2)
print( np.argwhere( flag))
print( t)

arr = np.random.randn(1,5)
print ( arr )
print( arr * 0.1)
"""

"""
test example for roulette wheel
"""
p = np.array([1,2,3,4])
cumsum = np.cumsum(p)
ind = np.argwhere( np.random.rand()*sum(p) <= np.cumsum( p ) )
print( ind[0][0] )





