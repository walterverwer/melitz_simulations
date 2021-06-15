from scipy.optimize import fsolve
import scipy.integrate as integrate
import math
import numpy as np
import matplotlib.pyplot as plt


## Define all the parameters
### Free trade with limited mobility l_E = 0

### Parameter setup
f = 1
f_x = 1
f_e = 10
theta = 0.5
sigma = 2
alpha = 0.8
tau = 2.0

delta = ((sigma*(1-alpha))/(sigma*(1-alpha)+alpha))*(-1/sigma)

## uniform distribution parameters (0,100)
phi_upper = 100 # upperbar
def mu(phi_upper):
    return(1/phi_upper)# density function

### Equilibrium conditions
def equations(p):
    phi_star, phi_star_x, phi_tilde, phi_tilde_x = p
    return (
        phi_tilde - (integrate.quad(lambda x: mu(x)*x**(sigma-1), phi_star, phi_upper)[0])**(1/(sigma-1)),
        phi_tilde_x - (integrate.quad(lambda x: mu(x)*x**(sigma-1), phi_star_x, phi_upper)[0])**(1/(sigma-1)),
        phi_star_x/phi_star - (f_x/f)**(1/(delta*(1-sigma)))*(tau**(-1/delta)),
        (1-phi_star/100)*(f*((phi_tilde/phi_star)**delta)-f)+(1-phi_star_x/100)*(f_x*((phi_tilde_x/phi_star_x)**delta)-f_x) - theta*f_e
    )

### Solve it
w, x, y, z =  fsolve(equations, [1, 1, 1, 1])

### What are the parameters
print(equations((w, x, y, z)))


#%% Simulation for phi_star_x based on tau

t = np.linspace(1,2,10)


threshold_star_x = np.empty(len(t))
count=0
for i in t:
    f = 1
    f_x = 1
    f_e = 10
    theta = 0.5
    sigma = 2
    alpha = 0.8
    tau = i
    w, x, y, z =  fsolve(equations, (1,1,1,1))
    threshold_star_x[count] = x
    count += 1
    
plt.plot(t,threshold_star_x)
plt.ylabel('$\\phi^*_x$')
plt.xlabel('$\\tau$')


#%% Equilibrium Labor Mobility










