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
f_e = 2
theta = 0.5
sigma = 2
alpha = 0.7
tau = 2.0

delta = (1-sigma)/(sigma*(alpha-1)-alpha)

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
        phi_star_x/phi_star - (f_x/f)**(1/delta)*(tau**((sigma-1)/delta)),
        (1-phi_star/100)*(f*((phi_tilde/phi_star)**delta)-f)+(1-phi_star_x/100)*(f_x*((phi_tilde_x/phi_star_x)**delta)-f_x) - theta*f_e
    )

### Solve it
w, x, y, z =  fsolve(equations, [1,1,1,1])

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

### Parameter setup
f = 1
f_x = 1
f_e = 2
theta = 0.5
sigma = 2
alpha = 0.8
tau = 2.0
h = 2.0
e = 1.0

def gamma(a,b):
    x = (h/a)*(((1-alpha)/alpha))*(h/e)**(1-alpha) + (e/a)*(((1-alpha)/alpha))*(h/e)**alpha
    y = (h/b)*(((1-alpha)/alpha))*(h/e)**(1-alpha) + (e/b)*(((1-alpha)/alpha))*(h/e)**alpha
    return (x/y)**(1-sigma)

x = gamma(1.0,2.0)

def eq_labor_mobility(p):
    phi_star, phi_star_x, phi_tilde, phi_tilde_x = p
    return (
        phi_tilde - (integrate.quad(lambda x: mu(x)*x**(sigma-1), phi_star, phi_upper)[0])**(1/(sigma-1)),
        phi_tilde_x - (integrate.quad(lambda x: mu(x)*x**(sigma-1), phi_star_x, phi_upper)[0])**(1/(sigma-1)),
        (1-phi_star/100)*(1/gamma(phi_star, phi_tilde) - 1)*f + (1-phi_star_x/100)*(1/gamma(phi_star_x, phi_tilde_x) - 1)*f_x - theta*f_e,
        gamma(phi_star_x, phi_star) - tau**(sigma-1)*(f_x/f)
    )

### Solve it
w, x, y, z =  fsolve(eq_labor_mobility, [1, 1, 1, 1])

### What are the parameters
print(eq_labor_mobility((w, x, y, z)))


### Plot range of tau with exporting threshold
t = np.linspace(1,2.5,50)
op_phi_star = np.empty(len(t))
op_phi_star_x = np.empty(len(t))
op_phi_tilde = np.empty(len(t))
op_phi_tilde_x = np.empty(len(t))
count=0
for i in t:
    f = 1
    f_x = 1
    f_e = 2
    theta = 0.5
    sigma = 2
    alpha = 0.8
    tau = i
    h = 2.0
    e = 1.0
    w, x, y, z =  fsolve(eq_labor_mobility, (1,1,1,1))
    op_phi_star[count] = w
    op_phi_star_x[count] = x
    op_phi_tilde[count] = y
    op_phi_tilde_x[count] = z
    count += 1
    
fig, axs = plt.subplots(2,2)
axs[0,0].plot(t, op_phi_star)
axs[0,0].set_ylabel('$\phi^*$',rotation=0, labelpad=11, fontsize=13)
axs[0,0].set_xlabel('$\\tau$', fontsize=13)

axs[0,1].plot(t, op_phi_star_x)
axs[0,1].set_xlabel('$\\tau$', fontsize=13)
axs[0,1].set_ylabel('$\phi^*_x$',rotation=0, labelpad=11, fontsize=13)

axs[1,0].plot(t, op_phi_tilde)
axs[1,0].set_xlabel('$\\tau$', fontsize=13)
axs[1,0].set_ylabel('$\\tilde{\phi}$',rotation=0,  labelpad=11, fontsize=13)

axs[1,1].plot(t, op_phi_tilde_x)
axs[1,1].set_xlabel('$\\tau$', fontsize=13)
axs[1,1].set_ylabel('$\\tilde{\phi}_x$',rotation=0, labelpad=11, fontsize=13)
fig.tight_layout()
fig.savefig('labor_mobility_sim.pdf')




