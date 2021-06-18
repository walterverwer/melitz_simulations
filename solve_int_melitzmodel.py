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
phi_upper = 50 # upperbar


### Equilibrium conditions
def equations(p):
    phi_star, phi_star_x, phi_tilde, phi_tilde_x = p
    return (
        phi_tilde - (integrate.quad(lambda x: 1/(phi_upper - phi_star)*x**(sigma-1), phi_star, phi_upper)[0])**(1/(sigma-1)),
        phi_tilde_x - (integrate.quad(lambda x: 1/(phi_upper - phi_star_x)*x**(sigma-1), phi_star_x, phi_upper)[0])**(1/(sigma-1)),
        phi_star_x/phi_star - (f_x/f)**(1/delta)*(tau**((sigma-1)/delta)),
        (1-phi_star/phi_upper)*(f*((phi_tilde/phi_star)**delta)-f)+(1-phi_star_x/phi_upper)*(f_x*((phi_tilde_x/phi_star_x)**delta)-f_x) - theta*f_e
    )

### Solve it
w, x, y, z=  fsolve(equations, [1, 1, 1, 1])

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


#%% Equilibrium with Labor Mobility

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

def eq_labor_mobility(p):
    phi_star, phi_star_x, phi_tilde, phi_tilde_x = p
    return (
        phi_tilde - (integrate.quad(lambda x: 1/(phi_upper - phi_star)*x**(sigma-1), phi_star, phi_upper)[0])**(1/(sigma-1)),
        phi_tilde_x - (integrate.quad(lambda x: 1/(phi_upper - phi_star_x)*x**(sigma-1), phi_star_x, phi_upper)[0])**(1/(sigma-1)),
        (1-phi_star/100)*(1/gamma(phi_star, phi_tilde) - 1)*f + (1-phi_star_x/100)*(1/gamma(phi_star_x, phi_tilde_x) - 1)*f_x - theta*f_e,
        gamma(phi_star_x, phi_star) - tau**(sigma-1)*(f_x/f)
    )

### Solve it
w, x, y, z =  fsolve(eq_labor_mobility, [1, 1, 1, 1])

### What are the parameters
print(eq_labor_mobility((w, x, y, z)))


#%% Equilibrium comparison for tau

# Info: LM := labor mobility case; WO:= without labor mobility

# Parameters (trade costs will vary)
f = 1
f_x = 1
f_e = 2
theta = 0.5
sigma = 2
alpha = 0.8
tau = 1.0
h = 2.0
e = 1.0

# Define varying trade costs parameters
t = np.linspace(1,2.5,50) # tau

# Without Labor Mobility (WO)
WO_phi_star = np.empty(len(t))
WO_phi_star_x = np.empty(len(t))
WO_phi_tilde = np.empty(len(t))
WO_phi_tilde_x = np.empty(len(t))

# With Labor Mobility (LM)
LM_phi_star = np.empty(len(t))
LM_phi_star_x = np.empty(len(t))
LM_phi_tilde = np.empty(len(t))
LM_phi_tilde_x = np.empty(len(t))

# Tau case
count=0
for i in t:
    tau = i
    
    w, x, y, z =  fsolve(eq_labor_mobility, (1,1,1,1))
    a, b, c, d =  fsolve(equations, (1,1,1,1))
    
    LM_phi_star[count] = w
    LM_phi_star_x[count] = x
    LM_phi_tilde[count] = y
    LM_phi_tilde_x[count] = z
    
    WO_phi_star[count] = a
    WO_phi_star_x[count] = b
    WO_phi_tilde[count] = c
    WO_phi_tilde_x[count] = d
    
    count += 1

# Both
fig, axs = plt.subplots(2,2)
axs[0,0].plot(t, LM_phi_star, label='With labor mobility')
axs[0,0].plot(t, WO_phi_star, label='Without labor mobility')
axs[0,0].set_ylabel('$\phi^*$',rotation=0, labelpad=11, fontsize=13)
axs[0,0].set_xlabel('$\\tau$', fontsize=13)
axs[0,0].legend()

axs[0,1].plot(t, LM_phi_star_x)
axs[0,1].plot(t, WO_phi_star_x)
axs[0,1].set_xlabel('$\\tau$', fontsize=13)
axs[0,1].set_ylabel('$\phi^*_x$',rotation=0, labelpad=11, fontsize=13)

axs[1,0].plot(t, LM_phi_tilde)
axs[1,0].plot(t, WO_phi_tilde)
axs[1,0].set_xlabel('$\\tau$', fontsize=13)
axs[1,0].set_ylabel('$\\tilde{\phi}$',rotation=0,  labelpad=11, fontsize=13)

axs[1,1].plot(t, LM_phi_tilde_x)
axs[1,1].plot(t, WO_phi_tilde_x)
axs[1,1].set_xlabel('$\\tau$', fontsize=13)
axs[1,1].set_ylabel('$\\tilde{\phi}_x$',rotation=0, labelpad=11, fontsize=13)
fig.tight_layout()
fig.savefig('comp_tau.pdf')

# Difference:
fig, axs = plt.subplots(2,2)
axs[0,0].plot(t, LM_phi_star - WO_phi_star)
axs[0,0].set_ylabel('$\Delta\phi^*$',rotation=0, labelpad=11, fontsize=13)
axs[0,0].set_xlabel('$\\tau$', fontsize=13)

axs[0,1].plot(t, LM_phi_star_x - WO_phi_star_x)

axs[0,1].set_xlabel('$\\tau$', fontsize=13)
axs[0,1].set_ylabel('$\Delta\phi^*_x$',rotation=0, labelpad=11, fontsize=13)

axs[1,0].plot(t, LM_phi_tilde - WO_phi_tilde)
axs[1,0].set_xlabel('$\\tau$', fontsize=13)
axs[1,0].set_ylabel('$\Delta\\tilde{\phi}$',rotation=0,  labelpad=11, fontsize=13)

axs[1,1].plot(t, LM_phi_tilde_x - WO_phi_tilde_x)
axs[1,1].set_xlabel('$\\tau$', fontsize=13)
axs[1,1].set_ylabel('$\Delta\\tilde{\phi}_x$',rotation=0, labelpad=11, fontsize=13)
fig.tight_layout()
fig.savefig('comp_tau_diff.pdf')


# Hypotheses:
fig, axs = plt.subplots(2,2)
axs[0,0].plot(t, LM_phi_tilde - WO_phi_tilde)
axs[0,0].set_ylabel('$\Delta\\tilde{\phi}$',rotation=0, labelpad=18, fontsize=13)
axs[0,0].set_xlabel('$\\tau$', fontsize=13)
axs[0,0].title.set_text('Hypothesis 1')

axs[0,1].plot(t, LM_phi_tilde_x - WO_phi_tilde_x)
axs[0,1].set_xlabel('$\\tau$', fontsize=13)
axs[0,1].set_ylabel('$\Delta\\tilde{\phi}_x$',rotation=0, labelpad=18, fontsize=13)
axs[0,1].title.set_text('Hypothesis 2')

axs[1,0].plot(t, (LM_phi_star_x / LM_phi_star) - (WO_phi_star_x / WO_phi_star))
axs[1,0].set_xlabel('$\\tau$', fontsize=13)
axs[1,0].set_ylabel('$\Delta\\left(\\frac{\phi^*_x}{\phi^*}\\right)$',rotation=0,  labelpad=18, fontsize=13)
axs[1,0].title.set_text('Hypothesis 3')

axs[1,1].plot(t, (LM_phi_tilde - WO_phi_tilde) - (LM_phi_tilde_x - WO_phi_tilde_x))
axs[1,1].set_xlabel('$\\tau$', fontsize=13)
axs[1,1].set_ylabel('$\Delta\\tilde{\phi}_x$',rotation=0, labelpad=18, fontsize=13)
axs[1,1].title.set_text('Hypothesis 4')

fig.tight_layout()
fig.savefig('hypo.pdf')

#%% Equilibrium comparison for f

# Info: LM := labor mobility case; WO:= without labor mobility

# Parameters (trade costs will vary)
f = 1
f_x = 1
f_e = 2
theta = 0.5
sigma = 2
alpha = 0.8
tau = 1.0
h = 2.0
e = 1.0

# Define varying trade costs parameters
fee = np.linspace(1,2.5,50) # f

# Without Labor Mobility (WO)
WO_phi_star = np.empty(len(t))
WO_phi_star_x = np.empty(len(t))
WO_phi_tilde = np.empty(len(t))
WO_phi_tilde_x = np.empty(len(t))

# With Labor Mobility (LM)
LM_phi_star = np.empty(len(t))
LM_phi_star_x = np.empty(len(t))
LM_phi_tilde = np.empty(len(t))
LM_phi_tilde_x = np.empty(len(t))

# Tau case
count=0
for i in fee:
    f = i
    
    w, x, y, z =  fsolve(eq_labor_mobility, (1,1,1,1))
    a, b, c, d =  fsolve(equations, (1,1,1,1))
    
    LM_phi_star[count] = w
    LM_phi_star_x[count] = x
    LM_phi_tilde[count] = y
    LM_phi_tilde_x[count] = z
    
    WO_phi_star[count] = a
    WO_phi_star_x[count] = b
    WO_phi_tilde[count] = c
    WO_phi_tilde_x[count] = d
    
    count += 1

# Both:
fig, axs = plt.subplots(2,2)
axs[0,0].plot(t, LM_phi_star, label='With labor mobility')
axs[0,0].plot(t, WO_phi_star, label='Without labor mobility')
axs[0,0].set_ylabel('$\phi^*$',rotation=0, labelpad=11, fontsize=13)
axs[0,0].set_xlabel('$f$', fontsize=13)
axs[0,0].legend()

axs[0,1].plot(t, LM_phi_star_x)
axs[0,1].plot(t, WO_phi_star_x)
axs[0,1].set_xlabel('$f$', fontsize=13)
axs[0,1].set_ylabel('$\phi^*_x$',rotation=0, labelpad=11, fontsize=13)

axs[1,0].plot(t, LM_phi_tilde)
axs[1,0].plot(t, WO_phi_tilde)
axs[1,0].set_xlabel('$f$', fontsize=13)
axs[1,0].set_ylabel('$\\tilde{\phi}$',rotation=0,  labelpad=11, fontsize=13)

axs[1,1].plot(t, LM_phi_tilde_x)
axs[1,1].plot(t, WO_phi_tilde_x)
axs[1,1].set_xlabel('$f$', fontsize=13)
axs[1,1].set_ylabel('$\\tilde{\phi}_x$',rotation=0, labelpad=11, fontsize=13)
fig.tight_layout()
fig.savefig('comp_fee.pdf')

# Difference:
fig, axs = plt.subplots(2,2)
axs[0,0].plot(t, LM_phi_star - WO_phi_star)
axs[0,0].set_ylabel('$\Delta\phi^*$',rotation=0, labelpad=11, fontsize=13)
axs[0,0].set_xlabel('$f$', fontsize=13)

axs[0,1].plot(t, LM_phi_star_x - WO_phi_star_x)

axs[0,1].set_xlabel('$f$', fontsize=13)
axs[0,1].set_ylabel('$\Delta\phi^*_x$',rotation=0, labelpad=11, fontsize=13)

axs[1,0].plot(t, LM_phi_tilde - WO_phi_tilde)
axs[1,0].set_xlabel('$f$', fontsize=13)
axs[1,0].set_ylabel('$\Delta\\tilde{\phi}$',rotation=0,  labelpad=11, fontsize=13)

axs[1,1].plot(t, LM_phi_tilde_x - WO_phi_tilde_x)
axs[1,1].set_xlabel('$f$', fontsize=13)
axs[1,1].set_ylabel('$\Delta\\tilde{\phi}_x$',rotation=0, labelpad=11, fontsize=13)
fig.tight_layout()
fig.savefig('comp_fee_diff.pdf')




