# import libraries from parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import *
from scipy.stats import binom
e = exp_from_arguments()        # get arguments

def p(x,y):
    return x*y + (1-x)*(1-y)

# parameters
n = 10                  # total number of participants
k = n/2                 # half of participants

x = 0.9                         # pred. acc
ys = np.linspace(0,1,num=n+1)   # prob. honest

qs = np.linspace(0,1,num=101)   # prior for innocent players

pr_ally = (k-1)/(n-1)
pr_enemy = k/(n-1)

for q in qs:
    x = 0.9
    z0 = [binom.cdf(k-1, n-1, 1-x)]*len(ys)
    z1, z2, z3, z4, z5 = [], [], [], [], []
    for y in ys:
        # dishonest (flip)
        p1 = p(x,y)
        z1.append(binom.cdf(k-1, n-1, 1-p1))

        # dishonest (random)
        p2 = x*y + 0.5*(1-y)
        z2.append(binom.cdf(k-1, n-1, 1-p2))

        # dishonest-but-rational (all)
        c = int(y*n)/2      # number of honest players in each team
        if c >= k-1:
            p3_a = x
        else:
            p3_a = (c/(k-1))*x + ((k-1-c)/(k-1))*(x*y + q*(1-y))
        p3_e = (c/k)*x + ((k-c)/k)*(x*y + (1-q)*(1-y))
        p3 = pr_ally*p3_a + pr_enemy*p3_e

        z3.append(binom.cdf(k-1, n-1, 1-p3))

        # dishonest-but-rational (ally)
        p4_a = p3_a
        p4_e = x
        p4 = pr_ally*p4_a + pr_enemy*p4_e
        z4.append(binom.cdf(k-1, n-1, 1-p4))

    if q == 1.0:
        plt.plot(ys,z0[::-1],'--',label=f'no dishonest player',color='black',alpha=0.3,linewidth=5)
        plt.plot(ys,z1[::-1],label='dishonest (flip)',color='blue',linewidth=1.5,alpha=q)
        plt.plot(ys,z2[::-1],label='dishonest (random)',color='orange',linewidth=1.5,alpha=q)
        plt.plot(ys,z3[::-1],label='dishonest-but-rational (all)',color='green',linewidth=1,alpha=q)
        plt.plot(ys,z4[::-1],label='dishonest-but-rational (ally)',color='red',linewidth=1,alpha=q)

    else:
        # first two won't change
        #plt.plot(ys,z1[::-1],color='blue',linewidth=1,alpha=q/2)
        #plt.plot(ys,z2[::-1],color='orange',linewidth=1,alpha=q/2)
        plt.plot(ys,z3[::-1],color='green',linewidth=1,alpha=q/2)
        plt.plot(ys,z4[::-1],color='red',linewidth=1,alpha=q/2)

plt.xlabel(r'Portion of dishonest player ($p_{atk}$)')
plt.ylabel(r'Expected correctness ($\mathbb{E}[C]$)')
plt.legend()
plt.savefig('figures/fig_04_num.pdf')