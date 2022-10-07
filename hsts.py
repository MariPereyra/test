import numpy as np
from scipy.stats import gamma as g
from matplotlib import pyplot as plt
from scipy.fft import fft, ifft, fftfreq

# %%

g1 = g(a=3, loc=0, scale=0.8)
g2 = g(a=2, loc=0.5, scale=0.05)

x = np.linspace(0, 20, 1000)

fig, ax = plt.subplots(2, 2)

ax[0,0].plot(x, g1.pdf(x))
ax[0,0].plot(x, g2.pdf(x))
ax[0,0].plot(x, g1.pdf(x) + g2.pdf(x))

r1 = g1.rvs(size=500)
r2 = g2.rvs(size=100)
r = np.concatenate([r1, r2])

ax[0, 1].hist(r, bins=20)

ax[1, 0].hist(r, bins=12)
ax[1, 1].hist(r, bins=120)

# %% Explorar diferente nro. de bines

fig, ax = plt.subplots(1, 4)

N = 300
N1 = int(N*0.8)
N2 = N - N1
r1 = g1.rvs(size=N1)
r2 = g2.rvs(size=N2)
r = np.concatenate([r1, r2])

ax[0].hist(r, bins=12)
ax[1].hist(r, bins=20)
ax[2].hist(r, bins=60)
ax[3].hist(r, bins=120)

# %% Explorar diferente nro. de bines

fig, ax = plt.subplots(1, 4)

N = 1000
N1 = int(N*0.8)
N2 = N - N1
r1 = g1.rvs(size=N1)
r2 = g2.rvs(size=N2)
r = np.concatenate([r1, r2])

ax[0].hist(r, bins=12)
ax[1].hist(r, bins=20)
ax[2].hist(r, bins=60)
ax[3].hist(r, bins=120)

# %% Metodo de Knuth

from astropy.stats import knuth_bin_width as kn

dx, edges = kn(r, return_bins=True)
plt.hist(r, bins=edges)


# %% Metodo de Scargle

from astropy.stats import bayesian_blocks as bb

edges = bb(r, fitness='events', p0=0.01)
plt.hist(r, bins=edges)

# %% Metodo de Scargle

N = 500
r = np.concatenate([g1.rvs(size=int(N*0.8)), g2.rvs(size=int(N*0.2))])
r.sort()
g = lambda x: 0.8*g1.pdf(x) + 0.2*g2.pdf(x)

ecdf = np.linspace(0, 1, N)
ref = (r-r.min())/(r.max()-r.min())

fig = plt.figure(dpi=150, figsize=(6, 12))
ax = fig.add_subplot(411)

ax.plot(r, g(r))
ax.hist(r, bins=100, density=True)

ax = fig.add_subplot(412)
ax.plot(r, ecdf)
ax.plot(r, ref)

ax = fig.add_subplot(413)
res = ecdf-ref

n = 12  # number of frequencies
cols = []
f = np.pi / max(r)
for i in range(1, n+1):
    cols.append(np.sin(f*i*r))
M = np.column_stack(cols)
A = M.transpose()
pars = np.linalg.solve(A@M, A@res)

ax.plot(r, res, linewidth=4, alpha=0.5, color='coral')
model = np.repeat(0, len(r))
for i, a in enumerate(pars):
    mi = a*np.sin(f*(i+1)*r)
    model = model + mi

ax.plot(r, model, color='k', label='fit')
ax.grid()


ax = fig.add_subplot(414)

dmodel = np.repeat(0, len(r))
for i, a in enumerate(pars):
    mi = a*np.cos(f*(i+1)*r)*f*(i+1)
    dmodel = dmodel + mi
ax.plot(r, dmodel)
ax.plot(r, g(r))

plt.tight_layout()

# %%


