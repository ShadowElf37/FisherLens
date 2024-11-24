import numpy as np

new = np.array([[1.836214287890062e-27, 2.0518390683079987e-26, 3.1112206655907685e-26], [3.2010521914769854e-27, 2.806166040585091e-25, 1.7774326767821104e-23], [3.66129331573554e-27, 3.1693518560830103e-24, 1.5912332246267115e-21], [7.580515107049344e-27, 3.7152765357903646e-23, 4.1822914003989664e-20], [8.369979060016429e-27, 4.245646772447925e-22, 4.498792369870086e-18], [8.670939028540317e-27, 4.8626892442488664e-21, 1.3491303574151182e-16], [1.2228485748278109e-26, 5.0860873379914125e-20, 5.130687432907451e-16], [7.012118140872936e-26, 5.482189222055548e-19, 2.5750864610618834e-14], [6.506048589094239e-25, 5.599551426166845e-18, 2.433412001132613e-13], [6.451914578768011e-24, 5.614395849537843e-17, 2.7884056915156963e-12]])

m = np.array(tuple(range(-6, 4)))
n = (0,2,4)

boddy_m = np.array([-6, -5, -4, -3, -2, 0, 3])
#print(np.equal(m, boddy_m))
new[np.logical_not(np.logical_or(np.logical_or(m < -1, m == 0), m == 3))] = float('nan')



old = np.array([
    [1.1e-27, 1.1e-25, 2.7e-24],
    [2.1e-27, 9.9e-25, 3.3e-22],
    [4.4e-27, 8.6e-24, 5.2e-20],
    [1.2e-26, 1.9e-22, 9.5e-19],
    [6.5e-26, 2.0e-21, 1.0e-17],
    [float('nan')]*3,
    [8.4e-24, 1.5e-19, 9.6e-16],
    [float('nan')]*3,
    [float('nan')]*3,
    [8.9e-21, 1.4e-16, 1.0e-12],
])

# this is from milky way satellites oops
"""old = np.array(
[
[5.7e-33, 4.2e-36, 1.6e-39],
[4.5e-32, 4.2e-33, 1.6e-34],
[3.6e-31, 3.9e-30, 1.2e-29],
[4.5e-30, 6.2e-28, 1.5e-25],
[7.3e-29, 4.9e-26, 3.0e-23],
[7.3e-28, 4.9e-25, 3.9e-22],
[7.3e-27, 6.2e-24, 3.9e-21],
[7.3e-26, 6.2e-23, 3.9e-20],
[7.3e-25, 6.2e-22, 3.9e-19],
[7.3e-24, 6.2e-21, 3.9e-18],
]
)"""

diff = np.log10(old/new)

import matplotlib.pyplot as plot
import matplotlib.cm as cm
import matplotlib.colors as colors


c = colors.LinearSegmentedColormap.from_list("", ["#f00","#fff","#aff","#5ff","#0ff"])
c.set_bad('black')

fig, ax = plot.subplots(figsize=(5,6))
graph = ax.imshow(diff, extent=(-1, 5, boddy_m.max()+0.5, boddy_m.min()-0.5), interpolation='none', cmap=c, vmin=-1, vmax=3)

ax.set_xlabel('Power Law $n$')
ax.set_ylabel('Mass (log$_{10}$GeV)')
ax.set_xticks(n)
ax.set_yticks(m)
for (j,i),label in np.ndenumerate(diff):
    if label == float('nan'): continue
    ax.text(n[i],m[j],round(label, 1),ha='center',va='center')
cbar = fig.colorbar(graph)
cbar.set_label('Orders of Magnitude Improvement')
ax.set_title('CMB-S4 Improvement over Planck')

plot.savefig('sigma_improvement.png')

plot.show()