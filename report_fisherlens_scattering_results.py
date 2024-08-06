import numpy as np
from generate_sigma_for_paper import step_table, STEP_TABLE_OFFSET

f_sky = 0.4
TAU_PRIOR = True

covs = []

from math import floor, log10

def fexp(f):
    return int(floor(log10(abs(f)))) if f != 0 else 0
def fman(f):
    return f/10**fexp(f)
def float_to_latex(f):
    return fr"{round(fman(f), 1)} \times 10^{{{fexp(f)}}}"

data = [["" for _ in range(3)] for _ in range(10)]
#print(data)
for mi, m in enumerate(range(-6, 4)):
    for ni, n in enumerate((0,2,4)):
        try:
            fn = f'CLASS_delens/results/scatter_m{m}_n{n}_s{step_table[(m,n)]+STEP_TABLE_OFFSET}.0.pkl'
            #print(fn)
            fish = np.load(fn, allow_pickle=True)['fisherGaussian']['delensed']
            if TAU_PRIOR:
                fish[4, :] = 0
                fish[:, 4] = 0
                fish[4, 4] = 1 / 0.0074 ** 2
            cov = np.linalg.inv(fish) / f_sky
            data[mi][ni] = float_to_latex(np.sqrt(cov[-1, -1]) * 2)
            #print(mi, ni, m, n, data[mi][ni])
            #print(f'm = 10^{m} GeV\tn = {n}\tσ <=', float_to_latex(, 'cm^2')
            covs.append(cov)
        except:
            print(f'Skipping m = 10^{m} GeV, n = {n}')

print(data)

#print(data)
def make_table(row_names, col_names, data_2d_array):
    tab = r"""\begin{table}[]
\begin{tabular}{"""
    tab += '|'.join('c' for _ in col_names)
    tab += '}\n'
    tab += ' & '.join(map(lambda s: '$' + s + '$', col_names))
    tab += r' \\ \hline'
    tab += '\n'
    for i,row in enumerate(row_names):
        tab += '$'+row+'$' + ' & '
        #print(data_2d_array[i])
        tab += ' & '.join(map(lambda s: '$'+s+'$', data_2d_array[i]))
        if i != len(row_names)-1:
            tab += r' \\'
        tab += '\n'
    tab += """\end{tabular}
\end{table}"""

    return tab

print(make_table(
    [f"10^{{{i}}}" for i in range(-6, 4)],
    ["", "n=0", "n=2", "n=4"],
    data))