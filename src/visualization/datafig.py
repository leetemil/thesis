import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

uniparcfile = Path('data/files/uniparc.txt')
pdbfile = Path('data/files/pdb.txt')

uniparc = np.loadtxt(uniparcfile, skiprows = 1, dtype = str)
uniparc = {int(d.split('-')[0]) : int(n) for d, n in uniparc}

pdb = np.loadtxt(pdbfile, skiprows = 1, dtype = int)
pdb = {year : n for year, n, _ in pdb}

years = range(2000, 2020)

plt.plot(years, [pdb[year] for year in years], label = 'Protein Data Bank')
plt.plot(years, [uniparc[year] for year in years], label = 'UniParc')

plt.title('Protein Database Sizes')
plt.gca().get_xaxis().set_major_locator(matplotlib.ticker.MaxNLocator(integer = True))
plt.xlabel('Year')
plt.ylabel('Size (log scale)')
plt.yscale('log')
plt.legend()
plt.savefig('../report/figures/protein_size.pdf', bbox_inches='tight')