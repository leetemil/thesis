import matplotlib
import matplotlib.pyplot as plt
import numpy as np

MEP = False

font_size = 11

if MEP:
    labels = ['BLAT', 'CALM', 'GAL4', 'HSP82', 'RASH']
    unirep = [0.50, 0.30, 0.54, 0.45, 0.35]
    vae = [0.77, 0.27, 0.63, 0.55, 0.48]
    wavenet = [0.61, 0.29, 0.54, 0.51, 0.43]

else:
    labels = ['Secondary Structure', 'Remote Homology', 'Fluorescence', 'Stability']
    unirep = [0.68, 0.12, 0.67, 0.74]
    wavenet = [0.38, 0.17, 0.47, 0.58]

x = np.arange(len(labels))  # the label locations
width = 0.20 if MEP else 0.35

fig, ax = plt.subplots(figsize=[10,5])

scale = 1 if MEP else 0.5

rects1 = ax.bar(x - scale * width, unirep, width, label='UniRep')#, color='gray')
rects3 = ax.bar(x + scale * width, wavenet, width, label='WaveNet')#, color='royalblue')

if MEP:
    rects2 = ax.bar(x, vae, width, label='VAE')#, color='green')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Spearman\'s $\\rho$' if MEP else 'Score', fontsize = font_size)
ax.set_ylim(0,1)
# ax.set_title('')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize = font_size)
ax.legend(fontsize = font_size)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize = font_size)

autolabel(rects1)
autolabel(rects3)

if MEP:
    autolabel(rects2)

fig.tight_layout()
name = 'MEP' if MEP else 'TAPE'
plt.savefig(f'../report/figures/{name}_barchart.pdf', bbox_inces='tight')

# plt.show()
