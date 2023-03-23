import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main():
	seq_acc_plot()
	char_acc_plot()


def seq_acc_plot():
	nesting_values = list(range(1, 11))
	hybrid_acc = np.array([1.0, 0.89800, 0.85200, 0.83800, 0.80500, 0.74800, 0.74600, 0.69000, 0.63800, 0.63400])*100
	hybrid_std = np.array([0.0, 0.03219, 0.04045, 0.02676, 0.03640, 0.03219, 0.06888, 0.05983, 0.06462, 0.05295])*100
	hybrid_nofilter_acc = np.array([0.99800, 0.87500, 0.83700, 0.81400, 0.80200, 0.73400, 0.72200, 0.59600, 0.48600, 0.40100])*100
	hybrid_nofilter_std = np.array([0.00400, 0.03106, 0.05832, 0.03720, 0.04600, 0.05987, 0.06925, 0.05200, 0.05257, 0.03506])*100
	davinci_acc = np.array([0.98, 0.83, 0.51, 0.37, 0.16, 0.16, 0.05, 0.10, 0.10, 0.09])*100
	davinci_std = np.array([0.04, 0.10, 0.14, 0.12, 0.09, 0.09, 0.07, 0.14, 0.10, 0.05])*100
	transformer_acc = np.array([0.76300, 0.77200, 0.16100, 0.06300, 0.04600, 0.03000, 0.03300, 0.02900, 0.01300, 0.01600])*100
	transformer_std = np.array([0.20120, 0.09442, 0.04549, 0.03100, 0.01428, 0.01183, 0.02002, 0.01578, 0.01187, 0.01744])*100

	ax = sns.lineplot(x=nesting_values, y=hybrid_acc, label='Hybrid')
	ax = sns.lineplot(x=nesting_values, y=hybrid_nofilter_acc, label='Hybrid alt.')
	ax = sns.lineplot(x=nesting_values, y=davinci_acc, ax=ax, label='Davinci')
	ax = sns.lineplot(x=nesting_values, y=transformer_acc, ax=ax, label='End-to-end')
	plt.fill_between(nesting_values, hybrid_acc-hybrid_std, hybrid_acc+hybrid_std, alpha=0.2)
	plt.fill_between(nesting_values, hybrid_nofilter_acc-hybrid_nofilter_std, hybrid_nofilter_acc+hybrid_nofilter_std, alpha=0.2)
	plt.fill_between(nesting_values, davinci_acc-davinci_std, davinci_acc+davinci_std, alpha=0.2)
	plt.fill_between(nesting_values, transformer_acc-transformer_std, transformer_acc+transformer_std, alpha=0.2)
	plt.ylabel('Sequence Accuracy (%)', fontsize=14)
	plt.xlabel('Expression Nesting', fontsize=14)
	plt.savefig('../reports/figures/main_figure.pdf', bbox_inches='tight')
	plt.show()


def char_acc_plot():
	nesting_values = list(range(1, 11))
	hybrid_acc = np.array([1.0, 0.93771, 0.92187, 0.91429, 0.89835, 0.86021, 0.85073, 0.81564, 0.76960, 0.74095])*100
	hybrid_std = np.array([0.0, 0.01937, 0.02683, 0.02488, 0.02940, 0.02918, 0.04749, 0.03955, 0.04778, 0.03624])*100
	davinci_acc = np.array([0.98, 0.89, 0.69, 0.59, 0.41, 0.39, 0.34, 0.35, 0.29, 0.31])*100
	davinci_std = np.array([0.05, 0.08, 0.10, 0.10, 0.10, 0.12, 0.08, 0.14, 0.09, 0.09])*100
	transformer_acc = np.array([0.89121, 0.89740, 0.56026, 0.47437, 0.44861, 0.43321, 0.42213, 0.40757, 0.40305, 0.39717])*100
	transformer_std = np.array([0.10327, 0.04599, 0.03259, 0.02495, 0.02187, 0.01861, 0.02126, 0.02568, 0.01947, 0.01052])*100
	ax = sns.lineplot(x=nesting_values, y=hybrid_acc, label='Hybrid')
	ax = sns.lineplot(x=nesting_values, y=davinci_acc, ax=ax, label='Davinci')
	ax = sns.lineplot(x=nesting_values, y=transformer_acc, ax=ax, label='End-to-end')
	plt.fill_between(nesting_values, hybrid_acc-hybrid_std, hybrid_acc+hybrid_std, alpha=0.2)
	plt.fill_between(nesting_values, davinci_acc-davinci_std, davinci_acc+davinci_std, alpha=0.2)
	plt.fill_between(nesting_values, transformer_acc-transformer_std, transformer_acc+transformer_std, alpha=0.2)
	plt.ylabel('Character Accuracy (%)', fontsize=14)
	plt.xlabel('Expression Nesting', fontsize=14)
	plt.savefig('../reports/figures/characc_figure.pdf', bbox_inches='tight')
	plt.show()


if __name__ == '__main__':
	main()
