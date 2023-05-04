import hydra
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime as dt
import os
from data.generator import LTEGenerator, LTEStepsGenerator
from model.Transformer import Transformer
from model.SolverCombiner import SolverCombiner
from model.test import batch_acc, batch_seq_acc, _fix_output_shape
from test_ood import build_generator, load_model
import warnings
import logging


@hydra.main(config_path="../conf", config_name="test_start2end", version_base='1.2')
def main(cfg):
	warnings.filterwarnings('always', category=UserWarning)
	now_day = dt.now().strftime('%Y-%m-%d')
	now_time = dt.now().strftime('%H:%M')
	logging.basicConfig(filename=os.path.join(hydra.utils.get_original_cwd(), f'../logs/{now_day}_{now_time}_{cfg.ckpt[:-4]}_test_start2end.txt'),
			filemode='a',
			format='%(message)s',
			datefmt='%H:%M:%S',
			level=logging.INFO)
	lte, lte_kwargs = build_generator(cfg)
	model = load_model(cfg, lte)
	if cfg.multi or cfg.multi_nofilter:
		model = SolverCombiner(model, cfg)
	n_samples = f'_{str(cfg.n_samples)}-samples'
	no_filter = '_nofilter' if cfg.multi_nofilter else ''

	ax, df = test_ood_start2end(model, lte, cfg.max_nes, generator_kwargs={'batch_size': cfg.bs,
																 'start_to_end': cfg.start_to_end,
																 'filtered_s2e': cfg.filtered_s2e,
																 'split': 'test',
																 'ops': cfg.ops})
	plt.savefig(os.path.join(hydra.utils.get_original_cwd(),
		f"../reports/figures/{cfg.ckpt[:-4]}_start2end{n_samples}{no_filter}_nes{cfg.max_nes}.pdf"))
	df = df.set_index('Nesting')
	df = np.round(df, 5)
	df.T.to_latex(os.path.join(hydra.utils.get_original_cwd(),
		f"../reports/tables/{cfg.ckpt[:-4]}_start2end{n_samples}{no_filter}_nes{cfg.max_nes}.tex"))


def test_ood_start2end(model, generator, max_nes, num_samples=10, tf=False, generator_kwargs=None, plot_ax=None, plot_label=None):
	accuracy_values = []
	accuracy_std_values = []
	seq_acc_values = []
	seq_acc_std_values = []
	nesting_values = []
	avg_halting = []
	std_halting = []

	for n in range(1, max_nes+1):
		logging.info(f"\n--- nesting {n} ---")
		same_nes_acc, same_nes_seq_acc, same_nes_halting = np.zeros(num_samples), np.zeros(num_samples), np.zeros(num_samples)

		for sample_idx in range(num_samples):
			values = generator.generate_batch(1, n, **generator_kwargs)

			if isinstance(generator, LTEStepsGenerator):
				X, Y, lenX, lenY, mask = values
			else:
				X, Y, lenX, lenY = values
			lenY = torch.tensor(lenY, device=X.device)

			with torch.no_grad():
				if isinstance(model, Transformer):
					output = model(X, Y=None, tf=tf)
					running = torch.tensor([True]*X.size(0))
				else:
					output = model(X, Y, tf=tf, max_nes=n)
					running = model.running[-1]
			
			if running.any():
				if ~running.any():
					output[~running] = generator._build_batch([['#']*output.size(1)]*(~running).sum(), y=True)
				if output.size() != Y[:, 1:].size():
					warn_str = f"Outputs shape {output.size()} different from targets shape {Y[:, 1:].size()}. Fixing."
					logging.info(warn_str)
					warnings.warn(warn_str)
					output = _fix_output_shape(output, Y[:, 1:], generator)

				logging.info("Output")
				logging.info(generator.y_to_str(output[:20]))
				logging.info("Target")
				logging.info(generator.y_to_str(Y[:, 1:][:20]))

				acc_avg, acc_std = batch_acc(output, Y[:, 1:], Y.size(-1), generator)
				seq_acc_avg, seq_acc_std = batch_seq_acc(output, Y[:, 1:], generator, lenY)
				same_nes_acc[sample_idx] = acc_avg.item()
				same_nes_seq_acc[sample_idx] = seq_acc_avg.item()
				same_nes_halting[sample_idx] = (~running).sum()
			else:
				same_nes_acc[sample_idx] = 0
				same_nes_seq_acc[sample_idx] = 0
		accuracy_values += [same_nes_acc.mean()]
		seq_acc_values += [same_nes_seq_acc.mean()]
		accuracy_std_values += [same_nes_acc.std()]
		seq_acc_std_values += [same_nes_seq_acc.std()]
		avg_halting += [same_nes_halting.mean()]
		std_halting += [same_nes_halting.std()]
		nesting_values += [n]

	df = pd.DataFrame()
	df['Character Accuracy'] = accuracy_values
	df['Character Accuracy Std'] = accuracy_std_values
	df['Sequence Accuracy'] = seq_acc_values
	df['Sequence Accuracy Std'] = seq_acc_std_values
	df['Avg Halting'] = avg_halting
	df['Std Halting'] = std_halting
	df['Nesting'] = nesting_values

	ax = sns.barplot(data=df, x='Nesting', y='Character Accuracy', label=plot_label, ax=plot_ax, color='tab:blue')
	if isinstance(model, SolverCombiner):
		ax = sns.lineplot(x=range(max_nes), y=[s/generator_kwargs['batch_size'] for s in avg_halting], marker='o', color='tab:cyan')
	return ax, df


if __name__ == "__main__":
	main()
