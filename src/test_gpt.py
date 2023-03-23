import hydra
import omegaconf
import openai
import os
import time
import warnings
import json
import torch
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from data.generator import LTEStepsGenerator
from model.test import batch_acc, batch_seq_acc, _fix_output_shape
from visualization.test_ood import build_generator


@hydra.main(config_path="../conf", config_name="gpt", version_base='1.2')
def main(cfg):
	openai.api_key = os.getenv("OPENAI_API_KEY")
	lte, lte_kwargs = build_generator(cfg)
	ax, df = test_ood(lte, lte_kwargs, cfg, min_nesting=cfg.min_nesting, max_nesting=cfg.max_nesting, num_samples=cfg.num_samples)
	plt.savefig(os.path.join(hydra.utils.get_original_cwd(),
		f"../reports/figures/{cfg.model_name}_start2end.pdf"))
	df = df.set_index('Nesting')
	df = np.round(df, 5)
	df.T.to_latex(os.path.join(hydra.utils.get_original_cwd(),
		f"../reports/tables/{cfg.model_name}_start2end.tex"))


def test_ood(generator, generator_kwargs, cfg, min_nesting=1, max_nesting=10, num_samples=10):
	accuracy_values = []
	accuracy_std_values = []
	seq_acc_values = []
	seq_acc_std_values = []
	nesting_values = []
	WAIT = 1 if cfg.model_name == 'text-davinci-003' else 3

	for nes_value in range(min_nesting, max_nesting+1):
		same_nes_acc, same_nes_seq_acc = np.zeros(num_samples), np.zeros(num_samples)

		for sample_idx in range(num_samples):
			X, Y, lenX, lenY, mask = generator.generate_batch(1, nes_value, **generator_kwargs)
			lenY = torch.tensor(lenY, device=X.device)
			prompts = build_prompts(X, generator, generator_kwargs)
			answers = []

			for prompt in prompts:
				answer = ''
				while answer == '':
					try:
						answer = make_request(prompt, cfg)
					except (openai.error.RateLimitError, openai.error.APIError) as e:
						warnings.warn(str(e))
						time.sleep(WAIT)
				if answer != '#':
					answers.append(list(answer) + ['.'])
				else:
					answers.append(list(answer))

			output = generator._build_batch(answers, y=True)
			if output.size() != Y[:, 1:].size():
				warn_str = f"Outputs shape {output.size()} different from targets shape {Y[:, 1:].size()}. Fixing."
				warnings.warn(warn_str)
				print(generator.y_to_str(Y))
				print(generator.y_to_str(output))
				output = _fix_output_shape(output, Y[:, 1:], generator)

			acc_avg, acc_std = batch_acc(output, Y[:, 1:], Y.size(-1), generator)
			seq_acc_avg, seq_acc_std = batch_seq_acc(output, Y[:, 1:], generator, lenY)
			same_nes_acc[sample_idx] = acc_avg.item()
			same_nes_seq_acc[sample_idx] = seq_acc_avg.item()

		accuracy_values += [same_nes_acc.mean()]
		seq_acc_values += [same_nes_seq_acc.mean()]
		accuracy_std_values += [same_nes_acc.std()]
		seq_acc_std_values += [same_nes_seq_acc.std()]
		nesting_values += [nes_value]

	df = pd.DataFrame()
	df['Character Accuracy'] = accuracy_values
	df['Character Accuracy Std'] = accuracy_std_values
	df['Sequence Accuracy'] = seq_acc_values
	df['Sequence Accuracy Std'] = seq_acc_std_values
	df['Nesting'] = nesting_values

	ax = sns.barplot(data=df, x='Nesting', y='Character Accuracy', color='tab:blue')
	return ax, df


def make_request(prompt, cfg):
	if cfg.model_kind == 'completion':
		res = openai.Completion.create(
			model=cfg.model_name,
			prompt=prompt,
			stop='<END>',
		)
		answer = res['choices'][0]['text'].strip()
		answer = preprocess_output_davinci(answer)
	elif cfg.model_kind == 'chat':
		res = openai.ChatCompletion.create(
			model=cfg.model_name,
			messages=[{"role": "user", "content": prompt}],
		)
		answer = res['choices'][0]['message']['content'].strip()
		answer = preprocess_output_gpt_turbo(answer)
	else:
		raise ValueError(f'Wrong model kind {cfg.model_kind}.')
	
	# save model output
	with open(os.path.join(hydra.utils.get_original_cwd(), f"../chatgpt/res/{cfg.run_dt}_{cfg.model_name}.txt"), 'a') as res_f:
		res_f.write(json.dumps(res)+'\n')
	with open(os.path.join(hydra.utils.get_original_cwd(), f"../chatgpt/ans/{cfg.run_dt}_{cfg.model_name}.txt"), 'a') as ans_f:
		ans_f.write(prompt+'\n')
		ans_f.write(answer+'\n')

	return answer


def build_prompts(X, generator, generator_kwargs):
	generator_kwargs['split'] = 'valid'
	X_examples, Y_examples, _, _, _ = generator.generate_batch(1, 3, **generator_kwargs)
	generator_kwargs['split'] = 'test'
	X, X_examples, Y_examples = generator.x_to_str(X), generator.x_to_str(X_examples), generator.y_to_str(Y_examples)
	examples = [x.replace('#', '') + ' = ' + y.replace('#', '').replace('?', '').replace('.', '') + ' <END>' for x, y in zip(X_examples, Y_examples)]
	queries = [x.replace('#', '') + ' = ' for x in X]
	return ['\n'.join([e, q]) for e, q in zip(examples, queries)]


def preprocess_output_davinci(output):
	number_re = re.compile(r'^-?\d\d?$')
	if '=' in output:
		res = output.split('=')[-1]  # take the right hand side of the equality
		if number_re.match(res.strip()) is not None:  # check it is a number
			return res.strip()
		else:
			return '#'
	elif number_re.match(output) is not None:
		return output
	else:
		return '#'


def preprocess_output_gpt_turbo(output):
	raise NotImplementedError("Not implemented.")


if __name__ == '__main__':
	main()
