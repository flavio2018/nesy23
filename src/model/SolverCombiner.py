import torch
import numpy as np
import re
import logging


def contain_one_space(outputs):
	return (np.char.count(outputs, ' ', start=1, end=-1) == 1)


def cut_at_first_dot(outputs, running):
	cut_outputs = []

	for idx, r in enumerate(running):
		if r:
			cut_idx = re.search(r'\.', outputs[idx]).span(0)[0]  # postion of first dot
			cut_outputs.append(outputs[idx][:cut_idx])
		else:
			cut_outputs.append(outputs[idx])
	return np.array(cut_outputs)


def have_stopped(outputs):
	return np.array(['.' in o for o in outputs])


def inputs_contain_substrings(inputs, outputs, running):
	inputs_contain_substrings = []
	
	for idx, r in enumerate(running):
		if r:
			try:
				_, substring = outputs[idx].split()
				inputs_contain_substrings += [substring in inputs[idx]]
			except ValueError as e:
				print(e)
				print(outputs[idx])
				inputs_contain_substrings += [False]
		else:
			inputs_contain_substrings += [r]

	return np.array(inputs_contain_substrings)


def replace_substrings_in_inputs(inputs, outputs, running):
	next_inputs = []

	for idx, r in enumerate(running):
		if r:
			result, substring = outputs[idx].split()
			next_inputs.append(inputs[idx].replace(substring, result))
		else:
			next_inputs.append('()')
		   
	return next_inputs


class SolverCombiner:
	
	def __init__(self, model, cfg):
		self.model = model
		self.running = []
		self.multi = cfg.multi
		self.multi_nofilter = cfg.multi_nofilter
		self.n_samples = cfg.n_samples

	def __call__(self, X, Y=None, tf=False, max_nes=0):
		self.model.eval()
		self.running = []
		lte = self.model.generator
		running = np.array([True]*X.size(0))
		
		for cur_nes in range(max_nes):
			logging.info(f"\n~~~ cur_nes {cur_nes} ~~~")
			# Y = Y if (cur_nes == (max_nes - 1)) else None
			if self.multi:
				next_inputs, running = self.multi_fwd(X, n_samples=self.n_samples, tf=tf)
			elif self.multi_nofilter:
				next_inputs, running = self.multi_fwd_nofilter(X, n_samples=self.n_samples, running=running, tf=tf)
			else:
				output = self.model(X, Y=None, tf=tf)
			if not self.multi and not self.multi_nofilter:
				next_inputs, running = self.model_output_to_next_input(X, output, running)
			X = lte._build_batch([list(i) for i in next_inputs])
			self.running.append(running)
		return lte._build_batch([list(i) + ['.'] for i in next_inputs], y=True)

	def multi_fwd(self, X, n_samples, tf=False):
		def get_valid_outputs_freq(outputs, valid):
			outputs_freq = dict()
			for o, v in zip(outputs, valid):
				if v:
					# result, substring = o.split()
					outputs_freq.setdefault(o, 0)
					outputs_freq[o] += 1
			return outputs_freq

		multi_output_tensors = []
		lte = self.model.generator
		chararray_inputs = np.array([x.replace('#', '') for x in lte.x_to_str(X)])

		logging.info("Sampling...")
		for sample_idx in range(n_samples):
			output = self.model(X, Y=None, tf=tf)
			multi_output_tensors.append(output)
		logging.info("Done.")

		multi_output = np.array([lte.y_to_str(o) for o in multi_output_tensors]).T  # outputs on the same row correspond to the same input
		valid = np.full(multi_output.shape, fill_value=True)
		multi_output_have_stopped = np.array([have_stopped(o) for o in multi_output])
		logging.info(f"{multi_output_have_stopped.sum(axis=1).mean()} multi-outputs have stopped on avg")
		valid &= multi_output_have_stopped
		multi_output = np.array([cut_at_first_dot(o, v) for o, v in zip(multi_output, valid)])
		multi_output_have_1_space = np.array([contain_one_space(o) for o in multi_output])
		logging.info(f"{multi_output_have_1_space.sum(axis=1).mean()} multi-outputs have one space on avg")
		valid &= multi_output_have_1_space
		input_contain_multi_substring = np.array([inputs_contain_substrings(np.tile(i, n_samples), o, v) for i, o, v in zip(chararray_inputs, multi_output, valid)])
		logging.info(f"{input_contain_multi_substring.sum(axis=1).mean()} multi-outputs have strings contained in inputs on avg")
		logging.info("Examples")
		logging.info(f"Input: {chararray_inputs[0]}")
		logging.info(f"{multi_output[0, ~input_contain_multi_substring[0]][:20]}")
		logging.info(f"Input: {chararray_inputs[1]}")
		logging.info(f"{multi_output[1, ~input_contain_multi_substring[1]][:20]}")
		valid &= input_contain_multi_substring
		valid_multi_outputs_freq = [get_valid_outputs_freq(o, v) for o, v in zip(multi_output, valid)]
		logging.info(valid_multi_outputs_freq)
		
		final_output = []
		running = []
		for output_idx, output_freq in enumerate(valid_multi_outputs_freq):
			if len(output_freq) == 0:  # no output was valid
				final_output.append(multi_output[output_idx, 0])  # take the first sample by default
				running.append(False)
			elif len(output_freq) == 1:  # some outputs were valid, all had same result
				for valid_output, freq in output_freq.items():
					final_output.append(valid_output)
				running.append(True)
			elif len(output_freq) > 1:  # some outputs were valid, they had different results
				max_freq = -1
				candidate = None
				for valid_output, freq in output_freq.items():
					if freq > max_freq:
						max_freq = freq
						candidate = valid_output
				assert candidate is not None
				final_output.append(candidate)
				running.append(True)

		final_output, running = np.array(final_output), np.array(running)
		next_input = replace_substrings_in_inputs(chararray_inputs, final_output, running)
		return next_input, running

	def multi_fwd_nofilter(self, X, n_samples, running, tf=False):
		def get_outputs_freq(outputs):
			outputs_freq = dict()
			for o in outputs:
				outputs_freq.setdefault(o, 0)
				outputs_freq[o] += 1
			return outputs_freq

		multi_output_tensors = []
		lte = self.model.generator
		chararray_inputs = np.array([x.replace('#', '') for x in lte.x_to_str(X)])

		logging.info("Sampling...")
		for sample_idx in range(n_samples):
			output = self.model(X, Y=None, tf=tf)
			multi_output_tensors.append(output)
		logging.info("Done.")

		multi_output = np.array([lte.y_to_str(o) for o in multi_output_tensors]).T  # outputs on the same row correspond to the same input
		multi_outputs_freq = [get_outputs_freq(o) for o in multi_output]
		
		most_frequent_outputs = []
		for outputs_freqs in multi_outputs_freq:
			max_freq = -1
			candidate = None
			for output, freq in outputs_freqs.items():
				if freq > max_freq:
					max_freq = freq
					candidate = output
			assert candidate is not None
			most_frequent_outputs.append(candidate)
		chararray_outputs = np.array(most_frequent_outputs)
		
		# check output structure
		outputs_have_stopped = have_stopped(chararray_outputs)
		running &= outputs_have_stopped
		chararray_outputs = cut_at_first_dot(chararray_outputs, running)
		max_cut_length = max([len(o) for o in chararray_outputs])
		
		logging.info(f"{(~outputs_have_stopped).sum()} outputs have not stopped.")
		logging.info(f"{running.sum()} outputs are running.")
		
		outputs_are_well_formed = contain_one_space(chararray_outputs)
		logging.info(f"\n{(~outputs_are_well_formed & running).sum()} outputs are not well formed.")
		running &= outputs_are_well_formed

		if (~outputs_are_well_formed & running).sum() > 0:
			notwell_formed_running_inputs = chararray_inputs[~outputs_are_well_formed & running]
			num_log_idx = 10 if notwell_formed_running_inputs.shape[0] > 10 else notwell_formed_running_inputs.shape[0]
			log_idx = np.random.choice(notwell_formed_running_inputs.shape[0], size=num_log_idx, replace=False)
			
			logging.info('\n'.join([f"{i} → {o}"
				for i, o in zip(notwell_formed_running_inputs[log_idx], chararray_outputs[~outputs_are_well_formed & running][log_idx])]))
			
		# check substring in input
		inputs_do_contain_substrings = inputs_contain_substrings(chararray_inputs, chararray_outputs, running)
		logging.info(f"\n{(~inputs_do_contain_substrings & running).sum()} outputs have wrong substrings.")
		
		if (~inputs_do_contain_substrings & running).sum() > 0:
			inputs_without_substring_running = chararray_inputs[~inputs_do_contain_substrings & running]
			num_log_idx = 10 if inputs_without_substring_running.shape[0] > 10 else inputs_without_substring_running.shape[0]
			log_idx = np.random.choice(inputs_without_substring_running.shape[0], size=num_log_idx, replace=False)
			
			logging.info('\n'.join([f"{i} → {o}"
				for i, o in zip(inputs_without_substring_running[log_idx], chararray_outputs[~inputs_do_contain_substrings & running][log_idx])]))
			
		running &= inputs_do_contain_substrings
		next_input = replace_substrings_in_inputs(chararray_inputs, chararray_outputs, running)

		logging.info(f"\n{running.sum()} outputs are running.")
		
		return next_input, running

	def model_output_to_next_input(self, X, output_tensor, running):
		lte = self.model.generator
		y_vocab_itos = lte.y_vocab.get_itos()
		itos_f = np.vectorize(lambda x: y_vocab_itos[x])
		cur_input, output = lte.x_to_str(X), lte.y_to_str(output_tensor)
		chararray_outputs = np.array(output)
		chararray_inputs = np.array([x.replace('#', '') for x in cur_input])

		# check output structure
		outputs_have_stopped = have_stopped(chararray_outputs)
		running &= outputs_have_stopped
		chararray_outputs = cut_at_first_dot(chararray_outputs, running)
		max_cut_length = max([len(o) for o in chararray_outputs])
		
		logging.info(f"{(~outputs_have_stopped).sum()} outputs have not stopped.")
		logging.info(f"{running.sum()} outputs are running.")
		
		outputs_are_well_formed = contain_one_space(chararray_outputs)
		logging.info(f"\n{(~outputs_are_well_formed & running).sum()} outputs are not well formed.")

		if (~outputs_are_well_formed & running).sum() > 0:
			notwell_formed_running_inputs = chararray_inputs[~outputs_are_well_formed & running]
			num_log_idx = 10 if notwell_formed_running_inputs.shape[0] > 10 else notwell_formed_running_inputs.shape[0]
			log_idx = np.random.choice(notwell_formed_running_inputs.shape[0], size=num_log_idx, replace=False)
			top2_logits, top2_idx = output_tensor[torch.tensor(~outputs_are_well_formed & running, device=output_tensor.device)][torch.tensor(log_idx[:10])].topk(k=2, dim=-1)
			
			logging.info('\n'.join([f"{i} → {o}"
				for i, o in zip(notwell_formed_running_inputs[log_idx], chararray_outputs[~outputs_are_well_formed & running][log_idx])]))
			logging.info("\nTop 2 logits & predictions for first 10 ill-formed model outputs")
			logging.info('\n\n'.join([f"{logits[:max_cut_length]}\n{idx[:max_cut_length]}"
				for logits, idx in zip(top2_logits.cpu().numpy().round(decimals=2), itos_f(top2_idx.cpu().numpy()))]))
			
		running &= outputs_are_well_formed
		logging.info(f"{running.sum()} outputs are running.")
		
		# check substring in input
		inputs_do_contain_substrings = inputs_contain_substrings(chararray_inputs, chararray_outputs, running)
		logging.info(f"\n{(~inputs_do_contain_substrings & running).sum()} outputs have wrong substrings.")
		
		if (~inputs_do_contain_substrings & running).sum() > 0:
			inputs_without_substring_running = chararray_inputs[~inputs_do_contain_substrings & running]
			num_log_idx = 10 if inputs_without_substring_running.shape[0] > 10 else inputs_without_substring_running.shape[0]
			log_idx = np.random.choice(inputs_without_substring_running.shape[0], size=num_log_idx, replace=False)
			top2_logits, top2_idx = output_tensor[torch.tensor(~inputs_do_contain_substrings & running, device=output_tensor.device)][torch.tensor(log_idx[:10])].topk(k=2, dim=-1)
			
			logging.info('\n'.join([f"{i} → {o}"
				for i, o in zip(inputs_without_substring_running[log_idx], chararray_outputs[~inputs_do_contain_substrings & running][log_idx])]))
			logging.info("\nTop 2 logits & predictions for first 10 no-substring model outputs")
			logging.info('\n\n'.join([f"{logits[:max_cut_length]}\n{idx[:max_cut_length]}"
				for logits, idx in zip(top2_logits.cpu().numpy().round(decimals=2), itos_f(top2_idx.cpu().numpy()))]))
			
		running &= inputs_do_contain_substrings
		next_input = replace_substrings_in_inputs(chararray_inputs, chararray_outputs, running)

		logging.info(f"\n{running.sum()} outputs are running.")
		
		return next_input, running
