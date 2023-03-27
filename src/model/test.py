import torch
from data.generator import _PAD
import warnings


def compute_loss(loss, outputs, target, generator):
	if not isinstance(outputs, list):
		outputs = [outputs[:, pos, :] for pos in range(outputs.size(1))]
	cumulative_loss = 0
	idx_pad = generator.y_vocab[_PAD]
	idx_targets = target.argmax(dim=-1)
	mask = (idx_targets != idx_pad).type(torch.int32)
	for char_pos, output in enumerate(outputs):
		char_loss = loss(output, torch.argmax(target[:, char_pos, :].squeeze(), dim=1))
		masked_char_loss = char_loss * mask[:, char_pos]
		cumulative_loss += masked_char_loss.sum()
	avg_loss = cumulative_loss / mask.sum()
	return avg_loss


def _fix_output_shape(output, Y, generator):
    # fix pred/target shape mismatch
    if output.size(1) < Y.size(1):
        missing_timesteps = Y.size(1) - output.size(1)
        pad_vecs = torch.nn.functional.one_hot(torch.tensor(generator.y_vocab['#'],
                                                            device=Y.device)).tile(output.size(0),
                                                                                   missing_timesteps, 1)
        output = torch.concat([output, pad_vecs], dim=1)
    elif output.size(1) > Y.size(1):
        output = output[:, :Y.size(1)]
    return output
	

def batch_acc(outputs, targets, vocab_size, generator):
	if isinstance(outputs, list):
		outputs = torch.concat([o.unsqueeze(1) for o in outputs], dim=1)	
	idx_pad = generator.y_vocab[_PAD]
	idx_targets = targets.argmax(dim=-1)
	mask = (idx_targets != idx_pad).to(outputs.device)
	idx_outs = outputs.argmax(dim=-1)
	out_equal_target = (idx_outs == idx_targets).type(torch.FloatTensor).to(outputs.device)
	valid_out_equal_target = torch.masked_select(out_equal_target, mask)
	return valid_out_equal_target.mean(), valid_out_equal_target.std()


def batch_seq_acc(outputs, targets, generator, len_Y):
	idx_pad = generator.y_vocab[_PAD]
	idx_targets = targets.argmax(dim=-1)
	mask = (idx_targets != idx_pad).type(torch.int32)
	idx_outs = outputs.argmax(dim=-1)
	out_equal_target = (idx_outs == idx_targets).type(torch.int32)
	masked_out_equal_target = out_equal_target * mask
	num_equal_chars_per_seq = masked_out_equal_target.sum(dim=-1)
	pred_is_exact = (num_equal_chars_per_seq == len_Y).type(torch.FloatTensor)
	return pred_is_exact.mean(), pred_is_exact.std()
