import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Dropout, LayerNorm, Linear, MultiheadAttention, Sequential


class Transformer(torch.nn.Module):
    
    def __init__(self, d_model, num_heads, num_layers, generator, label_pe=False, deterministic=True, max_range_pe=5000, dropout=0.1, device='cuda'):
        super(Transformer, self).__init__()
        self.x_emb = Linear(len(generator.x_vocab), d_model)
        self.y_emb = Linear(len(generator.y_vocab), d_model)
        self.encoder = Encoder(d_model, num_heads, num_layers, dropout=dropout, label_pe=label_pe, device=device)
        self.decoder = Decoder(d_model, num_heads, num_layers, dropout=dropout, label_pe=label_pe, device=device)
        self.final_proj = Linear(d_model, len(generator.y_vocab))
        self.generator = generator
        self.deterministic = deterministic
    
    def forward(self, X, Y=None, tf=False):
        if Y is not None:
            return self._fwd(X, Y, tf=tf)
        else:
            return self._test_fwd(X)

    def _fwd(self, X, Y, tf=False):
        src_mask = (X.argmax(-1) == self.generator.x_vocab['#'])
        tgt_mask = (Y.argmax(-1) == self.generator.y_vocab['#'])
        X = self.x_emb(X)

        if not tf:
            X, src_mask = self._encoder(X, src_mask)
            Y_pred_v = Y[:, 0, :].unsqueeze(1)
            output = Y_pred_v
            for t in range(Y.size(1)):
                Y_pred = self.y_emb(Y_pred_v)
                self.Y_logits = self.decoder(X, Y_pred, src_mask, None)
                Y_pred = self.final_proj(self.Y_logits)
                Y_pred = Y_pred[:, -1].unsqueeze(1)  # take only the last pred
                output = torch.concat([output, Y_pred], dim=1) 
                pred_idx = Y_pred.argmax(-1) 
                Y_sample = F.one_hot(pred_idx, num_classes=len(self.generator.y_vocab)).type(torch.FloatTensor).to(X.device)
                Y_pred_v = torch.concat([Y_pred_v, Y_sample], dim=1)
            return output[:, 1:, :]
        else:
            Y = self.y_emb(Y)
            X, src_mask = self._encoder(X, src_mask)
            self.Y_logits = self.decoder(X, Y, src_mask, tgt_mask)
            return self.final_proj(self.Y_logits)
    
    def _encoder(self, X, src_mask):
        return self.encoder(X, src_mask), src_mask

    def _test_fwd(self, X):
        it, max_it = 0, 100
        src_mask = (X.argmax(-1) == self.generator.x_vocab['#'])
        X = self.x_emb(X)
        EOS_idx = self.generator.y_vocab['.']
        encoding, src_mask = self._encoder(X, src_mask)
        stopped = torch.zeros(X.size(0)).type(torch.BoolTensor).to(X.device)
        Y_pred_v = torch.tile(F.one_hot(torch.tensor([self.generator.y_vocab['?']]), num_classes=len(self.generator.y_vocab)), dims=(X.size(0), 1, 1)).type(torch.FloatTensor).to(X.device)
        output = Y_pred_v

        while not stopped.all() and (it < max_it):
            it += 1
            Y_pred = self.y_emb(Y_pred_v)
            self.Y_logits = self.decoder(encoding, Y_pred, src_mask, None)
            Y_pred = self.final_proj(self.Y_logits)
            Y_pred = Y_pred[:, -1].unsqueeze(1)  # take only the last pred
            output = torch.concat([output, Y_pred], dim=1)     
            if self.deterministic:
                pred_idx = Y_pred.argmax(-1) 
            else:
                pred_idx = torch.multinomial(F.softmax(Y_pred.squeeze(), dim=-1), num_samples=1)  # we can squeeze bc we take 1 sample
            Y_sample = F.one_hot(pred_idx, num_classes=len(self.generator.y_vocab)).type(torch.FloatTensor).to(X.device)
            Y_pred_v = torch.concat([Y_pred_v, Y_sample], dim=1)
            stopped = torch.logical_or((pred_idx.squeeze() == EOS_idx), stopped)
        return output[:, 1:, :]

    def _test_fwd_encode_step(self, X):
        src_mask = (X.argmax(-1) == self.generator.x_vocab['#'])
        X = self.x_emb(X)
        encoding, src_mask = self._encoder(X, src_mask)
        return encoding, src_mask

    def _test_fwd_decode_step(self, encoding, src_mask, Y_pred_v):
        Y_pred = self.y_emb(Y_pred_v)
        self.Y_logits = self.decoder(encoding, Y_pred, src_mask, None)
        Y_pred = self.final_proj(self.Y_logits)
        Y_pred = Y_pred[:, -1].unsqueeze(1)  # take only the last pred
        return Y_pred


class Encoder(torch.nn.Module):

    def __init__(self, d_model, num_heads, num_layers, dropout=0.1, max_range_pe=5000, label_pe=False, device='cpu'):
        super(Encoder, self).__init__()
        self.device = device
        positional_encoding = _gen_timing_signal(max_range_pe, d_model)
        self.register_buffer('positional_encoding', positional_encoding)
        
        self.MHSA = MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.dropout1 = Dropout(dropout)
        self.layer_norm1 = LayerNorm(d_model)
        self.transition_fn = Sequential(Linear(d_model, d_model),
                                        torch.nn.ReLU(),
                                        Linear(d_model, d_model))
        self.dropout2 = Dropout(dropout)
        self.layer_norm2 = LayerNorm(d_model)
        self.num_layers = num_layers
        self.label_pe = label_pe

    def forward(self, X, src_mask):
        X = self._pe(X)
        X = self._encoder(X, src_mask)
        return X

    def _encoder(self, X, src_mask):
        Xt, attn = self.MHSA(X, X, X, key_padding_mask=src_mask)
        X = X + self.dropout1(Xt)
        X = self.layer_norm1(X)
        X = X + self.dropout2(self.transition_fn(X))
        X = self.layer_norm2(X)
        return X

    def _pe(self, X):
        if self.label_pe:
            max_seq_len = X.size(1)
            max_pe_pos = self.positional_encoding.size(1)
            val, idx = torch.sort(torch.randint(low=0, high=max_pe_pos, size=(max_seq_len,)))
            return X + self.dropout1(self.positional_encoding[:, val, :])
        else:
            return X + self.dropout1(self.positional_encoding[:, :X.size(1), :])


class Decoder(torch.nn.Module):
    
    def __init__(self, d_model, num_heads, num_layers, dropout=0.1, max_range_pe=5000, label_pe=False, device='cpu'):
        super(Decoder, self).__init__()
        self.device = device
        positional_encoding = _gen_timing_signal(max_range_pe, d_model)
        self.register_buffer('positional_encoding', positional_encoding)
        
        self.MHSA = MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.dropout1 = Dropout(dropout)
        self.layer_norm1 = LayerNorm(d_model)
        self.MHA = MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.dropout2 = Dropout(dropout)
        self.layer_norm2 = LayerNorm(d_model)
        self.transition_fn = Sequential(Linear(d_model, d_model),
                                        torch.nn.ReLU(),
                                        Linear(d_model, d_model))
        self.dropout3 = Dropout(dropout)
        self.layer_norm3 = LayerNorm(d_model)
        self.num_layers = num_layers
        self.device = device
        self.label_pe = label_pe
    
    def forward(self, X, Y, src_mask, tgt_mask):
        Y = self._pe(Y)
        Y = self._decoder(X, Y, src_mask, tgt_mask)
        return Y

    def _decoder(self, X, Y, src_mask, tgt_mask):
        Yt, attn = self.MHSA(Y, Y, Y, attn_mask=_gen_bias_mask(Y.size(1), self.device), key_padding_mask=tgt_mask)
        Y = Y + self.dropout1(Yt)
        Y = self.layer_norm1(Y)
        Yt, attn = self.MHA(Y, X, X, key_padding_mask=src_mask)
        Y = Y + self.dropout2(Yt)
        Y = self.layer_norm2(Y)
        Y = self.dropout3(self.transition_fn(Y))
        Y = self.layer_norm3(Y)
        return Y

    def _pe(self, X):
        if self.label_pe:
            max_seq_len = X.size(1)
            max_pe_pos = self.positional_encoding.size(1)
            val, idx = torch.sort(torch.randint(low=0, high=max_pe_pos, size=(max_seq_len,)))
            return X + self.dropout1(self.positional_encoding[:, val, :])
        else:
            return X + self.dropout1(self.positional_encoding[:, :X.size(1), :])


def _gen_bias_mask(max_len, device):
    """
    Generates bias values (True) to mask future timesteps during attention
    """
    np_mask = np.triu(np.full([max_len, max_len], 1), 1)
    torch_mask = torch.from_numpy(np_mask).type(torch.BoolTensor).to(device)
    
    return torch_mask


def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.zeros((scaled_time.shape[0], 2*scaled_time.shape[1]))
    signal[:, 0::2] = np.sin(scaled_time)
    signal[:, 1::2] = np.cos(scaled_time)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], 
                    'constant', constant_values=[0.0, 0.0])
    signal =  signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)
