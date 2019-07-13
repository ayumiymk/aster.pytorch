from __future__ import absolute_import

import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


class AttentionRecognitionHead(nn.Module):
  """
  input: [b x 16 x 64 x in_planes]
  output: probability sequence: [b x T x num_classes]
  """
  def __init__(self, num_classes, in_planes, sDim, attDim, max_len_labels):
    super(AttentionRecognitionHead, self).__init__()
    self.num_classes = num_classes # this is the output classes. So it includes the <EOS>.
    self.in_planes = in_planes
    self.sDim = sDim
    self.attDim = attDim
    self.max_len_labels = max_len_labels

    self.decoder = DecoderUnit(sDim=sDim, xDim=in_planes, yDim=num_classes, attDim=attDim)

  def forward(self, x):
    x, targets, lengths = x
    batch_size = x.size(0)
    # Decoder
    state = torch.zeros(1, batch_size, self.sDim)
    outputs = []

    for i in range(max(lengths)):
      if i == 0:
        y_prev = torch.zeros((batch_size)).fill_(self.num_classes) # the last one is used as the <BOS>.
      else:
        y_prev = targets[:,i-1]

      output, state = self.decoder(x, state, y_prev)
      outputs.append(output)
    outputs = torch.cat([_.unsqueeze(1) for _ in outputs], 1)
    return outputs

  # inference stage.
  def sample(self, x):
    x, _, _ = x
    batch_size = x.size(0)
    # Decoder
    state = torch.zeros(1, batch_size, self.sDim)

    predicted_ids, predicted_scores = [], []
    for i in range(self.max_len_labels):
      if i == 0:
        y_prev = torch.zeros((batch_size)).fill_(self.num_classes)
      else:
        y_prev = predicted

      output, state = self.decoder(x, state, y_prev)
      output = F.softmax(output, dim=1)
      score, predicted = output.max(1)
      predicted_ids.append(predicted.unsqueeze(1))
      predicted_scores.append(score.unsqueeze(1))
    predicted_ids = torch.cat(predicted_ids, 1)
    predicted_scores = torch.cat(predicted_scores, 1)
    # return predicted_ids.squeeze(), predicted_scores.squeeze()
    return predicted_ids, predicted_scores

  def beam_search(self, x, beam_width, eos):

    def _inflate(tensor, times, dim):
      repeat_dims = [1] * tensor.dim()
      repeat_dims[dim] = times
      return tensor.repeat(*repeat_dims)

    # https://github.com/IBM/pytorch-seq2seq/blob/fede87655ddce6c94b38886089e05321dc9802af/seq2seq/models/TopKDecoder.py
    batch_size, l, d = x.size()
    # inflated_encoder_feats = _inflate(encoder_feats, beam_width, 0) # ABC --> AABBCC -/-> ABCABC
    inflated_encoder_feats = x.unsqueeze(1).permute((1,0,2,3)).repeat((beam_width,1,1,1)).permute((1,0,2,3)).contiguous().view(-1, l, d)

    # Initialize the decoder
    state = torch.zeros(1, batch_size * beam_width, self.sDim)
    pos_index = (torch.Tensor(range(batch_size)) * beam_width).long().view(-1, 1)

    # Initialize the scores
    sequence_scores = torch.Tensor(batch_size * beam_width, 1)
    sequence_scores.fill_(-float('Inf'))
    sequence_scores.index_fill_(0, torch.Tensor([i * beam_width for i in range(0, batch_size)]).long(), 0.0)
    # sequence_scores.fill_(0.0)

    # Initialize the input vector
    y_prev = torch.zeros((batch_size * beam_width)).fill_(self.num_classes)

    # Store decisions for backtracking
    stored_scores          = list()
    stored_predecessors    = list()
    stored_emitted_symbols = list()

    for i in range(self.max_len_labels):
      output, state = self.decoder(inflated_encoder_feats, state, y_prev)
      log_softmax_output = F.log_softmax(output, dim=1)

      sequence_scores = _inflate(sequence_scores, self.num_classes, 1)
      sequence_scores += log_softmax_output
      scores, candidates = sequence_scores.view(batch_size, -1).topk(beam_width, dim=1)

      # Reshape input = (bk, 1) and sequence_scores = (bk, 1)
      y_prev = (candidates % self.num_classes).view(batch_size * beam_width)
      sequence_scores = scores.view(batch_size * beam_width, 1)

      # Update fields for next timestep
      predecessors = (candidates / self.num_classes + pos_index.expand_as(candidates)).view(batch_size * beam_width, 1)
      state = state.index_select(1, predecessors.squeeze())

      # Update sequence socres and erase scores for <eos> symbol so that they aren't expanded
      stored_scores.append(sequence_scores.clone())
      eos_indices = y_prev.view(-1, 1).eq(eos)
      if eos_indices.nonzero().dim() > 0:
        sequence_scores.masked_fill_(eos_indices, -float('inf'))

      # Cache results for backtracking
      stored_predecessors.append(predecessors)
      stored_emitted_symbols.append(y_prev)

    # Do backtracking to return the optimal values
    #====== backtrak ======#
    # Initialize return variables given different types
    p = list()
    l = [[self.max_len_labels] * beam_width for _ in range(batch_size)]  # Placeholder for lengths of top-k sequences

    # the last step output of the beams are not sorted
    # thus they are sorted here
    sorted_score, sorted_idx = stored_scores[-1].view(batch_size, beam_width).topk(beam_width)
    # initialize the sequence scores with the sorted last step beam scores
    s = sorted_score.clone()

    batch_eos_found = [0] * batch_size  # the number of EOS found
                                        # in the backward loop below for each batch
    t = self.max_len_labels - 1
    # initialize the back pointer with the sorted order of the last step beams.
    # add pos_index for indexing variable with b*k as the first dimension.
    t_predecessors = (sorted_idx + pos_index.expand_as(sorted_idx)).view(batch_size * beam_width)
    while t >= 0:
      # Re-order the variables with the back pointer
      current_symbol = stored_emitted_symbols[t].index_select(0, t_predecessors)
      t_predecessors = stored_predecessors[t].index_select(0, t_predecessors).squeeze()
      eos_indices = stored_emitted_symbols[t].eq(eos).nonzero()
      if eos_indices.dim() > 0:
        for i in range(eos_indices.size(0)-1, -1, -1):
          # Indices of the EOS symbol for both variables
          # with b*k as the first dimension, and b, k for
          # the first two dimensions
          idx = eos_indices[i]
          b_idx = int(idx[0] / beam_width)
          # The indices of the replacing position
          # according to the replacement strategy noted above
          res_k_idx = beam_width - (batch_eos_found[b_idx] % beam_width) - 1
          batch_eos_found[b_idx] += 1
          res_idx = b_idx * beam_width + res_k_idx

          # Replace the old information in return variables
          # with the new ended sequence information
          t_predecessors[res_idx] = stored_predecessors[t][idx[0]]
          current_symbol[res_idx] = stored_emitted_symbols[t][idx[0]]
          s[b_idx, res_k_idx] = stored_scores[t][idx[0], [0]]
          l[b_idx][res_k_idx] = t + 1

      # record the back tracked results
      p.append(current_symbol)

      t -= 1

    # Sort and re-order again as the added ended sequences may change
    # the order (very unlikely)
    s, re_sorted_idx = s.topk(beam_width)
    for b_idx in range(batch_size):
      l[b_idx] = [l[b_idx][k_idx.item()] for k_idx in re_sorted_idx[b_idx,:]]

    re_sorted_idx = (re_sorted_idx + pos_index.expand_as(re_sorted_idx)).view(batch_size*beam_width)

    # Reverse the sequences and re-order at the same time
    # It is reversed because the backtracking happens in reverse time order
    p = [step.index_select(0, re_sorted_idx).view(batch_size, beam_width, -1) for step in reversed(p)]
    p = torch.cat(p, -1)[:,0,:]
    return p, torch.ones_like(p)


class AttentionUnit(nn.Module):
  def __init__(self, sDim, xDim, attDim):
    super(AttentionUnit, self).__init__()

    self.sDim = sDim
    self.xDim = xDim
    self.attDim = attDim

    self.sEmbed = nn.Linear(sDim, attDim)
    self.xEmbed = nn.Linear(xDim, attDim)
    self.wEmbed = nn.Linear(attDim, 1)

    # self.init_weights()

  def init_weights(self):
    init.normal_(self.sEmbed.weight, std=0.01)
    init.constant_(self.sEmbed.bias, 0)
    init.normal_(self.xEmbed.weight, std=0.01)
    init.constant_(self.xEmbed.bias, 0)
    init.normal_(self.wEmbed.weight, std=0.01)
    init.constant_(self.wEmbed.bias, 0)

  def forward(self, x, sPrev):
    batch_size, T, _ = x.size()                      # [b x T x xDim]
    x = x.view(-1, self.xDim)                        # [(b x T) x xDim]
    xProj = self.xEmbed(x)                           # [(b x T) x attDim]
    xProj = xProj.view(batch_size, T, -1)            # [b x T x attDim]

    sPrev = sPrev.squeeze(0)
    sProj = self.sEmbed(sPrev)                       # [b x attDim]
    sProj = torch.unsqueeze(sProj, 1)                # [b x 1 x attDim]
    sProj = sProj.expand(batch_size, T, self.attDim) # [b x T x attDim]

    sumTanh = torch.tanh(sProj + xProj)
    sumTanh = sumTanh.view(-1, self.attDim)

    vProj = self.wEmbed(sumTanh) # [(b x T) x 1]
    vProj = vProj.view(batch_size, T)

    alpha = F.softmax(vProj, dim=1) # attention weights for each sample in the minibatch

    return alpha


class DecoderUnit(nn.Module):
  def __init__(self, sDim, xDim, yDim, attDim):
    super(DecoderUnit, self).__init__()
    self.sDim = sDim
    self.xDim = xDim
    self.yDim = yDim
    self.attDim = attDim
    self.emdDim = attDim

    self.attention_unit = AttentionUnit(sDim, xDim, attDim)
    self.tgt_embedding = nn.Embedding(yDim+1, self.emdDim) # the last is used for <BOS> 
    self.gru = nn.GRU(input_size=xDim+self.emdDim, hidden_size=sDim, batch_first=True)
    self.fc = nn.Linear(sDim, yDim)

    # self.init_weights()

  def init_weights(self):
    init.normal_(self.tgt_embedding.weight, std=0.01)
    init.normal_(self.fc.weight, std=0.01)
    init.constant_(self.fc.bias, 0)

  def forward(self, x, sPrev, yPrev):
    # x: feature sequence from the image decoder.
    batch_size, T, _ = x.size()
    alpha = self.attention_unit(x, sPrev)
    context = torch.bmm(alpha.unsqueeze(1), x).squeeze(1)
    yProj = self.tgt_embedding(yPrev.long())
    # self.gru.flatten_parameters()
    output, state = self.gru(torch.cat([yProj, context], 1).unsqueeze(1), sPrev)
    output = output.squeeze(1)

    output = self.fc(output)
    return output, state