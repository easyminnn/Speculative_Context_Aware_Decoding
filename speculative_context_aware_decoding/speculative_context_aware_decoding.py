import math

import torch
from torch.nn import Module, ModuleList
from torch import nn, einsum, Tensor
import torch.nn.functional as F

from rotary_embedding_torch import RotaryEmbedding

from collections import namedtuple

from einops import rearrange

# constants

Cache = namedtuple('Cache', ['cached_kvs', 'embeds'])

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

# rotary embeddings

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len, device = self.inv_freq.device).type_as(self.inv_freq)
        freqs = einsum('i, j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos, t):
    seq_len = t.shape[-2]
    pos = pos[-seq_len:, :]
    return t * pos.cos() + rotate_half(t) * pos.sin()

# different decoding strategies

@torch.no_grad()
def base_decoding(
    net: Module,
    prompt: Tensor,
    seq_len: int,
    temperature = 1.,
    filter_thres = 0.9,
):
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    sample_num_times = max(0, seq_len - prompt_seq_len)

    cache = None

    for _ in range(sample_num_times):
        logits, cache = net(out, cache = cache, return_cache = True)
        logits = logits[:, -1]

        logits = top_k(logits, thres = filter_thres)
        sample = gumbel_sample(logits, temperature = temperature, dim = -1)

        out = torch.cat((out, sample[..., None]), dim = -1)

    return out[..., prompt_seq_len:]

# speculative decoding functions

def safe_div(num, den, eps = 1e-10):
    return num / max(den, eps)

def find_first_true_index(bool_tensor, dim = -1):
    return (bool_tensor.cumsum(dim = dim) == 0).sum(dim = dim)

@torch.no_grad()
def speculative_context_aware_decoding(
    net: Module,
    small_net: Module,
    prompt: Tensor,
    seq_len: int,
    gamma: int = 5,
    temperature = 1.,
    filter_thres = 0.9,
    lenience = 1.,
    pad_id = 0,
    beta = 0.3,
    alpha = 0.1
):
    """
    Implementation for speculative contrastive decoding

    net: large model
    small_net: small model
    prompt: starting sequence of tokens
    seq_len: total desired length of sequence
    gamma: number of speculative steps
    temperature: sampling temperature
    filter_thres: threshold for top-k filtering
    lenience: lenience factor
    pad_id: token id for padding
    beta : parameter for context aware decoding 
    alpha : parameter for plausibility constraint

    Returns:
        out (Tensor): generated tokens after the prefix
    """

   
    batch, prompt_seq_len, out, device = *prompt.shape, prompt.clone(), prompt.device
    # out : a copy of the prompt tensor which will keep getting appended with new tokens
    # batch : how many sequences are processed in parallel (batch = 1)
    # prompt_seq_len : length of the prompt sequence (prompt_seq_len = 128)
    sample_num_times = max(0, seq_len - prompt_seq_len)
    
    cache = None
    small_cache = None
    # cache = None because we don't have any cache yet

    num_steps = 0
    total_accepted = 0
    # num_steps : number of speculative steps taken
    # total_accepted : number of how many times the speculative token was accepted

    batch_range = torch.arange(batch, device = device, dtype = torch.long)[..., None]
    # batch_range : tensor of shape (batch, 1) with values from 0 to batch-1
    seq_lens = torch.full((batch,), prompt_seq_len, device = device, dtype = torch.long)
    # seq_lens : store the length of the sequence generated so far for each batch
    # initialized with the length of the prompt sequence
     
    while (seq_lens < seq_len).any(): # check if the length of the generated sequence is less than the desired length

        # predict with smaller network

        all_small_logits = [] # logits from the small model for each speculative token
        q_sampled_out = [] # chosen speculative token from the small model

        for _ in range(gamma):   # for each speculative gamma step, predict the next token using the small model
            small_logits, small_cache = small_net(
                out,
                seq_start_pos = out.shape[-1] - seq_lens,
                cache = small_cache,
                return_cache = True
            )
    
            small_logits = small_logits[:, -1] # get the logits of the last token

            small_logits = top_k(small_logits, thres = filter_thres) 
            all_small_logits.append(small_logits) # append the top-k logits to the list

            sample = gumbel_sample(small_logits, temperature = temperature, dim = -1)
            # sample a token from the logits
            out = torch.cat((out, sample[..., None]), dim = -1)
            # concat the sampled token to the sequence(out = prompt + generated tokens)
            seq_lens += 1

            q_sampled_out.append(rearrange(sample, 'b -> b 1 1'))
            # append the sampled token to the list

        # after gamma steps, which is 'for' loop, we have gamma tokens sampled from the small model

        q_sampled_out = torch.cat(q_sampled_out, dim = -2)  # [batch, gamma, 1]
        small_logits = torch.stack(all_small_logits, dim = -2) # [1, gamma, 256]
        
        # logits for each speculative step

        ### verify with larger network (with the large model)###
        logits, cache = net(
            out,
            seq_start_pos = out.shape[-1] - seq_lens,
            cache = cache,
            return_cache = True
        )
        # logits : includes the logits for the last gamma+1 tokens
        
        logits = logits[..., -(gamma + 1):-1, :] # we only need the gamma tokens
        logits = top_k(logits, thres = filter_thres) 
        
        ### prob and prob of small model (p(x) and q(x) in algorithm 1) ###
        prob = safe_div(logits, temperature).softmax(dim = -1) # prob from the large model
        small_prob = safe_div(small_logits, temperature).softmax(dim = -1) # prob from the small model

        # print("1. prob")
        # print(prob[0][0])
        # print("2. small_prob")
        # print(small_prob[0][0])


        # contrastive probability
        # prob_contrast = torch.where(
        #     prob >= (alpha * prob.max(dim=-1, keepdim=True)[0]),
        #     torch.log(prob) - torch.log(
        #         torch.where(small_prob > 0, small_prob, torch.ones_like(small_prob))
        #     ),
        #     torch.full_like(prob, float('-inf'))
        # )

        # this is for weighted prob (not contrast, just for test)
        prob_contrast = torch.where(
            prob >= (alpha * prob.max(dim=-1, keepdim=True)[0]),
            torch.add((1-beta) * prob , beta * small_prob), 
            torch.full_like(prob, float('-inf'))
        )
            
        # print(type(prob_contrast))
        # print(prob_contrast[0][0])
       
        
        # conteastive prob -> softmax 
        prob_contrast_normalized = safe_div(prob_contrast, temperature).softmax(dim = -1)

        # print(prob_contrast_normalized[0][0])

        
        ### 여기 보완 필요. 여기도 contrastive 해야 하는데 그냥 next logit만 해버림###
        next_logits = logits[..., -1:, :] # logits for the next token
        next_prob = safe_div(next_logits, temperature).softmax(dim = -1) # probability of the next token
        #########################

        p, prob_next = prob_contrast_normalized[:, :], next_prob[:, -1] # p(x) for the last gamma tokens and the next token

        p = p.gather(-1, q_sampled_out) # p now contains the probabilities of the tokens sampled from the small model
        
        q = small_prob.gather(-1, q_sampled_out) * lenience  #  q_sampled_out is the token sampled from the small model
        # for example, if q_sampled out is [1, 2, 3], then p will contain the probabilities of the tokens at index 1, 2, 3

        p, q = [rearrange(t, 'b n 1 -> b n') for t in (p, q)]
        # flatten the tensors since it's just a single token probability now.
        # p and q are now of shape (batch, gamma)

        r = random_uniform = torch.zeros_like(q).float().uniform_(0, 1)
        # r is a random tensor of the same shape as q, and values between 0 and 1 (uniform distribution)

        #print(p / q)
        #print(r)
    
        accepted = find_first_true_index(r > (p / q))

        # print(accepted)


        
        # compare r and p/q
        # if r > p/q, then the token is accepted
        # if none fail, accepted = gamma
        # otherwise, accepted is the index of the first failure, and the tokens before that are accepted
        # token generation statistics
        total_accepted += accepted.float().mean() # add the mean of the accepted tokens to the total accepted
        num_steps += 1

        num_rejected = gamma - accepted
        has_rejected = num_rejected > 0

        accepted = rearrange(accepted, 'b -> b 1')
        accepted.clamp_(max = gamma - 1) # clamp the accepted value to gamma-1 because index starts from 0
        adjusted_prob = F.relu(prob_contrast_normalized[batch_range, accepted] - small_prob[batch_range, accepted])
        # normalize max(0, p - q) to get the adjusted probability
        adjusted_prob = adjusted_prob / adjusted_prob.sum(dim = -1, keepdim = True)
        adjusted_prob = rearrange(adjusted_prob, 'b 1 d -> b d')

     
        prob_next = torch.where(
            rearrange(has_rejected, '... -> ... 1'),
            adjusted_prob,
            prob_next
        )
        # if we have rejection, the gamma+1 token probability is adjusted

        ### do a bunch of slicing and align everything to the right, including kv caches ###

        seq_lens -= num_rejected # reduce the sequence length by the number of rejections
        max_seq_len = seq_lens.amax() # compute the maximum sequence length after rejections
        curr_len = out.shape[-1]

        seq_arange = torch.arange(max_seq_len, device = device, dtype = torch.long) + (curr_len - max_seq_len)

        seq_offset_indices = seq_arange - num_rejected[..., None]

        cached_kv, _ = cache
        small_cached_kv, _ = small_cache

    
        if batch > 1:
            small_cached_kv = F.pad(small_cached_kv, (0, 0, 0, 1))  # account for small model being a token behind
            
            out = out[batch_range, seq_offset_indices]

            cached_kv = rearrange(cached_kv, 'b ... n d -> b n ... d')
            small_cached_kv = rearrange(small_cached_kv, 'b ... n d -> b n ... d')

            cached_kv = cached_kv[batch_range, seq_offset_indices]
            small_cached_kv = small_cached_kv[batch_range, seq_offset_indices]

            cached_kv = rearrange(cached_kv, 'b n ... d -> b ... n d')
            small_cached_kv = rearrange(small_cached_kv, 'b n ... d -> b ... n d')

            small_cached_kv = small_cached_kv[..., :-1, :]
        
        else:
            # if batch size of 1, just slice to max_seq_len
            out = out[..., :max_seq_len]
            cached_kv = cached_kv[..., :max_seq_len, :]  #  accept the tokens before the rejection
            small_cached_kv = small_cached_kv[..., :max_seq_len, :] # accept the tokens before the rejection

        cache = (cached_kv, None)
        small_cache = (small_cached_kv, None)

        # sample the additional token, one of the tricks in the paper to better bound the worst case

        next_token = torch.multinomial(prob_next, 1)

        out = torch.cat((out, next_token), dim = -1)
        seq_lens += 1

        

    #### now left align ####
    # after finishing the loop, left align the tokens
    # seq_len_range + num_pad_left[..., None] is used to index the tokens
    num_pad_left = out.shape[-1] - seq_lens
    max_pad_left = num_pad_left.amax()
    out = F.pad(out, (0, max_pad_left), value = pad_id)

    seq_len_range = torch.arange(seq_len, device = device, dtype = torch.long)
    out = out[batch_range, seq_len_range + num_pad_left[..., None]]

    return out[..., prompt_seq_len:], total_accepted / num_steps
    # return the generated tokens after the prompt and the acceptance rate

@torch.no_grad()
def speculative_decoding(
    net: Module,
    small_net: Module,
    prompt: Tensor,
    seq_len: int,
    gamma: int = 5,
    temperature = 1.,
    filter_thres = 0.9,
    lenience = 1.,
    pad_id = 0
):
    """
    eq. algorithm 1 in paper https://arxiv.org/abs/2211.17192
    """

    batch, prompt_seq_len, out, device = *prompt.shape, prompt.clone(), prompt.device
    sample_num_times = max(0, seq_len - prompt_seq_len)

    cache = None
    small_cache = None

    num_steps = 0
    total_accepted = 0

    batch_range = torch.arange(batch, device = device, dtype = torch.long)[..., None]
    seq_lens = torch.full((batch,), prompt_seq_len, device = device, dtype = torch.long)

    while (seq_lens < seq_len).any():

        # predict with smaller network

        all_small_logits = []
        q_sampled_out = []

        for _ in range(gamma):
            small_logits, small_cache = small_net(
                out,
                seq_start_pos = out.shape[-1] - seq_lens,
                cache = small_cache,
                return_cache = True
            )

            small_logits = small_logits[:, -1]

            small_logits = top_k(small_logits, thres = filter_thres)
            all_small_logits.append(small_logits)

            sample = gumbel_sample(small_logits, temperature = temperature, dim = -1)
            out = torch.cat((out, sample[..., None]), dim = -1)
            seq_lens += 1

            q_sampled_out.append(rearrange(sample, 'b -> b 1 1'))

        q_sampled_out = torch.cat(q_sampled_out, dim = -2)
        small_logits = torch.stack(all_small_logits, dim = -2)

        # verify with larger network

        logits, cache = net(
            out,
            seq_start_pos = out.shape[-1] - seq_lens,
            cache = cache,
            return_cache = True
        )

        logits = logits[..., -(gamma + 1):, :]
        logits = top_k(logits, thres = filter_thres)

        # prob and prob of small model (p(x) and q(x) in algorithm 1)

        prob = safe_div(logits, temperature).softmax(dim = -1)
        small_prob = safe_div(small_logits, temperature).softmax(dim = -1)

        p, prob_next = prob[:, :-1], prob[:, -1]

        p = p.gather(-1, q_sampled_out)
        q = small_prob.gather(-1, q_sampled_out) * lenience

        p, q = [rearrange(t, 'b n 1 -> b n') for t in (p, q)]

        r = random_uniform = torch.zeros_like(q).float().uniform_(0, 1)

        accepted = find_first_true_index(r > (p / q))

        total_accepted += accepted.float().mean()
        num_steps += 1

        num_rejected = gamma - accepted
        has_rejected = num_rejected > 0

        accepted = rearrange(accepted, 'b -> b 1')
        accepted.clamp_(max = gamma - 1)
        adjusted_prob = F.relu(prob[batch_range, accepted] - small_prob[batch_range, accepted])
        adjusted_prob = adjusted_prob / adjusted_prob.sum(dim = -1, keepdim = True)
        adjusted_prob = rearrange(adjusted_prob, 'b 1 d -> b d')

        prob_next = torch.where(
            rearrange(has_rejected, '... -> ... 1'),
            adjusted_prob,
            prob_next
        )

        # do a bunch of slicing and align everything to the right, including kv caches

        seq_lens -= num_rejected
        max_seq_len = seq_lens.amax()
        curr_len = out.shape[-1]

        seq_arange = torch.arange(max_seq_len, device = device, dtype = torch.long) + (curr_len - max_seq_len)
        seq_offset_indices = seq_arange - num_rejected[..., None]

        cached_kv, _ = cache
        small_cached_kv, _ = small_cache

        if batch > 1:
            small_cached_kv = F.pad(small_cached_kv, (0, 0, 0, 1))  # account for small model being a token behind

            out = out[batch_range, seq_offset_indices]

            cached_kv = rearrange(cached_kv, 'b ... n d -> b n ... d')
            small_cached_kv = rearrange(small_cached_kv, 'b ... n d -> b n ... d')

            cached_kv = cached_kv[batch_range, seq_offset_indices]
            small_cached_kv = small_cached_kv[batch_range, seq_offset_indices]

            cached_kv = rearrange(cached_kv, 'b n ... d -> b ... n d')
            small_cached_kv = rearrange(small_cached_kv, 'b n ... d -> b ... n d')

            small_cached_kv = small_cached_kv[..., :-1, :]
        else:
            # if batch size of 1, just slice to max_seq_len
            out = out[..., :max_seq_len]
            cached_kv = cached_kv[..., :max_seq_len, :]
            small_cached_kv = small_cached_kv[..., :max_seq_len, :]

        cache = (cached_kv, None)
        small_cache = (small_cached_kv, None)

        # sample the additional token, one of the tricks in the paper to better bound the worst case

        next_token = torch.multinomial(prob_next, 1)

        out = torch.cat((out, next_token), dim = -1)
        seq_lens += 1

    # now left align

    num_pad_left = out.shape[-1] - seq_lens
    max_pad_left = num_pad_left.amax()
    out = F.pad(out, (0, max_pad_left), value = pad_id)

    seq_len_range = torch.arange(seq_len, device = device, dtype = torch.long)
    out = out[batch_range, seq_len_range + num_pad_left[..., None]]

    return out[..., prompt_seq_len:], total_accepted / num_steps


# attention and feedforward

class CausalAttention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        dim_inner = dim_head * heads

        self.norm = nn.RMSNorm(dim)

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        x,
        cache = None,
        context_mask = None,
        rotary_emb = None
    ):
        h, device = self.heads, x.device

        x = self.norm(x)

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h)

        if exists(cache):
            ck, cv = cache.unbind(dim = 1)
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

        cached_kv = torch.stack((k, v), dim = 1)

        if exists(rotary_emb):
            q = apply_rotary_pos_emb(rotary_emb, q)
            k = apply_rotary_pos_emb(rotary_emb, k)

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)

        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        if exists(context_mask):
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~context_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, cached_kv

def FeedForward(dim, mult = 4):
    dim_inner = dim * mult
    return nn.Sequential(
        nn.RMSNorm(dim),
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

# main class

class Decoder(Module):
    def __init__(
        self,
        *,
        num_tokens,  # vocabulary size
        dim, # dimension of each token embedding
        depth, # number of transformer blocks
        heads = 8, # number of attention heads
        dim_head = 64, # dimension of each attention head
        ff_mult = 4, # feedforward inner dimension multiplier (dim * ff_mult) = 
        ignore_index = -1,
        early_exit_layer = None,  # Can early exit on certain layer
        early_exit_extra_transformer_blocks = 0,
        detach_early_exit_hiddens = False
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim) # Turns integer token IDs into dense vectors of size dim.

      
        self.layers = ModuleList([])

        self.rotary_emb = RotaryEmbedding(dim = dim_head)

        for _ in range(depth):
            self.layers.append(ModuleList([
                CausalAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ])) ## attention + FF blck => one transformer block

        self.to_logits = nn.Sequential(
            nn.RMSNorm(dim), # Normalization (Root Mean Square)
            nn.Linear(dim, num_tokens, bias = False) # produce logits for each token
        )

        self.detach_early_exit_hiddens = detach_early_exit_hiddens
        self.early_exit_layer = early_exit_layer
        self.to_early_exit_logits = None
        self.early_exit_transformer_blocks = ModuleList([])

        if exists(early_exit_layer):
            for _ in range(early_exit_extra_transformer_blocks):
                self.early_exit_transformer_blocks.append(ModuleList([
                    CausalAttention(dim = dim, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                    FeedForward(dim = dim, mult = ff_mult)
                ]))

            self.to_early_exit_logits = nn.Sequential(
                nn.RMSNorm(dim),
                nn.Linear(dim, num_tokens, bias = False)
            )

        self.ignore_index = ignore_index

    def forward(
        self,
        x, # input tokens [batch, seq_len]
        return_loss = False, # return loss if True
        return_cache = False, # return cache if True
        seq_start_pos = None, # start position of the sequence (left padding)
        cache = None,
        early_exit_cache = None,
        return_early_exit_only = False,
        start_from_early_exit_hiddens = False
    ):
        
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]
            # if return_loss is True, then the last token is removed from the input and stored in labels
            # x : input goes from 0 to N-1
            # labels : input goes from 1 to N
            # predict the next token given the previous tokens

        x = self.token_emb(x) # get the token embeddings for the input tokens

        # handle seq start pos offset

        self_attn_kv_mask = None
        if exists(seq_start_pos): # if seq_start_pos is provided
            batch, seq_len = x.shape[:2] # get the batch size and sequence length
            seq_range = torch.arange(seq_len, device = x.device, dtype = torch.long) # create a tensor with values from 0 to seq_len (to be used as positional encoding)
            self_attn_kv_mask = seq_range >= seq_start_pos[..., None] # to make mask so that the model doesn't look at the future tokens

        # relative positional encoding

        rotary_emb = self.rotary_emb(x.shape[-2])

        # setup cache

        new_cached_kvs = []

        cache_kvs = cache_embeds = None

        if exists(cache):
            cache_kvs, cache_embeds = cache

        if exists(cache_kvs):
            iter_cache_kvs = iter(cache_kvs.unbind(dim = 1))
        else:
            iter_cache_kvs = iter([])

        # handle if previous cached embedding layer from early exit layer passed in

        layers = self.layers

        if start_from_early_exit_hiddens:
            assert not return_early_exit_only and exists(early_exit_cache)
            early_exit_layer_index = self.early_exit_layer

            early_cache_kvs, cache_embeds = early_exit_cache

            cache_embeds_len = cache_embeds.shape[-2]

            assert cache_embeds_len <= x.shape[-2]

            early_exit_layers, layers = layers[:early_exit_layer_index], layers[early_exit_layer_index:]
            x = x[:, cache_embeds_len:]

            if exists(early_cache_kvs):
                iter_early_cache_kvs = iter(early_cache_kvs.unbind(dim = 1))
            else:
                iter_early_cache_kvs = iter([])

            for ind, (attn, ff) in enumerate(early_exit_layers):
                residual = x
                attn_out, cached_kv = attn(x, context_mask = self_attn_kv_mask, rotary_emb = rotary_emb, cache = next(iter_early_cache_kvs, None))
                x = residual + attn_out

                new_cached_kvs.append(cached_kv)

                x = ff(x) + x

            x = torch.cat((cache_embeds, x), dim = -2)

        # if cache passed in, just use the last token

        if exists(cache):
            num_tokens_keep = x.shape[-2] - cache_kvs.shape[-2]
            x = x[:, -num_tokens_keep:]

        early_exit_hiddens = None

        # main transformer body

        for ind, (attn, ff) in enumerate(layers):
            layer = ind + 1

            residual = x
            attn_out, cached_kv = attn(x, rotary_emb = rotary_emb, cache = next(iter_cache_kvs, None))
            x = residual + attn_out # add the residual connection

            new_cached_kvs.append(cached_kv)

            x = ff(x) + x # add the another residual connection

            if layer == self.early_exit_layer:
                early_exit_hiddens = x

                if self.detach_early_exit_hiddens:
                    early_exit_hiddens = early_exit_hiddens.detach()

                for early_exit_attn, early_exit_ff in self.early_exit_transformer_blocks:
                    residual = x
                    attn_out, cached_kv = early_exit_attn(x, rotary_emb = rotary_emb, cache = next(iter_cache_kvs, None))
                    x = residual + attn_out

                    new_cached_kvs.append(cached_kv)

                    x = early_exit_ff(x) + x

                if return_early_exit_only:
                    break

        new_cached_kvs = torch.stack(new_cached_kvs, dim = 1)

        to_logits = self.to_logits if not return_early_exit_only else self.to_early_exit_logits
        # if return_early_exit_only is True, then use the to_early_exit_logits function
        logits = to_logits(x) # logits for each token
                            # [batch, seq_len, num_tokens==classes]

        if not return_loss:
            if not return_cache:
                return logits

            if exists(cache_embeds):
                x = torch.cat((cache_embeds, x), dim = -2)

            return logits, Cache(new_cached_kvs, x)

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'), # batch, classes, seq_len
            labels,
            ignore_index = self.ignore_index
        )  # compute the cross entropy loss

        if not exists(self.to_early_exit_logits):
            return loss

        early_exit_logits = self.to_early_exit_logits(early_exit_hiddens)

        early_exit_loss = F.cross_entropy(
            rearrange(early_exit_logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.ignore_index
        )

        return loss, early_exit_loss
