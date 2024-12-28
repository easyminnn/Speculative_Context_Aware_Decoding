import gzip
import random
import tqdm
import numpy as np
import time
import wandb
from functools import wraps, partial

import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn import DataParallel
from torch.cuda import synchronize, Event
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
# ---------------- NEW: Additional libraries for metrics ----------------
import mauve
#from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
# ---------------------------------------------------------------------
from huggingface_hub import login
login(token = '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')

from accelerate import Accelerator 
accelerator = Accelerator()

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


torch.cuda.set_device(0)

timer = partial(Event, enable_timing = True)

from speculative_context_aware_decoding import (
    base_decoding,
    speculative_context_aware_decoding, 
    speculative_decoding
)

#wandb.init(project="speculative-context-aware-decoding")
wandb.init(mode="disabled")

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRAD_ACCUM_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
PRIME_LENGTH = 128
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 512
GAMMA = 5

DEVICE_STR = 'cuda' if torch.cuda.is_available() else 'cpu'

# helpers

####################################

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

def benchmark(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        start_event = timer()
        end_event = timer()
        start_event.record()

        out = fn(*args, **kwargs)

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        return out, elapsed_time_ms
    return inner

# NEW: Pre-load our SimCSE model for coherence
#simcse_model = SentenceTransformer("princeton-nlp/sup-simcse-bert-base-uncased")

# NEW: Helper function for Diversity
def compute_diversity(text, n_min=2, n_max=4):
    """
    text: the generated text (string).
    Returns a float in [0,1] (usually) – product of (# unique n-grams / # total n-grams) for n in [2..4].
    """
    # For enwik8, let's treat each character as a "token"
    tokens = list(text)
    diversity_product = 1.0

    for n in range(n_min, n_max + 1):
        n_grams = []
        for i in range(len(tokens) - n + 1):
            n_gram = tuple(tokens[i : i + n])
            n_grams.append(n_gram)
        
        total_ngrams = len(n_grams)
        unique_ngrams = len(set(n_grams))

        ratio = 0.0 if total_ngrams == 0 else unique_ngrams / total_ngrams
        diversity_product *= ratio

    return diversity_product

# NEW: Helper function for Coherence
"""
def compute_coherence(prompt_text, continuation_text):
    
    prompt_text: string
    continuation_text: string
    Returns a float: the cosine similarity of SimCSE embeddings
    
    emb_prompt = simcse_model.encode([prompt_text], convert_to_tensor=True)
    emb_cont   = simcse_model.encode([continuation_text], convert_to_tensor=True)
    sim = F.cosine_similarity(emb_prompt, emb_cont)
    return sim.item()  # float
"""
# NEW: Helper function for MAUVE

def compute_mauve_score(generated_texts, reference_texts, max_text_length=400):
    """
    generated_texts: list of strings
    reference_texts: list of strings
    Returns: mauve_score (float)
    """
    mauve_result = mauve.compute_mauve(
        p_text=generated_texts,
        q_text=reference_texts,
        device_id=0 if torch.cuda.is_available() else -1,
        max_text_length=max_text_length,
        verbose=False
    )
    return mauve_result.mauve




# instantiate transformer

device = torch.device(DEVICE_STR)

# model = Decoder(
#     num_tokens = 256,
#     dim = 512,
#     depth = 10
# ).to(device)
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype = torch.float16,
    device_map = "auto",
    cache_dir = "./cache/model",
    )

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

model = FSDP(
    model
    )

# small model

# small_model = Decoder(
#     num_tokens = 256,
#     dim = 512,
#     depth = 3
# ).to(device)

small_model_name = "JackFram/llama-160m"
small_tokenizer = AutoTokenizer.from_pretrained(small_model_name)

small_model = AutoModelForCausalLM.from_pretrained(
        small_model_name,
        torch_dtype = torch.float16,
        device_map = "auto",
        cache_dir = "./cache/small_model",

    )

if small_tokenizer.pad_token is None:
    small_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    small_model.resize_token_embeddings(len(small_tokenizer))

small_model = DataParallel(small_model, device_ids = [0, 1])

# prepare enwik8 data

with gzip.open("./data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.to(accelerator.device)

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))

# optimizer

optim = Adam(model.parameters(), lr = LEARNING_RATE)
small_optim = Adam(small_model.parameters(), lr = LEARNING_RATE)

# training

for i, text_batch in enumerate(tqdm.tqdm(train_loader, total = NUM_BATCHES, mininterval = 10.0, desc = "training")):
    model.train()
    small_model.train()

    text_list = [decode_tokens(seq) for seq in text_batch]


    encoding = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=SEQ_LEN).to(device)
    input_ids = encoding["input_ids"].to(accelerator.device)
    attention_mask = encoding["attention_mask"].to(accelerator.device)

    labels= input_ids.clone()

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss

    small_encoding = small_tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=SEQ_LEN).to(device)
    small_input_ids = small_encoding["input_ids"].to(accelerator.device)
    small_labels = small_input_ids.clone()

    small_outputs = small_model(small_input_ids, labels=small_labels)
    small_loss = small_outputs.loss

    loss.sum().backward()
    small_loss.sum().backward()

    print(f"training loss: {loss.item():.3f}")
    print(f"training small loss: {small_loss.item():.3f}")

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    torch.nn.utils.clip_grad_norm_(small_model.parameters(), 0.5)

    optim.step()
    optim.zero_grad()

    small_optim.step()
    small_optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        small_model.eval()

        with torch.no_grad():
            valid_data = next(val_loader)
            valid_encoding = tokenizer(valid_data, return_tensors="pt", padding=True, truncation=True, max_length=SEQ_LEN).to(device)

            valid_input_ids = valid_encoding["input_ids"].to(accelerator.device)
            valid_attention_mask = valid_encoding["attention_mask"].to(accelerator.device)
            valid_labels = valid_input_ids.clone()

            valid_outputs = model(valid_input_ids, attention_mask=valid_attention_mask, labels=valid_labels)

            valid_loss = valid_outputs.loss

            print(f"validation loss: {valid_loss.item():.3f}")

            small_valid_encoding = small_tokenizer(valid_data, return_tensors="pt", padding=True, truncation=True, max_length=SEQ_LEN).to(accelerator.device)
            small_valid_input_ids = small_valid_encoding["input_ids"].to(accelerator.device)
            small_valid_labels = small_valid_input_ids.clone()

            small_valid_outputs = small_model(small_valid_input_ids, labels=small_valid_labels)
            small_valid_loss = small_valid_outputs.loss

            print(f"validation small loss: {small_valid_loss.item():.3f}")


    if i % GENERATE_EVERY == 0:
        model.eval()
        small_model.eval()

        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        print(f"%s \n\n %s", (prime, "*" * 100))

        prompt = inp[None, ...]

        # base decoding
        sampled, base_decode_elapsed = benchmark(base_decoding)(model, prompt, GENERATE_LENGTH)
        # speculative context aware decoding
        (spec_context_decode_sampled, spec_context_decode_num_accepted), spec_context_decode_elapsed = benchmark(speculative_context_aware_decoding)(model, small_model, prompt, GENERATE_LENGTH, GAMMA)
        # speculative decoding 
        (spec_decode_sampled, num_accepted), spec_decode_elapsed = benchmark(speculative_decoding)(model, small_model, prompt, GENERATE_LENGTH, GAMMA)


        base_decode_output = decode_tokens(sampled[0])
        spec_decode_output = decode_tokens(spec_decode_sampled[0])
        spec_context_decode_output = decode_tokens(spec_context_decode_sampled[0])

        print("\nbase decoding:\n\n", base_decode_output, "\n")
        print(type(base_decode_output))
        print("\nspec decoding:\n\n", spec_decode_output, "\n")
        print("\nspec context decoding:\n\n", spec_context_decode_output, "\n")


        # -------------- NEW: Compute metrics --------------

        # 1) Diversity
        base_div = compute_diversity(base_decode_output)
        spec_div = compute_diversity(spec_decode_output)
        spec_context_div = compute_diversity(spec_context_decode_output)

        # 2) Coherence
        """
        coherence(prime, base_decode_output)
        spec_coh = compute_coherence(prime, spec_decode_output)
        spec_context_coh = compute_coherence(prime, spec_context_decode_output)

        print(f"Base decoding diversity: {base_div:.4f} | coherence: {base_coh:.4f}")
        print(f"Spec decoding diversity: {spec_div:.4f} | coherence: {spec_coh:.4f}")
        print(f"Spec context decoding diversity: {spec_context_div:.4f} | coherence: {spec_context_coh:.4f}")
         """
        
        # 3) MAUVE - Single-sample example
        # Normally you'd want many references and many generations. 
        # We'll just pick a random slice from val_dataset for reference_text:
        ref_idx = random.randint(0, len(val_dataset) - 1)
        reference_slice = val_dataset[ref_idx][:GENERATE_LENGTH]
        reference_text = decode_tokens(reference_slice)
        # We'll do a single-sample MAUVE for demonstration:
        mauve_base = compute_mauve_score([base_decode_output], [reference_text])
        print(f"MAUVE (base, single-sample): {mauve_base:.4f}")

        # -------------------------------------------------



        print(f'base decoding in: {base_decode_elapsed:.3f}ms\n')
        print(f'spec decoding in: {spec_decode_elapsed:.3f}ms\n')
        print(f'spec context decoding in: {spec_context_decode_elapsed:.3f}ms\n')
        print(f'average num accepted for spec context decoding : {spec_context_decode_num_accepted:.1f} / {GAMMA}\n')
        print(f'average num accepted for spec decoding: {num_accepted:.1f} / {GAMMA}\n')

        wandb.log({
            "base_decoding_time": base_decode_elapsed,
            "spec_decoding_time": spec_decode_elapsed,
            "spec_context_decoding_time": spec_context_decode_elapsed,

            "spec_decoding_avg_num_accepted": float((num_accepted) / GAMMA),
            "spec_context_decoding_avg_num_accepted": float(spec_context_decode_num_accepted/GAMMA),
            

            # NEW logs for metrics
            "base_diversity": base_div,
            "spec_diversity": spec_div,
            "spec_context_diversity": spec_context_div,

            # "base_coherence": base_coh,
            # "spec_coherence": spec_coh,
            # "spec_context_coherence": spec_context_coh,

            # Single-sample MAUVE example
            # "mauve_base_single": mauve_base

            })
        

wandb.finish()
    
