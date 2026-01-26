import regex as re
import multiprocessing
import collections
from typing import BinaryIO, Iterable, Iterator, Any
import os
import tqdm
from concurrent.futures import ProcessPoolExecutor
import json


GPT2_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        
        # Create inverse maps for efficiency
        self.encoder = {v: k for k, v in self.vocab.items()}
        
        # Merges map for fast lookup: (bytes, bytes) -> merged_bytes
        # Store ranks
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        
        # Regex for pre-tokenization
        self.pat = re.compile(GPT2_SPLIT_PATTERN)
        
        # Special tokens handling
        self.special_token_to_id = {}
        if self.special_tokens:
            for st in self.special_tokens:
                st_bytes = st.encode("utf-8")
                # Special tokens must be in vocab. 
                # If they are not, we can't really encode them as single IDs unless we add them?
                # The train_bpe returns a vocab WITH special tokens. 
                # So we expect them to be in self.encoder.
                if st_bytes in self.encoder:
                    self.special_token_to_id[st] = self.encoder[st_bytes]

        # Cache for simple word encoding
        self.cache = {}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            raw_vocab = json.load(f)
            # JSON keys are always strings.
            # Int keys become strings in JSON.
            # Values: we expect to serialize bytes.
            # If we used latin-1 string for bytes in JSON:
            vocab = {int(k): v.encode('latin-1') for k, v in raw_vocab.items()}
            
        with open(merges_filepath, 'r', encoding='utf-8') as f:
             # Just assume list of lists of ints or strings?
             # Let's settle on a format. 
             # I will use: merges is list of [latin1_str, latin1_str]
             raw_merges = json.load(f)
             merges = [tuple(p.encode('latin-1') for p in pair) for pair in raw_merges]
             
        return cls(vocab, merges, special_tokens)

    def _bpe(self, token_bytes: bytes) -> list[int]:
        if token_bytes in self.cache:
            return self.cache[token_bytes]
            
        # Initial split: bytes
        word = [bytes([b]) for b in token_bytes]
        
        if not self.merges:
            # If no merges, just bytes
            ids = [self.encoder[b] for b in word]
            self.cache[token_bytes] = ids
            return ids

        while len(word) > 1:
            # Find the pair with lowest rank
            min_rank = float('inf')
            min_pair = None
            min_idx = -1
            
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                rank = self.bpe_ranks.get(pair, float('inf'))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair
                    min_idx = i
            
            if min_pair is None or min_rank == float('inf'):
                break
                
            # Merge
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == min_pair:
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            
        ids = [self.encoder[b] for b in word if b in self.encoder] # Safety check
        self.cache[token_bytes] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self._encode_text(text)
            
        # Handle special tokens by splitting
        escaped_special = [re.escape(st) for st in self.special_tokens]
        pattern = '|'.join(escaped_special)
        if not pattern:
             return self._encode_text(text)
             
        # Split but keep delimiters
        parts = re.split(f'({pattern})', text)
        ids = []
        for part in parts:
            if part in self.special_token_to_id:
                ids.append(self.special_token_to_id[part])
            elif part:
                ids.extend(self._encode_text(part))
        return ids

    def _encode_text(self, text: str) -> list[int]:
        ids = []
        for match in self.pat.finditer(text):
            token_text = match.group()
            token_bytes = token_text.encode('utf-8')
            ids.extend(self._bpe(token_bytes))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        byte_parts = []
        for i in ids:
            if i in self.vocab:
                part = self.vocab[i]
                byte_parts.append(part)
        
        combined = b"".join(byte_parts)
        return combined.decode('utf-8', errors='replace')

# --- Training Logic ---

def _process_chunk_for_counts(chunk_data):
    # chunk_data is (filepath, start, end, special_tokens)
    path, start, end, special_tokens = chunk_data
    
    local_counts = collections.Counter()
    pat = re.compile(GPT2_SPLIT_PATTERN)
    
    with open(path, 'rb') as f:
        f.seek(start)
        # Read raw bytes
        text_bytes = f.read(end - start)
    
    # Check if empty
    if not text_bytes:
        return local_counts

    text = text_bytes.decode('utf-8', errors='replace')
    
    if special_tokens:
        escaped_special = [re.escape(st) for st in special_tokens]
        pattern = '|'.join(escaped_special)
        parts = re.split(pattern, text)
    else:
        parts = [text]
        
    for part in parts:
        if not part:
            continue
        # Apply GPT-2 regex
        for match in pat.finditer(part):
            token_bytes = match.group().encode('utf-8')
            local_counts[token_bytes] += 1
            
    return local_counts

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Import here to avoid circular or top-level issues if running not as package
    try:
        from .pretokenization import find_chunk_boundaries
    except ImportError:
        # Fallback if running as script
        from cs336_basics.pretokenization import find_chunk_boundaries
    
    # 0. Validate special tokens
    special_tokens = list(set(special_tokens)) # unique
    
    # 1. Chunking
    split_token = special_tokens[0].encode('utf-8') if special_tokens else None
    
    num_processes = max(1, multiprocessing.cpu_count())
    
    boundaries = []
    with open(input_path, 'rb') as f:
        if split_token:
            boundaries = find_chunk_boundaries(f, num_processes, split_token)
        else:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                boundaries = [0, 0]
            else:
                chunk_size = size // num_processes
                boundaries = [i * chunk_size for i in range(num_processes)] + [size]
                boundaries = sorted(list(set(boundaries)))

    # Prepare args
    ranges = []
    for i in range(len(boundaries)-1):
        if boundaries[i] != boundaries[i+1]:
            ranges.append((input_path, boundaries[i], boundaries[i+1], special_tokens))
        
    word_counts = collections.Counter()
    
    if ranges:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            for result in executor.map(_process_chunk_for_counts, ranges):
                word_counts.update(result)
    
    # 2. Iterative Merging
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    
    # Add special tokens to vocab
    for st in special_tokens:
        st_bytes = st.encode('utf-8')
        # Check if already in vocab (it shouldn't be effectively, unless st is single byte)
        # BPE special tokens are usually treated as single entities.
        # We assign them new IDs.
        vocab[next_id] = st_bytes
        next_id += 1
        
    num_merges = vocab_size - len(vocab)
    splits = {word: [bytes([b]) for b in word] for word in word_counts}
    merges = []
    
    # Optimized stats
    index = collections.defaultdict(set)
    stats = collections.defaultdict(int)
    
    for word, freq in word_counts.items():
        split = splits[word]
        for i in range(len(split) - 1):
            pair = (split[i], split[i+1])
            index[pair].add(word)
            stats[pair] += freq

    for _ in tqdm.tqdm(range(num_merges), desc="Training BPE"):
        if not stats:
            break
            
        best_pair = max(stats.items(), key=lambda x: (x[1], x[0]))[0]
        new_token = best_pair[0] + best_pair[1]
        
        merges.append(best_pair)
        vocab[next_id] = new_token
        next_id += 1
        
        words_to_update = list(index[best_pair])
        
        for word in words_to_update:
            freq = word_counts[word]
            split = splits[word]

            # Reconstruct the new split
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == best_pair[0] and split[i+1] == best_pair[1]:
                    new_split.append(new_token)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            
            # Compute differences in pair counts for this word
            # We use local counters to handle multiple occurrences correctly
            old_pair_counts = collections.Counter()
            for k in range(len(split) - 1):
                old_pair_counts[(split[k], split[k+1])] += 1
                
            new_pair_counts = collections.Counter()
            for k in range(len(new_split) - 1):
                new_pair_counts[(new_split[k], new_split[k+1])] += 1
            
            # Update global stats
            for pair, count in old_pair_counts.items():
                stats[pair] -= count * freq
                if stats[pair] == 0:
                    del stats[pair]
            
            for pair, count in new_pair_counts.items():
                stats[pair] += count * freq
            
            # Update inverted index
            # If a pair was in old but not in new, we might need to remove this word from its index
            for pair in old_pair_counts:
                if pair not in new_pair_counts:
                    index[pair].discard(word)
                    if not index[pair]:
                        del index[pair]
            
            # If a pair is in new but was not in old, add this word to its index
            for pair in new_pair_counts:
                if pair not in old_pair_counts:
                    index[pair].add(word)
            
            splits[word] = new_split

        # Cleanup index for best_pair since it no longer exists in any word
        # (It should have been removed by the logic above, but safety check)
        if best_pair in index:
            del index[best_pair]
            
    return vocab, merges
