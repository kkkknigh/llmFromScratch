import regex as re
import multiprocessing
import collections
from typing import BinaryIO, Iterable, Iterator, Any
import os
import tqdm
from concurrent.futures import ProcessPoolExecutor
import json
import heapq


GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        
        self.encoder = {v: k for k, v in self.vocab.items()}
        
        # Merges map for fast lookup: (bytes, bytes) -> merged_bytes
        # Store ranks
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        
        # 预分词正则表达式
        self.pat = re.compile(GPT2_SPLIT_PATTERN)
        
        # 特殊标记
        self.special_token_to_id = {}
        if self.special_tokens:
            for st in self.special_tokens:
                st_bytes = st.encode("utf-8")
                if st_bytes in self.encoder:
                    self.special_token_to_id[st] = self.encoder[st_bytes]

        self.cache = {}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            raw_vocab = json.load(f)
            vocab = {int(k): v.encode('latin-1') for k, v in raw_vocab.items()}
            
        with open(merges_filepath, 'r', encoding='utf-8') as f:
             raw_merges = json.load(f)
             merges = [tuple(p.encode('latin-1') for p in pair) for pair in raw_merges]
             
        return cls(vocab, merges, special_tokens)

    def _bpe(self, token_bytes: bytes) -> list[int]:
        if token_bytes in self.cache:
            return self.cache[token_bytes]
            
        word = [bytes([b]) for b in token_bytes]
        
        if not self.merges:
            ids = [self.encoder[b] for b in word]
            self.cache[token_bytes] = ids
            return ids

        while len(word) > 1:
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
            
        ids = [self.encoder[b] for b in word if b in self.encoder] 
        self.cache[token_bytes] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self._encode_text(text)
            
        # 分割处理特殊标记
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True) # 从大到小分割
        escaped_special = [re.escape(st) for st in sorted_special_tokens]
        pattern = '|'.join(escaped_special)
        if not pattern:
             return self._encode_text(text)
             
        # 分割（保留终止符）
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


def _process_chunk_for_counts(chunk_data):
    # chunk_data: (filepath, start, end, special_tokens)
    # 对[start,end)预分词
    path, start, end, special_tokens = chunk_data
    
    local_counts = collections.Counter()
    pat = re.compile(GPT2_SPLIT_PATTERN)
    
    with open(path, 'rb') as f:
        f.seek(start)
        text_bytes = f.read(end - start)
    
    if not text_bytes:
        return local_counts

    text = text_bytes.decode('utf-8', errors='replace') # 字节转为字符
    # 按照特殊标记分割
    if special_tokens:
        escaped_special = [re.escape(st) for st in special_tokens] # 元字符转义
        pattern = '|'.join(escaped_special) # 特殊字符安全拼接
        parts = re.split(pattern, text)  # 分割
    else:
        parts = [text]
        
    # O(n^2)
    cache_local = {} # 节约反复编码开销【空间换时间】
    for part in parts: 
        if not part:
            continue
        for match in pat.finditer(part):
            token_text = match.group()
            if token_text in cache_local:
                token_bytes = cache_local[token_text]
            else:
                token_bytes = token_text.encode('utf-8')
                cache_local[token_text] = token_bytes
            local_counts[token_bytes] += 1 
            
    return local_counts

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    try:
        from .pretokenization import find_chunk_boundaries
    except ImportError:
        from cs336_basics.pretokenization import find_chunk_boundaries
    
    # 0. 特殊标记去重
    special_tokens = list(set(special_tokens)) 

    # 1. 根据第一个特殊标记分块
    split_token = special_tokens[0].encode('utf-8') if special_tokens else None
    
    num_processes = max(1, multiprocessing.cpu_count()) 
    
    boundaries = []
    with open(input_path, 'rb') as f:
        if split_token:
            boundaries = find_chunk_boundaries(f, num_processes, split_token)
        else:
            # 无特殊标记词，直接均分
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                boundaries = [0, 0]
            else:
                chunk_size = size // num_processes
                boundaries = [i * chunk_size for i in range(num_processes)] + [size]
                boundaries = sorted(list(set(boundaries)))
    # 分块预分词参数构造
    ranges = []
    for i in range(len(boundaries)-1):
        if boundaries[i] != boundaries[i+1]:
            ranges.append((input_path, boundaries[i], boundaries[i+1], special_tokens))
        
    word_counts = collections.Counter()
    # 2. 并行化预分词
    if ranges:
        ctx = multiprocessing.get_context("spawn") # 避免fork
        with ProcessPoolExecutor(max_workers=num_processes, mp_context=ctx) as executor:
            for result in executor.map(_process_chunk_for_counts, ranges):
                word_counts.update(result)
    
    # 初始化词汇表 映射：intID->字节对
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    
    # 词汇表添加特殊标记
    for st in special_tokens:
        st_bytes = st.encode('utf-8')
        vocab[next_id] = st_bytes
        next_id += 1
        
    num_merges = vocab_size - len(vocab) # 合并次数
    splits = {word: [bytes([b]) for b in word] for word in word_counts} # 词的字节表示->词的字节列表
    merges = []
    
    index = collections.defaultdict(set) # 相邻字节对pair->包含该字节对的词集合
    stats = collections.defaultdict(int) # byte-pair->出现频次
    
    # 初始化字节对频次计数
    for word, freq in word_counts.items():
        split = splits[word]
        for i in range(len(split) - 1):
            pair = (split[i], split[i+1])
            index[pair].add(word)
            stats[pair] += freq

    # pair的包装辅助定义（辅助堆按照频次->字符串降序排序）
    class MaxHeapPair:
        __slots__ = ('pair',)
        def __init__(self, pair):
            self.pair = pair
        def __lt__(self, other):
            return self.pair > other.pair
        def __eq__(self, other):
            return self.pair == other.pair

    # 堆初始化
    heap = []
    for pair, freq in stats.items():
        heapq.heappush(heap, (-freq, MaxHeapPair(pair)))

    for _ in tqdm.tqdm(range(num_merges), desc="Training BPE"):
        best_pair = None
        # 懒删除
        while heap:
            neg_freq, wrapped_pair = heapq.heappop(heap)
            pair = wrapped_pair.pair
            if stats.get(pair) == -neg_freq:  # 有效性验证
                best_pair = pair 
                break 
        
        if best_pair is None:
            break
            
        new_token = best_pair[0] + best_pair[1] 
        
        merges.append(best_pair)
        vocab[next_id] = new_token
        next_id += 1

        # 更新词频次统计+映射
        words_to_update = list(index[best_pair]) 
        
        for word in words_to_update:
            freq = word_counts[word]
            split = splits[word]

            # 重新分割单词
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == best_pair[0] and split[i+1] == best_pair[1]:
                    new_split.append(new_token)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            
            # 重新数byte-pair词频
            old_pair_counts = collections.Counter()
            for k in range(len(split) - 1):
                old_pair_counts[(split[k], split[k+1])] += 1
                
            new_pair_counts = collections.Counter()
            for k in range(len(new_split) - 1):
                new_pair_counts[(new_split[k], new_split[k+1])] += 1
            
            for pair, count in old_pair_counts.items():
                stats[pair] -= count * freq
                if stats[pair] == 0:
                    del stats[pair]
                else:
                    heapq.heappush(heap, (-stats[pair], MaxHeapPair(pair)))
            
            for pair, count in new_pair_counts.items():
                stats[pair] += count * freq
                heapq.heappush(heap, (-stats[pair], MaxHeapPair(pair)))
            
            # 更新映射
            for pair in old_pair_counts:
                if pair not in new_pair_counts:
                    index[pair].discard(word)
                    if not index[pair]:
                        del index[pair]
            
            for pair in new_pair_counts:
                if pair not in old_pair_counts:
                    index[pair].add(word)
            
            splits[word] = new_split

        if best_pair in index:
            del index[best_pair]
            
    return vocab, merges
