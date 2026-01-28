# CS336作业1（基础）：构建Transformer语言模型

# 1 作业概述

CS336团队 2025年春季

在本作业中，你将从头开始构建训练标准Transformer语言模型（LM）所需的所有组件，并进行训练

## 你将实现的内容

1. 字节对编码（BPE）分词器 (§2)
2. Transformer语言模型 (LM) (§3)
3. 交叉熵损失函数和AdamW优化器 (§4)
4. 训练循环，支持序列化和加载模型及优化器状态 (§5)

## 你将运行的内容

1. 在TinyStories数据集上训练BPE分词器。
2. 在数据集上运行你训练好的分词器，将其转换为整数ID序列。
3. 在TinyStories数据集上训练Transformer LM。
4. 使用训练好的Transformer LM生成样本并评估困惑度。
5. 在OpenWebText上训练模型，并将你获得的困惑度提交到

## 你可以使用什么

我们希望你从头开始构建这些组件。特别是，你不能使用torch.nn、torch.nn.functional或torch.optim中的任何定义，以下除外：

- torch.nn.Parameter
- torch.nn中的容器类（例如Module、ModuleList、Sequential等）¹
- torch.optim.Optimizer基类

你可以使用任何其他PyTorch定义。如果你想使用某个函数或类但不确定是否允许，请随时在Slack上提问。如有疑问，请考虑使用它是否会损害"从头开始"的精神

<footer>1参见PyTorch.org/docs/stable/nn.html#containers获取完整</footer>

## 关于AI工具的声明

允许使用ChatGPT等LLM进行低级编程问题或关于语言模型的高级概念问题，但直接使用它解决问题是

我们强烈建议你在完成作业时在IDE中禁用AI自动补全（例如Cursor Tab、GitHub CoPilot）（尽管非AI自动补全，例如函数名自动补全完全没问题）。我们发现AI自动补全使得更难深入参与

所有作业代码以及本说明可在GitHub上获取：

github -bas1cs

请git克隆仓库。如果有任何更新，我们会通知你，以便你可以git pull获取

## 代码结构

1. `cs336_basics/*`：这是你编写代码的地方。注意这里没有代码——你可以从头开始做你想做的任何事情！
2. `adapters.py`：你的代码必须具备一组功能。对于每个功能（例如缩放点积注意力），通过简单地调用你的代码来填写其实现（例如run_scaled_dot_product_attention）。注意：你对adapters.py的更改不应包含任何实质性逻辑；这是胶水代码。
3. `test_*.py`：这包含你必须通过的所有测试（例如test_scaled_dot_product_attention），它将调用adapters.py中定义的钩子。不要编辑测试

## 如何提交

你将向Gradescope提交以下文件：

- `writeup.pdf`：回答所有书面问题。请排版你的
- `code.zip`：包含你编写的所有代码

要提交到排行榜，请向以下地址提交PR：

github aderboard

参见排行榜仓库中的README.md获取详细提交

## 从哪里获取数据集

本作业将使用两个预处理数据集：TinyStories [Eldan和Li, 2023]和OpenWebText [Gokaslan等人, 2019]。这两个数据集都是单个大型纯文本文件。如果你与课程一起完成作业，你可以在任何非头节点的/data目录中找到这些文件

如果你在家学习，可以使用以下命令下载这些文件

# 低资源/降规模提示：初始化

在整个课程的作业讲义中，我们将为在GPU资源较少或没有GPU资源的情况下完成作业提供建议。例如，我们有时会建议你缩小数据集或模型规模，或解释如何在MacOS集成GPU或CPU上运行训练代码。你会在蓝色框中找到这些"低资源提示"（如此框所示）。即使你是注册学生并有权访问课程机器，这些提示也可能帮助你更快地迭代并节省时间，因此我们建议你阅读它们！

<footer>2</footer>

# 低资源/降规模提示：在Apple Silicon或CPU上完成作业1

使用工作人员提供的解决方案代码，我们可以在Apple M3 Max芯片（36GB RAM）上训练一个语言模型，在Metal GPU（MPS）上不到5分钟，使用CPU约30分钟，生成相当流畅的文本。如果这些术语对你不太熟悉，别担心！只要知道如果你有一台配置合理的现代笔记本电脑，并且你的实现正确且高效，你就能训练一个小型LM，生成质量尚可的简单儿童故事

在作业的后面部分，我们将解释如果你在CPU或

<footer>3</footer>

# 2 字节对编码（BPE）分词器

在本作业的第一部分，我们将训练和实现一个字节级字节对编码（BPE）分词器 [Sennrich等人, 2016, Wang等人, 2019]。特别是，我们将任意（Unicode）字符串表示为字节序列，并在此字节序列上训练我们的BPE分词器。稍后，我们将使用此分词器将文本（字符串）编码为标记（整数序列），用于语言建模。

## 2.1 Unicode标准

Unicode是一个将字符映射到整数码点的文本编码标准。截至Unicode 16.0（2024年9月发布），该标准定义了168种文字中的154,998个字符。例如，字符"s"的码点是115（通常记为U+0073，其中U+是常规前缀，0073是115的十六进制表示），字符"牛"的码点是29275。在Python中，你可以使用ord()函数将单个Unicode字符转换为其整数表示。chr()函数将整数Unicode码点转换为对应的字符串字符。

```python
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

### 问题（unicode1）：理解Unicode（1分）

**(a)** `chr(0)`返回什么Unicode字符？
> '\x00'

**(b)** 该字符的字符串表示形式`repr(chr(0))`与其打印表示形式有何不同？

> 字符串表示为'\x00'，打印形式为不可见字符

**(c)** 当该字符出现在文本中会发生什么？在Python解释器中尝试以下操作可能会有帮助，看看它是否符合你的预期：

```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```

~~~
>>> chr(0)
'\x00'
>>> print(chr(0))

>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
~~~

> \_str\_形式打印为不可见字符，\_repr_字符串形式表示为'\x00'

## 2.2 Unicode编码

虽然Unicode标准定义了从字符到码点（整数）的映射，但直接在Unicode码点上训练分词器是不切实际的，因为词汇表会过大（约150K项）且稀疏（因为许多字符相当罕见）。相反，我们将使用Unicode编码，它将Unicode字符转换为字节序列。Unicode标准本身定义了三种编码：UTF-8、UTF-16和UTF-32，其中UTF-8是互联网上的主导编码（超过98%的网页使用）。

要将Unicode字符串编码为UTF-8，我们可以使用Python中的`encode()`函数。要访问Python字节对象的底层字节值，我们可以遍历它（例如调用`list()`）。最后，我们可以使用`decode()`函数将UTF-8字节字符串解码为Unicode字符串。

```python
>>> test_string = "hello!こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello!\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # 获取编码字符串的字节值（0到255的整数）
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
>>> # 一个字节不一定对应一个Unicode字符！
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello!こんにちは!
```

通过将Unicode码点转换为字节序列（例如通过UTF-8编码），我们基本上是将码点序列（0到154,997范围内的整数）转换为字节值序列（0到255范围内的整数）。长度为256的字节词汇表更容易处理。使用字节级分词时，我们不必担心词汇表外标记，因为我们知道任何输入文本都可以表示为0到255范围内的整数序列。

### 问题（unicode2）：Unicode编码（3分）

**(a)** 与UTF-16或UTF-32相比，训练我们的分词器使用UTF-8编码字节有哪些原因？比较这些编码在各种输入字符串上的输出可能会有帮助。

> 变长编码，节约长度，ascii部分与ascii编码完全兼容

**(b)** 考虑以下（错误的）函数，它旨在将UTF-8字节字符串解码为Unicode字符串。为什么这个函数不正确？请提供一个会产生错误结果的输入字节字符串示例。

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

> ~~~
> >>> decode_utf8_bytes_to_str_wrong("我我我我我".encode("utf-8"))
>   File "<python-input-1>", line 1
>     decode_utf8_bytes_to_str_wrong("我我我我我".encode("utf-8"))
> IndentationError: unexpected indent
> ~~~
>
> 超出ascii码部分的内容由于不是单字节编码会出错

**(c)** 给出一个不解码为任何Unicode字符的双字节序列。

> C0 61,61可以单字节ascii表示

## 2.3 子词分词

虽然**字节级分词可以缓解词级分词器面临的词汇表外问题**，但将文本分词为字节会导致极长的输入序列。这会减慢模型训练速度，因为一个包含10个词的句子在词级语言模型中可能只有10个标记长，但在字符级模型中可能有50个或更多标记长（取决于词的长度）。处理这些较长的序列需要在模型的每一步进行更多计算。此外，在字节序列上进行语言建模很困难，因为更长的输入序列在模型中产生了长期依赖。

**子词分词**是词级分词器和字节级分词器之间的中点。注意，字节级分词器的词汇表有256个条目（字节值为0到255）。子词分词器用更大的词汇表大小来换取更好的输入字节序列压缩。例如，如果字节序列`b'the'`经常出现在我们的原始文本训练数据中，为其在词汇表中分配一个条目会将这个3标记序列减少为单个标记。

我们如何选择这些子词单元添加到词汇表中？Sennrich等人[2016]提出使用字节对编码（BPE；Gage, 1994），一种压缩算法，迭代地用单个新的未使用索引替换（"合并"）最频繁的字节对。注意，该算法向词汇表添加子词标记以最大化输入序列的压缩——如果一个词在输入文本中出现足够多次，它将被表示为单个子词标记。

通过BPE构建词汇表的子词分词器通常称为BPE分词器。在本作业中，我们将实现一个字节级BPE分词器，其词汇表项是字节或合并的字节序列，这在词汇表外处理和可管理的输入序列长度方面为我们提供了两全其美的效果。**构建BPE分词器词汇表**的过程被称为"训练"BPE分词器。

## 2.4 BPE分词器训练

BPE分词器训练过程包括三个主要步骤：

**词汇表初始化**：分词器词汇表是从字节字符串标记到整数ID的一对一映射。由于我们训练的是字节级BPE分词器，我们的初始词汇表就是所有字节的集合。由于有256种可能的字节值，我们的初始词汇表大小为256。

**预分词**：一旦你有了词汇表，原则上你可以统计字节在文本中相邻出现的频率，并从最频繁的字节对开始合并。然而，这在计算上非常昂贵，因为每次合并时我们都需要对整个语料库进行一次完整遍历。此外，直接在语料库上合并字节可能会导致仅在标点符号上不同的标记（例如`dog!` vs. `dog.`）。这些标记将获得完全不同的标记ID，即使它们可能具有很高的语义相似性（因为它们仅在标点符号上不同）。

为了避免这种情况，我们对语料库进行预分词。你可以将其视为对语料库进行的粗粒度分词，帮助我们统计字符对出现的频率。例如，单词`'text'`可能是一个出现10次的预分词。在这种情况下，当我们统计字符`'t'`和`'e'`相邻出现的频率时，我们会看到单词`'text'`中`'t'`和`'e'`相邻，因此可以将它们的计数增加10，而无需遍历整个语料库。由于我们训练的是字节级BPE模型，每个预分词都表示为UTF-8字节序列。

Sennrich等人[2016]的原始BPE实现通过简单地按空白字符分割（即`s.split("")`）进行预分词。相比之下，我们将使用基于正则表达式的预分词器（GPT-2使用；Radford等人, 2019），来自：

```python
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

交互式地使用此预分词器分割一些文本可能有助于更好地理解其行为：

```python
>>> # 需要`regex`包
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', 'text', 'that', 'i', "'ll", 'pre', '-', 'tokenize']
```

然而，在代码中使用它时，你应该使用`re.finditer`以避免在构建从预分词到其计数的映射时存储预分词后的单词。

**计算BPE合并**：现在我们已经将输入文本转换为预分词，并将每个预分词表示为UTF-8字节序列，我们可以**计算BPE合并（即训练BPE分词器）**。从高层次来看，BPE算法迭代地统计每一对字节，并识别频率最高的一对字节（"A","B"）。然后，这个最频繁字节对的每次出现都被合并，即替换为新的标记"AB"。这个新的合并标记被添加到词汇表中；因此，BPE训练后的最终词汇表大小等于初始词汇表大小（在我们的情况下为256）加上训练期间执行的BPE合并操作次数。为了在BPE训练期间提高效率，我们不考虑跨越预分词边界的字节对²。在计算合并时，通过**优先选择字典序更大的字节对**来确定性打破字节对频率的平局。例如，如果字节对`("A","B")`、`("A","C")`、`("B","ZZ")`和`("BA","A")`都具有最高频率，我们将合并`("BA","A")`：

```python
>>> max([("A","B"),("A","C"),("B","ZZ"),("BA","A")])
('BA','A')
```

**特殊标记**：通常，某些字符串（例如`<|endoftext|>`）用于编码元数据（例如文档之间的边界）。在编码文本时，通常希望将某些字符串视为"特殊标记"，这些标记**不应被拆分**为多个标记（即始终作为单个标记保留）。例如，序列结束字符串`<|endoftext|>`应始终保留为单个标记（即单个整数ID），这样我们就知道何时停止从语言模型生成。这些特殊标记必须添加到词汇表中，以便它们具有对应的固定标记ID。

Sennrich等人[2016]的算法1包含了一个效率较低的BPE分词器训练实现（基本上遵循我们上面概述的步骤）。作为第一个练习，实现并测试这个函数可能有助于检验你的理解。

### 示例（bpe_example）：BPE训练示例

这是来自Sennrich等人[2016]的一个示例化示例。考虑由以下文本组成的语料库：

```
low low low low low lower lower widest widest widest newest newest newest newest newest newest
```

词汇表有一个特殊标记`<|endoftext|>`。

**词汇表**：我们用特殊标记`<|endoftext|>`和256个字节值初始化词汇表。

**预分词**：为了简单起见并专注于合并过程，我们在此示例中假设预分词仅按空白字符分割。当我们预分词并计数时，我们得到频率表：

```python
{low: 5, lower: 2, widest: 3, newest: 6}
```

<footer>2注意，原始BPE公式[Sennrich等人, 2016]规定包含一个词尾标记。我们在训练字节级BPE模型时不添加词尾标记，因为所有字节（包括空白和标点符号）都已包含在模型的词汇表中。由于我们明确地表示空格和标点符号，学习到的BPE合并自然会反映这些词边界。</footer>

以`dict[tuple[bytes], int]`形式表示很方便，例如`{(l,o,w): 5, ...}`。注意，即使在Python中单个字节也是bytes对象。Python中没有表示单个字节的byte类型，就像没有表示单个字符的char类型一样。

**合并**：我们首先查看每一对连续的字节，并汇总它们出现的单词频率：`{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}`。字节对`('es')`和`('st')`频率相同，因此我们取字典序更大的`('st')`。然后我们将*预分词合并*，最终得到`{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}`。

第二轮中，我们看到`(e,st)`是最常见的字节对（计数为9），我们将合并为`{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}`。继续下去，最终得到的合并序列为`['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']`。

如果我们进行6次合并，得到`['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']`，我们的词汇表元素将是`[<|endoftext|>, [...256字节字符], st, est, ow, low, west, ne]`。使用这个词汇表和合并集，单词`newest`将被分词为`[ne, west]`。

## 2.5 BPE分词器训练实验

让我们在TinyStories数据集上训练一个字节级BPE分词器。数据集的查找/下载说明见第1节。开始之前，我们建议先查看TinyStories数据集以了解数据内容。

**并行化预分词**：你会发现**预分词步骤**是一个主要瓶颈。你可以使用内置的`multiprocessing`库**并行化代码**来加速预分词。具体而言，我们建议在预分词的并行实现中，对语料库进行分块，同时确保分块边界出现在特殊标记的开头。你可以自由使用以下链接中的启动代码来获取分块边界，然后用于在进程间分配工作：

[https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py)

这种分块始终是有效的，因为我们永远不希望跨文档边界进行合并。就本作业而言，你可以始终这样分割。无需担心接收不包含`<|endoftext|>`的超大语料库的边界情况。

**预分词前移除特殊标记**：在使用正则表达式模式（使用`re.finditer`）进行**预分词之前**，你应该从语料库（或分块）中**剥离所有特殊标记**。确保在特殊标记处分割，以便不能在它们分隔的文本之间发生合并。例如，如果你有一个语料库（或分块）如`[Doc 1]<|endoftext|>[Doc 2]`，你应该在特殊标记`<|endoftext|>`处分割，并分别预分词`[Doc 1]`和`[Doc 2]`，这样就不能跨文档边界发生合并。这可以使用`re.split`并以`"|".join(special_tokens)`作为分隔符（谨慎使用`re.escape`，因为`|`可能出现在特殊标记中）来完成。测试`test_train_bpe_special_tokens`将测试这一点。

**优化合并步骤**：上述示例中BPE训练的简单实现在每次合并时都会迭代所有字节对以识别最频繁的字节对，因此速度较慢。然而，每次合并后只有与合并字节对重叠的字节对计数会发生变化。因此，可以通过索引所有字节对的计数并增量更新这些计数，而不是显式迭代每对字节来统计频率，从而提高BPE训练速度。通过这种缓存过程可以获得显著的加速，尽管我们注意到BPE训练的合并部分在Python中是不可并行化的。

<footer>8</footer>

### 低资源/降规模提示：性能分析

你应该使用cProfile或scalene等性能分析工具来识别实现中的瓶颈，并专注于优化

### 低资源/降规模提示："降规模"

不要直接跳到在整个TinyStories数据集上训练分词器，我们建议你先在数据的一个小子上训练："调试数据集"。例如，你可以在TinyStories验证集上训练分词器，该验证集有22K个文档，而不是2.12M个。这说明了在可能的情况下降规模的一般策略：例如，使用更小的数据集、更小的模型规模等。选择调试数据集的大小或超参数配置需要仔细考虑：你希望你的调试集足够大，以便与完整配置具有相同的瓶颈（这样你做的优化将具有通用性），但又不能太大以至于需要很长时间才能

### 问题（train_bpe）：BPE分词器训练（15分）

**交付物**：编写一个函数，给定输入文本文件的路径，训练一个字节级BPE分词器。你的BPE训练函数应处理（至少）以下输入参数：

- `input_path: str`：BPE分词器训练文本文件的路径
- `vocab_size: int`：定义最大最终词汇表大小的正整数（包括初始字节词汇表、合并产生的词汇表项以及任何特殊标记）
- `special_tokens: list[str]`：要添加到词汇表中的字符串列表。这些特殊标记不会以其他方式影响BPE训练

你的BPE训练函数应返回生成的词汇表和合并：

- `vocab: dict[int, bytes]`：分词器词汇表，从int（词汇表中的标记ID）到bytes（标记字符串）的映射
- `merges: list[tuple[bytes, bytes]]`：训练产生的BPE合并列表。每个列表项是一个字节元组`(<token1>, <token2>)`，表示`<token1>`与`<token2>`合并。合并应按合并顺序排序

为了根据我们提供的测试测试你的BPE训练函数，你首先需要在`[adapters.run_train_bpe]`处实现测试适配器。然后运行`uv run pytest tests/test_train_bpe.py`。你的实现应能够通过所有测试。可选地（这可能需要大量时间投入），你可以使用某种系统语言实现训练方法的关键部分，例如C++（考虑使用cppyy）或Rust（使用PyO3）。如果你这样做，请注意哪些操作需要复制与直接从Python内存读取，并确保留下构建说明，或确保它仅使用pyproject.toml构建。另请注意，GPT-2正则表达式在大多数正则表达式引擎中支持不佳，在大多数支持的引擎中也会太慢。我们已验证Oniguruma速度相当快且支持负向前瞻，但Python中的regex包即使不是更快，也至少是

<footer>9</footer>

### 问题（train_bpe_tinystories）：在TinyStories上训练BPE（2分）

**(a)** 在TinyStories数据集上训练一个字节级BPE分词器，最大词汇表大小为10,000。确保将TinyStories的`<|endoftext|>`特殊标记添加到词汇表中。将生成的词汇表和合并序列化到磁盘以供进一步检查。训练花费了多少小时和内存？词汇表中最长的标记是什么？这合理吗？

**资源需求**：≤30分钟（无需GPU），≤30GB内存

**提示**：使用预分词期间的多进程以及以下两个事实，你应该能够在2分钟内完成BPE训练：

- `<|endoftext|>`标记分隔数据文件中的文档
- `<|endoftext|>`标记在BPE合并之前作为特殊情况处理

交付物：一到两句话回答

**(b)** 分析你的代码。分词器训练过程的哪个部分耗时最长？

交付物：一到两句话回答

接下来，我们将尝试在OpenWebText数据集上训练字节级BPE分词器。与之前一样，我们建议先查看数据集以更好地了解其

### 问题（train_bpe_expts_owt）：在OpenWebText上训练BPE（2分）

**(a)** 在OpenWebText数据集上训练一个字节级BPE分词器，最大词汇表大小为32,000。将生成的词汇表和合并序列化到磁盘以供进一步检查。词汇表中最长的标记是什么？这合理吗？

**资源需求**：≤12小时（无需GPU），≤100GB内存

交付物：一到两句话回答

**(b)** 比较和对比tokenize

交付物：一到两句话回答

训练

## 2.6 BPE分词器：编码和解码

在前面的作业部分中，我们实现了一个函数来在输入文本上训练BPE分词器，以获得分词器词汇表和BPE合并列表。现在，我们将实现一个BPE分词器，它加载提供的词汇表和合并列表，并使用它们将文本编码为标记

### 2.6.1 编码文本

通过BPE编码文本的过程与我们训练BPE词汇表的方式类似。主要有几个步骤：

**步骤1：预分词**。我们首先对序列进行预分词，并将每个预分词表示为UTF-8字节序列，就像我们在BPE训练中所做的那样。我们将在每个预分词内部将这些字节合并为词汇表元素，独立处理每个预分词（不允许跨预分词

**步骤2：应用合并**。然后我们采用BPE训练期间创建的词汇表元素合并序列，并以相同的顺序将其应用于我们的预分词

<footer>10</footer>

### 示例（bpe_encoding）：BPE编码示例

例如，假设我们的输入字符串是`'the cat ate'`，我们的词汇表是`{0: b'', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b'c', 8: b'a', 9: b'the', 10: b'at'}`，我们学习到的合并是`[(b't', b'h'), (b'', b'c'), (b'', b'a'), (b'th', b'e'), (b'a', b't')]`。首先，我们的预分词器会将此字符串分割为`['the', 'cat', 'ate']`。然后，我们将查看每个预分词并应用BPE合并。

第一个预分词`'the'`最初表示为`[b't', b'h', b'e']`。查看我们的合并列表，我们识别第一个适用的合并是`(b't', b'h')`，并使用它将其转换为`[b'th', b'e']`。然后，我们回到合并列表并识别下一个适用的合并是`(b'th', b'e')`，这将预分词转换为`[b'the']`。最后，查看合并列表，我们看到没有更多适用于该字符串的合并（因为整个预分词已合并为单个标记），所以我们已完成BPE合并的应用。对应的整数序列是

对剩余预分词重复此过程，我们看到预分词`'cat'`在应用BPE合并后表示为`[b'c', b'a', b't']`，变为整数序列`[7, 1, 5]`。最后一个预分词`'ate'`在应用BPE合并后为`[b'at', b'e']`，变为整数序列`[10, 3]`。因此，编码输入字符串的最终结果是`[9, 7, 1, 5, 10, 3]`。

**特殊标记**。你的分词器在编码文本时应能够正确处理用户定义的特殊标记（在构建时提供）。

**内存考虑**。假设我们想要对无法放入内存的大型文本文件进行分词。为了高效地对这种大文件（或任何其他数据流）进行分词，我们需要将其分解为可管理的块并依次处理每个块，这样内存复杂度是常数而不是随文本大小线性增长。在此过程中，我们需要确保标记不会跨越块边界，否则我们会得到与对整个序列进行分词的朴素方法不同的分词结果。

### 2.6.2 解码文本

要将整数标记ID序列解码回原始文本，我们可以简单地查找每个ID在词汇表中对应的条目（字节序列），将它们连接起来，然后将字节解码为Unicode字符串。注意，输入ID不能保证映射到有效的Unicode字符串（因为用户可以输入任何整数序列）。在输入标记ID不产生有效Unicode字符串的情况下，你应该使用官方的Unicode替换字符U+FFFD替换格式错误的字节³。`bytes.decode`的`errors`参数控制如何处理Unicode解码错误，使用`errors='replace'`会自动用替换字符替换格式错误的

### 问题（tokenizer）：实现分词器（15分）

**交付物**：实现一个Tokenizer类，给定一个词汇表和合并列表，将文本编码为整数ID并将整数ID解码为文本。你的分词器还应支持用户提供的特殊标记（如果它们尚未在词汇表中，则将其附加到词汇表）。

我们推荐以下接口：

```python
def __init__(self, vocab, merges, special_tokens=None)
```
从给定的词汇表、合并列表和（可选的）特殊标记列表构建分词器。此函数应接受以下参数：
- `vocab: dict[int, bytes]`
- `merges: list[tuple[bytes, bytes]]`
- `special_tokens: list[str] | None = None`

```python
def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)
```
类方法，从序列化的词汇表和合并列表（与你的BPE训练代码输出的格式相同）和（可选的）特殊标记列表构建并返回一个Tokenizer。此方法应接受以下附加参数：
- `vocab_filepath: str`
- `merges_filepath: str`
- `special_tokens: list[str] | None = None`

```python
def encode(self, text: str) -> list[int]
```
将输入文本编码为标记

```python
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]
```
给定字符串的可迭代对象（例如Python文件句柄），返回一个懒加载生成标记ID的生成器。这对于无法直接加载到内存的大文件的内存高效

```python
def decode(self, ids: list[int]) -> str
```
将标记ID序列解码为

为了根据我们提供的测试测试你的Tokenizer，你首先需要在`[adapters.get_tokenizer]`处实现测试适配器。然后运行`uv run pytest tests/test_tokenizer.py`。你的实现应能够通过所有测试。

### 2.7 实验

**问题（tokenizer_experiments）**：分词器实验（4分）

**(a)** 从TinyStories和OpenWebText中各采样10个文档。使用你之前训练的TinyStories和OpenWebText分词器（词汇表大小分别为10K和32K）将这些采样文档编码为整数ID。每个分词器的压缩比（字节/标记）是多少？

交付物：一到两句话回答。

**(b)** 如果使用TinyStories分词器对OpenWebText样本进行分词会发生什么？比较压缩比和/或定性描述发生了什么。

交付物：一到两句话回答。

**(c)** 估算你的分词器吞吐量（例如，字节/秒）。对完整的Pile数据集（825GB文本）进行分词需要多长时间？

交付物：一到两句话回答。

**(d)** 使用你的TinyStories和OpenWebText分词器将相应的训练和开发数据集编码为整数标记ID序列。我们稍后将使用此序列来训练语言模型。我们建议将标记ID序列序列化为数据类型为`uint16`的NumPy数组。为什么选择`uint16`是合适的？

<header>交付物：一到两句话</header>

<footer>13</footer>

现在我们已经完成了对分词器和语言建模数据处理的讨论，接下来构建Transformer语言模型。

让我们从高层次开始。语言模型接收整数标记ID序列作为输入，并输出每个位置的下一个标记预测。参见图1中的模型图。我们将这一概念形式化为

我们可以将这个分布视为为给定语言中的每个单词分配一个概率

现在更具体地，给定一个标记ID序列，Transformer语言模型使用标记嵌入层生成一系列向量。每个嵌入层接收一个形状为`(batch_size, sequence_length)`的整数张量，并生成一个形状为`(batch_size, sequence_length, ...)`的向量序列

图1：Transformer语言模型的高层架构图。该模型接收标记嵌入，将它们传递到多个Transformer块（灰色堆叠框）中，应用RMSNorm，然后使用输出嵌入层来预测下一个标记

### 3.1.2 预归一化Transformer块

嵌入之后，激活值由多个结构相同的神经网络层处理。标准的仅解码器Transformer语言模型由`num_layers`个相同的层组成（通常称为Transformer"块"）。每个Transformer块接收一个形状为`(batch_size, sequence_length, d_model)`的输入，并返回一个形状为`(batch_size, sequence_length, d_model)`的输出。每个块在序列中聚合信息（通过自注意力）并对其进行非线性变换（通过前馈网络）。

### 3.2 输出归一化和嵌入

经过`num_layers`个Transformer块之后，我们将获取最终激活值并将其转换为

我们将实现"预归一化"Transformer块（详见§3.5），它还需要在最终Transformer块之后使用层归一化（详见下文），以确保其输出正确

在此归一化之后，我们将使用标准的可学习线性变换将Transformer块的输出转换为预测的下一个标记对数几率（参见例如Radford等人[2018]公式

### 3.3 备注：批处理、Einsum和高效计算

在整个Transformer中，我们将对许多类似批次的输入执行相同的计算。以下是一些示例：

- 批次的元素：我们在每个批次元素上应用相同的Transformer前向操作
- 序列长度："位置级"操作如RMSNorm和前馈网络在序列的每个位置上操作方式相同
- 注意力头：注意力操作在"多头"注意力中跨注意力头进行批处理

拥有一个符合人体工程学的方式来执行此类操作非常有用，这种方式可以充分利用GPU，并且易于阅读和理解。许多PyTorch操作可以接受张量开头额外的"类似批次"维度，并跨这些维度重复/广播该操作

例如，假设我们正在执行一个位置级、批次化的操作。我们有一个"数据张量"`D`，其形状为`(batch_size, sequence_length, d_model)`，我们希望对一个形状为`(d_model, d_model)`的矩阵`A`执行批次化的向量-矩阵乘法。在这种情况下，`D @ A`将执行批次化的矩阵乘法，这是PyTorch中的高效原语，其中`(batch_size, sequence_length)`维度是批次化的

因此，假设你的函数可能会接收额外的类似批次维度，并将这些维度保留在PyTorch形状的开头，这会很有帮助。为了使张量能够以这种方式进行批次化，它们可能需要使用`view`、`reshape`和`transpose`的多步操作来塑形。这可能有点痛苦，而且通常很难读懂代码在做什么以及你的张量形状

一个更符合人体工程学的选择是在`torch.einsum`中使用einsum符号，或者使用与框架无关的库，如`einops`或`einx`。两个关键操作是`einsum`，它可以对输入张量的任意维度执行张量收缩；以及`rearrange`，它可以重新排序、连接和分割任意

<footer>15</footer>

维度。事实证明，机器学习中的几乎所有操作都是维度转换和张量收缩的某种组合，偶尔夹杂（通常是逐点的）非线性函数。这意味着当你使用einsum符号时，很多代码可读性更强、更灵活。

我们强烈建议学习并在本课程中使用einsum符号。以前没有接触过einsum符号的学生应该使用einops（文档在此），已经熟悉einops的学生应该学习更通用的einx（文档在此）。这两个包在我们提供的环境中都已安装。

这里我们给出一些如何使用einsum符号的示例。这些是对einops文档的补充，你应该先阅读einops文档。

**使用einops.einsum进行批次矩阵乘法**

```python
import torch
from einops import rearrange, einsum

# 基本实现
Y = D @ A.T

# 很难判断输入和输出形状及其含义
# D和A可以有哪些形状，其中是否有意外行为？

# Einsum是自文档化且健壮的
Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")

# 或者，D可以有任意前导维度但A受限制的批次版本
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
```

**示例（einstein_example2）：使用einops.rearrange进行广播操作**

我们有一批图像，对于每个图像，我们想根据某个缩放因子生成10个变暗版本：

```python
images = torch.randn(64, 128, 128, 3)  # (batch, height, width, channel)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)

# 重塑并相乘
dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1")
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images = images_rearr * dim_value

# 或者一步到位：
dimmed_images = einsum(
    images, dim_by,
    "batch height width channel, dim_value -> batch dim_value height width channel"
)
```

**示例（einstein_example3）：使用einops.rearrange进行像素混合**

假设我们有一批图像，表示为形状为`(batch, height, width, channel)`的张量，我们想对图像的所有像素执行线性变换，但该变换应独立地应用于每个通道。我们的线性变换由矩阵`B`表示，形状为`(height × width, height × width)`。

```python
channels_last = torch.randn(64, 32, 32, 3)  # (batch, height, width, channel)
B = torch.randn(32*32, 32*32)

# 重塑图像张量以跨所有像素混合
channels_last_flat = channels_last.view(
    -1, channels_last.size(1)*channels_last.size(2), channels_last.size(3)
)
channels_first_flat = channels_last_flat.transpose(1, 2)
channels_first_flat_transformed = channels_first_flat @ B.T
channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)

# 使用einops：
height = width = 32

# rearrange替代繁琐的torch view + transpose
channels_first = rearrange(
    channels_last,
    "batch height width channel -> batch channel (height width)"
)
channels_first_transformed = einsum(
    channels_first, B,
    "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out"
)
channels_last_transformed = rearrange(
    channels_first_transformed,
    "batch channel (height width) -> batch height width channel",
    height=height, width=width
)

# 或者，如果你敢的话：使用einx.dot一步到位（einx相当于einops.einsum）
height = width = 32
channels_last_transformed = einx.dot(
    "batch row_in col_in channel, (row_out col_out)(row_in col_in) -> batch row_out col_out channel",
    channels_last, B,
    col_in=width, col_out=width
)
```

第一个实现可以通过在前后放置注释来指示输入和输出形状来改进，但这很笨拙且容易出错。使用einsum符号，文档就是实现！

Einsum符号可以处理任意的输入批处理维度，而且一个关键好处是自文档化。使用einsum符号的代码中，输入和输出张量的相关形状更加清晰。对于剩余的张量，你可以考虑使用Tensor类型提示，例如使用jaxtyping库（不特定于Jax）。

我们将在作业2中更多地讨论使用einsum符号的性能影响，但现在知道它们几乎总是比替代方案更好！

### 3.3.1 数学符号和内存排序

许多机器学习论文使用行向量符号，这导致与NumPy和PyTorch中默认使用的行优先内存排序很好地匹配。使用行向量时，线性变换看起来像这样

$$y = x W^\top,$$

对应于行优先内存排序的$W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$和行向量$x \in \mathbb{R}^{1 \times d_{\text{in}}}$。

在线性代数中，通常更常见的是使用列向量，其中线性变换看起来像

$$y = W x,$$

给定行优先的$W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$和列向量$x \in \mathbb{R}^{d_{\text{in}}}$。在本作业中，我们将使用列向量进行数学符号表示，因为通常这样更容易理解数学原理。请记住，如果你想使用纯矩阵乘法符号，则必须使用行向量约定来应用矩阵，因为PyTorch使用行优先内存排序。如果你对矩阵操作使用einsum，这应该不是问题。

### 3.4 基本构建块：线性和嵌入模块

#### 3.4.1 参数初始化

有效训练神经网络通常需要仔细初始化模型参数——不好的初始化可能导致诸如梯度消失或爆炸等不良行为。预归一化Transformer对初始化异常鲁棒，但它们仍然会对训练速度和收敛产生重大影响。由于本作业已经很长，我们将细节留到作业3，而是给出一些应该适用于大多数情况的近似初始化。现在使用：

- 线性权重：$\mathcal{N}(\mu=0, \sigma^2=\frac{2}{d_{\text{in}}+d_{\text{out}}})$，在$[-3\sigma, 3\sigma]$处截断
- 嵌入：$\mathcal{N}(\mu=0, \sigma^2=1)$，在$[-3, 3]$处截断
- RMSNorm：$\gamma$初始化为1

你应该使用`torch.nn.init.trunc_normal_`来初始化截断正态权重。

#### 3.4.2 线性模块

线性层是Transformer和神经网络的基本构建块。首先，你将实现自己的Linear类，继承自`torch.nn.Module`，执行线性变换：

$$y = W x$$

注意，我们不包含偏置项，遵循大多数现代LLM的做法。

##### 问题（linear）：实现线性模块（1分）

**交付物**：实现一个继承自`torch.nn.Module`并执行线性变换的Linear类。你的实现应遵循PyTorch内置的`nn.Linear`模块的接口，只是没有偏置参数或权重。我们推荐以下接口：

```python
def __init__(self, in_features, out_features, device=None, dtype=None)
```
构建线性变换模块。此函数应接受以下参数：
- `in_features: int`：输入的最终维度
- `out_features: int`：输出的最终维度
- `device: torch.device | None = None`：存储参数的设备
- `dtype: torch.dtype | None = None`：参数的数据类型

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```
对输入应用线性变换。

确保：
- 继承`nn.Module`
- 调用超类构造函数
- 构建并存储参数为$W$（而不是$W^\top$）出于内存排序原因，将其放入`nn.Parameter`
- 当然，不要使用`nn.Linear`或`nn.functional.linear`

对于初始化，使用上面的设置以及`torch.nn.init.trunc_normal_`来初始化权重。

为了测试你的Linear模块，在`[adapters.run_linear]`处实现测试适配器。适配器应将给定的权重加载到你的Linear模块中。你可以为此目的使用`Module.load_state_dict`。然后运行`uv run pytest -k test_linear`。

#### 3.4.3 嵌入模块

如上所述，Transformer的第一层是嵌入层，它将整数标记ID映射到维度为`d_model`的向量空间。我们将实现一个继承自`torch.nn.Module`的自定义Embedding类（因此不应使用`nn.Embedding`）。forward方法应通过索引一个形状为`(vocab_size, d_model)`的嵌入矩阵，使用一个形状为`(batch_size, sequence_length)`的`torch.LongTensor`的标记ID，来选择每个标记ID的嵌入向量。

##### 问题（embedding）：实现嵌入模块（1分）

**交付物**：实现一个继承自`torch.nn.Module`并执行嵌入查找的Embedding类。你的实现应遵循PyTorch内置的`nn.Embedding`模块的接口。我们推荐以下接口：

```python
def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)
```
构建嵌入模块。此函数应接受以下参数：
- `num_embeddings: int`：词汇表大小
- `embedding_dim: int`：嵌入向量的维度，即$d_{\text{model}}$
- `device: torch.device | None = None`：存储参数的设备
- `dtype: torch.dtype | None = None`：参数的数据类型

```python
def forward(self, token_ids: torch.Tensor) -> torch.Tensor
```
查找给定标记ID的嵌入向量。

确保：
- 继承`nn.Module`
- 调用超类构造函数
- 将嵌入矩阵初始化为`nn.Parameter`
- 存储嵌入矩阵时让`d_model`作为最终维度
- 当然，不要使用`nn.Embedding`或`nn.functional.embedding`

同样，使用上面的设置进行初始化，并使用`torch.nn.init.trunc_normal_`来初始化权重。

为了测试你的实现，在`[adapters.run_embedding]`处实现测试适配器。然后运行`uv run pytest -k test_embedding`。

### 3.5 预归一化Transformer块

每个Transformer块有两个子层：多头自注意力机制和位置前馈网络（Vaswani等人, 2017, 第3.1节）。

原始Transformer论文中的模型在每个两个子层周围使用残差连接，然后进行层归一化。这种架构通常称为"后归一化"Transformer，因为层归一化应用于子层输出。然而，各种工作发现，将层归一化从每个子层的输出移动到每个子层的输入（在最终Transformer块后增加一个层归一化）可以提高Transformer训练稳定性 [Nguyen和Salazar, 2019; Xiong等人, 2020]——参见图2查看这种"预归一化"Transformer块的直观表示。然后通过残差连接将每个Transformer块子层的输出添加到子层输入上（Vaswani等人, 2017, 第5.4节）。对预归一化的一个直觉是，存在一个从Transformer输入到最终输出的干净"残差流"，没有任何归一化，这据称可以改善梯度流。这种预归一化Transformer现在是当今语言模型中使用的标准（例如GPT-3、LLaMA、PaLM等），因此我们将实现这种变体。我们将依次实现预归一化Transformer块的每个组件。

#### 3.5.1 均方根层归一化

Vaswani等人[2017]的原始Transformer实现使用层归一化 [Ba等人, 2016]来归一化激活值。遵循Touvron等人[2023]，我们将使用均方根层归一化（RMSNorm; Zhang和Sennrich, 2019, 公式4）进行层归一化。给定一个激活值向量$a \in \mathbb{R}^{d_{\text{model}}}$，RMSNorm将每个激活值$a_i$缩放如下：

$$\text{RMSNorm}(a_i) = \frac{a_i}{\text{RMS}(a)} g_i$$

其中$\text{RMS}(a) = \sqrt{\frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} a_i^2 + \epsilon}$。这里，$g_i$是一个可学习的"增益"参数（总共有$d_{\text{model}}$个这样的参数），$\epsilon$是一个超参数，通常固定为$1e-5$。

你应该在平方输入之前将其转换为`torch.float32`以防止溢出。总的来说，你的forward方法应如下所示：

```python
in_dtype = x.dtype
x = x.to(torch.float32)
# 你的代码执行RMSNorm
...
result = ...
# 以原始dtype返回结果
return result.to(in_dtype)
```

##### 问题（rmsnorm）：均方根层归一化（1分）

**交付物**：将RMSNorm实现为`torch.nn.Module`。我们推荐以下接口：

```python
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)
```
构建RMSNorm模块。此函数应接受以下参数：
- `d_model: int`：模型的隐藏维度
- `eps: float = 1e-5`：用于数值稳定性的epsilon值
- `device: torch.device | None = None`：存储参数的设备
- `dtype: torch.dtype | None = None`：参数的数据类型

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```
处理形状为`(batch_size, sequence_length, d_model)`的输入张量并返回相同形状的张量。

注意：在执行归一化之前记住将输入转换为`torch.float32`（稍后转换回原始dtype），如上所述。

为了测试你的实现，在`[adapters.run_rmsnorm]`处实现测试适配器。然后运行`uv run pytest -k test_rmsnorm`。

#### 3.5.2 位置前馈网络

每个Transformer块包含一个位置前馈网络，它对序列中的每个位置应用非线性变换。这一层有时称为"前馈"或"FFN"层。位置前馈网络通常由两个线性变换组成，中间有一个ReLU激活：

$$\text{FFN}(x) = W_2 \max(0, W_1 x)$$

然而，我们将如Shazeer[2020]所述实现门控线性单元（GLU）变体（称为SwiGLU），并使用SiLU激活函数。GLU和SiLU都单独改善了Transformer，Shazeer发现它们的组合（SwiGLU）在语言建模中效果最好。GLU将激活分成两部分，将它们相乘，类似于LSTM中的门控机制：

$$\text{GLU}(x) = (x W_1) \otimes \sigma(x V)$$

在实践中，我们通常使用一个矩阵执行此操作$W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$，然后将结果分成两部分。我们将使用SwiGLU变体，它使用没有额外可学习参数的$\text{SiLU}(x) = x \cdot \sigma(x)$：

$$\text{SwiGLU}(x) = \text{SiLU}(x W_1) \otimes (x V)$$

在实践中，我们可以将$W_1$和$V$合并为一个形状为$d_{\text{model}} \times 2 d_{\text{ff}}$的矩阵，或保持它们分开。

Shazeer[2020]首先提出将SiLU/Swish激活与GLU结合，并进行了实验，表明SwiGLU在语言建模任务上优于ReLU和SiLU（无门控）等基线。在作业的后面部分，你将比较SwiGLU和SiLU。虽然我们提到了这些组件的一些启发式论据（论文提供了更多支持证据），但保持一个经验性的视角是很好的：Shazeer论文中一句著名的话是

> 我们没有解释为什么这些架构似乎有效；我们将它们的成功，像其他一切一样，归因于神圣的仁慈。

##### 问题（positionwise_feedforward）：实现位置前馈网络（2分）

**交付物**：实现SwiGLU前馈网络，由SiLU激活函数和GLU组成。

注意：在这个特定情况下，你可以自由地在实现中使用`torch.sigmoid`以获得数值稳定性。

你应该在实现中将$d_{\text{ff}}$设置为大约$\frac{8}{3} \times d_{\text{model}}$，同时确保内层前馈网络的维度是64的倍数，以充分利用GPU张量核心。为了根据我们提供的测试测试你的实现，你需要在`[adapters.run_swiglu]`处实现测试适配器。然后运行`uv run pytest -k test_swiglu`来测试你的实现。

#### 3.5.3 相对位置嵌入

为了向模型注入位置信息，我们将实现旋转位置嵌入 [Su等人, 2021]，通常称为RoPE。对于给定位置$i$处的查询标记$q^{(i)} = W_q x^{(i)} \in \mathbb{R}^d$，我们将应用成对旋转矩阵$R^i$，得到$q'^{(i)} = R^i q^{(i)} = R^i W_q x^{(i)}$。这里，$R^i$将旋转嵌入元素的成对元素$q^{(i)}_{2k-1:2k}$作为2D向量，旋转角度为$\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d}}$，对于$k \in \{1, \ldots, d/2\}$和某个常数$\Theta$。因此，我们可以将$R^i$视为大小为$d \times d$的分块对角矩阵，其中块为$R^i_k$，对于$k \in \{1, \ldots, d/2\}$，具有

$$R^i_k = \begin{bmatrix}
\cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\
\sin(\theta_{i,k}) & \cos(\theta_{i,k})
\end{bmatrix}$$

因此我们得到完整的旋转矩阵

$$R^i = \begin{bmatrix}
R^i_1 & 0 & 0 & \dots & 0 \\
0 & R^i_2 & 0 & \dots & 0 \\
0 & 0 & R^i_3 & \dots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \dots & R^i_{d/2}
\end{bmatrix}$$

其中0表示$2 \times 2$的零矩阵。虽然可以构造完整的$d \times d$矩阵，但好的解决方案应该利用这个矩阵的性质来更高效地实现变换。由于我们只关心给定序列中标记的相对旋转，我们可以在不同层和不同批次之间重用我们为$\cos(\theta_{i,k})$和$\sin(\theta_{i,k})$计算的值。如果你想优化它，可以让所有层引用单个RoPE模块，并在`init`期间使用`self.register_buffer(persistent=False)`创建一个2D预计算的sin和cos值缓冲区，而不是`nn.Parameter`（因为我们不想学习这些固定的余弦和正弦值）。然后对我们为$q^{(i)}$所做的完全相同的旋转过程也对$k^{(j)}$进行，通过相应的$R^j$旋转。注意，这一层没有可学习的参数。

##### 问题（rope）：实现RoPE（2分）

**交付物**：实现一个`RotaryPositionalEmbedding`类，对输入张量应用RoPE。推荐以下接口：

```python
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)
```
构建RoPE模块并在需要时创建缓冲区。
- `theta: float`：RoPE的$\Theta$值
- `d_k: int`：查询和key向量的维度
- `max_seq_len: int`：将输入的最大序列长度
- `device: torch.device | None = None`：存储缓冲区的设备

```python
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
```
处理形状为`(..., seq_len, d_k)`的输入张量并返回相同形状的张量。注意，你应该容忍`x`具有任意数量的批处理维度。你应该假设标记位置是形状为`(..., seq_len)`的张量，指定`x`沿序列维度的标记位置。

你应该使用标记位置沿序列维度切片你的（可能预计算的）cos和sin张量。

为了测试你的实现，完成`[adapters.run_rope]`并确保它通过`uv run pytest -k test_rope`。

#### 3.5.4 缩放点积注意力

我们现在将如Vaswani等人[2017]（第3.2.1节）所述实现缩放点积注意力。作为预备步骤，注意力操作的定义将使用softmax，该操作将未归一化的分数向量转换为归一化分布：

$$\text{softmax}(v)_i = \frac{\exp(v_i)}{\sum_{j=1}^n \exp(v_j)}$$

注意，$\exp(v_i)$对于大的值可能变为`inf`（此时`inf/inf=NaN`）。我们可以通过注意到softmax操作对向所有输入添加任何常数$c$是不变的来避免这一点。我们可以利用这个性质来提高数值稳定性——通常，我们会从$v_i$的所有元素中减去其最大元素，使新的最大元素为0。现在你将实现softmax，使用这个技巧来提高数值稳定性。

##### 问题（softmax）：实现softmax（1分）

**交付物**：编写一个函数来应用softmax操作。你的函数应接受两个参数：一个张量和一个维度`i`，并将softmax应用于输入张量的第`i`个维度。输出张量应具有与输入张量相同的形状，但其第`i`个维度现在将具有归一化的概率分布。使用从第`i`个维度的所有元素中减去第`i`个维度中的最大值这一技巧来避免数值稳定性问题。

为了测试你的实现，完成`[adapters.run_softmax]`并确保它通过`uv run pytest -k test_softmax_matches_pytorch`。

我们现在可以数学上定义注意力操作如下：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q^\top K}{\sqrt{d_k}}\right) V$$

其中$Q \in \mathbb{R}^{n \times d_k}$，$K \in \mathbb{R}^{m \times d_k}$，$V \in \mathbb{R}^{m \times d_v}$。这里，$Q$、$K$和$V$都是此操作的输入——注意这些不是可学习参数。如果你想知道为什么这不是$Q K^\top$，参见3.3.1。

**掩码**：有时方便地掩码注意力操作的输出。掩码的形状应为$M \in \{\text{True}, \text{False}\}^{n \times m}$，该布尔矩阵的每一行$i$表示查询$i$应该关注哪些key。按照惯例（有点令人困惑），位置$(i,j)$处的`True`值表示查询$i$确实关注key$j`，`False`值表示查询不关注该key。换句话说，"信息流动"发生在值为`True`的$(i,j)$对上。例如，考虑一个$1 \times 3$的掩码矩阵，其条目为`[[True, True, False]]`。单个查询向量只关注前两个key。

计算上，使用掩码比在子序列上计算注意力要高效得多，我们可以通过取softmax前的值$\left(\frac{Q^\top K}{\sqrt{d_k}}\right)$并在掩码矩阵中任何`False`条目的位置添加$-\infty$来实现。

##### 问题（scaled_dot_product_attention）：实现缩放点积注意力（5分）

**交付物**：实现缩放点积注意力函数。你的实现应处理形状为`(batch_size, ..., seq_len, d_k)`的key和查询以及形状为`(batch_size, ..., seq_len, d_v)`的值，其中`...`表示任何数量的其他类似批次的维度（如果提供）。实现应返回形状为`(batch_size, ..., d_v)`的输出。参见第3.3节对类似批次的讨论。

你的实现还应支持可选的用户提供的布尔掩码，形状为`(seq_len, seq_len)`。具有掩码值`True`的位置的注意力概率应集体和为1，具有掩码值`False`的位置的注意力概率应为零。

为了根据我们提供的测试测试你的实现，你需要在`[adapters.run_scaled_dot_product_attention]`处实现测试适配器。

`uv run pytest -k test_scaled_dot_product_attention`在第三阶输入张量上测试你的实现，而`uv run pytest -k test_4d_scaled_dot_product_attention`在第四阶输入张量上测试你的实现。

#### 3.5.5 因果多头自注意力

我们将实现Vaswani等人[2017]第3.2.2节中描述的多头自注意力机制。回顾一下，从数学上讲，应用多头注意力的操作定义如下：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$

其中$\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$，$Q_i, K_i, V_i$分别是$Q, K, V$中第$i \in \{1, \ldots, h\}$个切片，大小为$d_k$或$d_v$。Attention是我们在§3.5.4中定义的缩放点积注意力操作。由此我们可以形成多头自注意力操作：

$$\text{MultiHeadSelfAttention}(x) = W_O \text{MultiHead}(W_Q x, W_K x, W_V x)$$

这里的可学习参数是$W_Q \in \mathbb{R}^{h d_k \times d_{\text{model}}}$，$W_K \in \mathbb{R}^{h d_k \times d_{\text{model}}}$，$W_V \in \mathbb{R}^{h d_v \times d_{\text{model}}}$，和$W_O \in \mathbb{R}^{d_{\text{model}} \times h d_v}$。由于在多头注意力操作中$Q$、$K$、$V$会被切片，我们可以认为$W_Q$、$W_K$和$W_V$在每个头的输出维度上被分割。当这部分完成后，你应该总共只用三次矩阵乘法来计算key、value和query投影⁵。

<footer>⁵作为延伸目标，尝试将key、query和value投影合并到单个权重矩阵中，这样只需要一次矩阵乘法。</footer>

**因果掩码**：你的实现应防止模型关注序列中的未来标记。换句话说，如果模型被给定标记序列$t_1, \ldots, t_n$，并且我们想要计算前缀$t_1, \ldots, t_i$（其中$i < n$）的下一个词预测，模型不应该能够访问（关注）位置$t_{i+1}, \ldots, t_n$处的标记表示，因为在推理期间生成文本时它将无法访问这些标记（这些未来的标记会泄漏关于真实下一个词身份的信息，使语言建模预训练目标变得简单）。对于输入标记序列$t_1, \ldots, t_n$，我们可以通过运行$n$次多头自注意力（针对序列中的$n$个唯一前缀）来简单地防止访问未来标记。相反，我们将使用因果注意力掩码，它允许标记$i$关注序列中所有位置$j \leq i$。你可以使用`torch.triu`或广播索引比较来构建此掩码，并且应该利用你在§3.5.4中的缩放点积注意力实现已经支持注意力掩码这一事实。

**应用RoPE**：RoPE应应用于query和key向量，但不应用于value向量。此外，头维度应作为批次维度处理，因为在多头注意力中，注意力是独立应用于每个头的。这意味着精确相同的RoPE旋转应应用于每个头的query和key向量。

##### 问题（multihead_self_attention）：实现因果多头自注意力（5分）

**交付物**：将因果多头自注意力实现为`torch.nn.Module`。你的实现至少应接受以下参数：

- `d_model: int`：Transformer块输入的维度
- `num_heads: int`：多头自注意力中使用的头数

遵循Vaswani等人[2017]，设置$d_k = d_v = d_{\text{model}} / h$。

为了根据我们提供的测试测试你的实现，在`[adapters.run_multihead_self_attention]`处实现测试适配器。然后运行`uv run pytest -k test_multihead_self_attention`来测试你的实现。

### 3.6 完整的Transformer LM

现在让我们开始组装Transformer块（参考图2会很有帮助）。Transformer块包含两个"子层"，一个用于多头自注意力，另一个用于前馈网络。在每个子层中，我们首先执行RMSNorm，然后执行主要操作（MHA/FF），最后添加残差连接。

具体来说，Transformer块的前半部分（第一个"子层"）应实现以下更新以从输入$x$产生输出$y$：

$$y = x + \text{MultiHeadSelfAttention}(\text{RMSNorm}(x))$$

##### 问题（transformer_block）：实现Transformer块（3分）

按照§3.5中的描述和图2所示实现预归一化Transformer块。你的Transformer块至少应接受以下参数：

- `d_model: int`：Transformer块输入的维度
- `num_heads: int`：多头自注意力中使用的头数
- `d_ff: int`：位置前馈网络内层的维度

为了测试你的实现，实现适配器`[adapters.run_transformer_block]`。然后运行`uv run pytest -k test_transformer_block`来测试你的实现。

**交付物**：通过提供测试的Transformer块代码。

现在我们按照图1中的高层图示将这些块组合在一起。按照我们在第3.1.1节中对嵌入的描述，将其输入到`num_layers`个Transformer块中，然后传递给三个输出层以获得词汇表上的分布。

##### 问题（transformer_lm）：实现Transformer LM（3分）

是时候将它们整合在一起了！按照§3.1中的描述和图1所示实现Transformer语言模型。至少，你的实现应接受上述Transformer块的所有构造参数，以及以下附加参数：

- `vocab_size: int`：词汇表大小，用于确定标记嵌入矩阵的维度
- `context_length: int`：最大上下文长度，用于确定位置嵌入矩阵的维度
- `num_layers: int`：使用的Transformer块数量

为了根据我们提供的测试测试你的实现，你首先需要在`[adapters.run_transformer_lm]`处实现测试适配器。然后运行`uv run pytest -k test_transformer_lm`来测试你的实现。

**交付物**：通过上述测试的Transformer LM模块。

**资源核算**：能够了解Transformer的各个部分如何消耗计算和内存是有用的。我们将执行一些基本的"FLOPs核算"。Transformer中的绝大多数FLOPs都是矩阵乘法，因此我们的核心方法很简单：

1. 写下Transformer前向传递中的所有矩阵乘法
2. 将每个矩阵乘法转换为所需的FLOPs

对于第二步，以下事实将很有用：

**规则**：给定$A \in \mathbb{R}^{m \times n}$和$B \in \mathbb{R}^{n \times p}$，矩阵-矩阵乘积$AB$需要$2mnp$个FLOPs。

要理解这一点，注意$(AB)[i,j] = A[i,:] \cdot B[:,j]$，这个点积需要$n$次加法和$n$次乘法（$2n$个FLOPs）。然后，由于矩阵-矩阵乘积$AB$有$m \times p$个条目，FLOPs总数为$(2n)(mp) = 2mnp$。

现在，在解决下一个问题之前，遍历你的Transformer块和Transformer LM的每个组件，列出所有的矩阵乘法及其相关的FLOPs成本可能会有所帮助。

##### 问题（transformer_accounting）：Transformer LM资源核算（5分）

**(a)** 考虑GPT-2XL，其配置如下：
- `vocab_size`: 50,257
- `context_length`: 1,024
- `num_layers`: 48
- `d_model`: 1,600
- `num_heads`: 25
- `d_ff`: 6,400

假设我们使用此配置构建模型。我们的模型将有多少可训练参数？假设每个参数都使用单精度浮点数表示，仅加载此模型需要多少内存？

交付物：一到两句话回答。

**(b)** 确定完成GPT-2XL形状模型前向传递所需的矩阵乘法。这些矩阵乘法总共需要多少FLOPs？假设我们的输入序列具有`context_length`。

交付物：矩阵乘法列表（带描述）以及总的FLOPs数量。

**(c)** 基于以上分析，模型的哪些部分需要最多的FLOPs？

交付物：一到两句话回答。

**(d)** 对GPT-2small（12层，768 `d_model`，12个头）、GPT-2medium（24层，1024 `d_model`，16个头）和GPT-2large（36层，1280 `d_model`，20个头）重复你的分析。随着模型规模的增加，Transformer LM的哪些部分占总FLOPs的比例更大或更小？

交付物：对于每个模型，提供模型组件及其相关FLOPs的分解（作为前向传递所需总FLOPs的比例）。此外，提供一到两句话描述模型规模的变化如何改变每个组件FLOPs的相对比例。

**(e)** 取GPT-2XL并将上下文长度增加到16,384。单次前向传递的总FLOPs如何变化？模型组件FLOPs的相对贡献如何变化？

交付物：一到两句话回答。

<footer>28</footer>

# 4 训练Transformer LM

现在我们已经有了预处理数据的步骤（通过分词器）和模型（Transformer）。剩下的工作是构建支持训练的所有代码。这包括以下内容：

- **损失**：我们需要定义损失函数（交叉熵）
- **优化器**：我们需要定义优化器来最小化此损失（AdamW）
- **训练循环**：我们需要所有支持基础设施来加载数据、保存检查点和管理训练

## 4.1 交叉熵损失

回顾Transformer语言模型为每个序列$x$（长度为$m+1$）和$i=1, \ldots, m$定义了分布$p_\theta(x_{i+1} \mid x_{1:i})$。给定由长度为$m$的序列组成的训练集$D$，我们定义标准交叉熵（负对数似然）损失函数：

$$\ell(\theta; D) = \frac{1}{|D| m} \sum_{x \in D} \sum_{i=1}^m -\log p_\theta(x_{i+1} \mid x_{1:i})$$

（注意，Transformer中的单次前向传递会生成所有$i=1, \ldots, m$的$p_\theta(x_{i+1} \mid x_{1:i})$）

具体来说，Transformer为每个位置$i$计算对数几率$o_i \in \mathbb{R}^{\text{vocab_size}}$，这导致：

$$p(x_{i+1} \mid x_{1:i}) = \text{softmax}(o_i)[x_{i+1}] = \frac{\exp(o_i[x_{i+1}])}{\sum_{a=1}^{\text{vocab_size}} \exp(o_i[a])}$$

交叉熵损失通常相对于对数几率向量$o_i \in \mathbb{R}^{\text{vocab_size}}$和目标$x_{i+1}$定义⁶。

实现交叉熵损失需要像softmax一样注意数值问题。

##### 问题（cross_entropy）：实现交叉熵（2分）

**交付物**：编写一个函数来计算交叉熵损失，该函数接收预测的对数几率$o_i$和目标$x_{i+1}$，并计算交叉熵$\ell_i = -\log \text{softmax}(o_i)[x_{i+1}]$。你的函数应处理以下内容：
- 减去最大元素以提高数值稳定性
- 尽可能约简log和exp
- 处理任何额外的批处理维度并返回跨批次的平均值。与第3.3节一样，我们假设类似批次的维度始终在词汇表大小维度之前

实现`[adapters.run_cross_entropy]`，然后运行`uv run pytest -k test_cross_entropy`来测试你的实现。

**困惑度** 交叉熵足以用于训练，但在评估模型时，我们还希望报告困惑度。对于长度为$m$的序列，我们遭受交叉熵损失$\ell_1, \ldots, \ell_m$：

$$\text{perplexity} = \exp\left(\frac{1}{m} \sum_{i=1}^m \ell_i\right)$$

<footer>⁶注意$o_i[k]$指向量$o_i$中索引$k$处的值。这对应于在$x_{i+1}$上的Dirac delta分布与预测的softmax($o_i$)分布之间的交叉熵。</footer>

<footer>29</footer>

## 4.2 SGD优化器

现在有了损失函数，我们将开始探索优化器。最简单的基于梯度的优化器是随机梯度下降（SGD）。我们从随机初始化的参数$\theta_0$开始。然后对于每个步骤$t=0, \ldots, T-1$，我们执行以下更新：

$$\theta_{t+1} \leftarrow \theta_t - \alpha_t \nabla L(\theta_t; B_t)$$

其中$B_t$是从数据集$D$中采样的随机数据批次，学习率$\alpha_t$和批次大小$|B_t|$是超参数。

### 4.2.1 在PyTorch中实现SGD

为了实现我们的优化器，我们将继承PyTorch的`torch.optim.Optimizer`类。Optimizer子类必须实现两个方法：

```python
def __init__(self, params, ...)
```
应初始化你的优化器。这里，`params`将是一个要优化的参数集合（或参数组，以防用户想对模型的不同部分使用不同的超参数，如学习率）。确保将`params`传递给基类的`__init__`方法，它将存储这些参数以供`step`使用。你可以根据优化器接受其他参数（例如学习率是常见的），并将它们作为字典传递给基类构造函数，其中键是你为这些参数选择的名称（字符串）。

```python
def step(self, closure: Optional[Callable] = None)
```
应对参数进行一次更新。在训练循环中，这将在反向传播后调用，因此你可以访问上一批次的梯度。此方法应遍历每个参数张量`p`并就地修改它们，即设置`p.data`（基于梯度`p.grad`）来修改与该参数关联的张量。

PyTorch优化器API有一些微妙之处，因此用示例解释更容易。为了使示例更丰富，我们将实现SGD的一个 slight 变体，其中学习率随训练衰减，从初始学习率$\alpha$开始，并随时间采取越来越小的步骤：

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{t+1}} \nabla L(\theta_t; B_t)$$

让我们看看这个版本的SGD如何作为PyTorch优化器实现：

```python
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # 获取学习率
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # 获取与p关联的状态
                t = state.get("t", 0)  # 从状态获取迭代次数，或初始值
                grad = p.grad.data  # 获取损失相对于p的梯度
                p.data -= lr / math.sqrt(t + 1) * grad  # 就地更新权重张量
                state["t"] = t + 1  # 递增迭代次数

        return loss
```

在`__init__`中，我们将参数传递给优化器，以及默认超参数，传递给基类构造函数（参数可能以组的形式传入，每组有不同的超参数）。如果参数只是单个`torch.nn.Parameter`对象的集合，基构造函数将创建单个组并分配默认超参数。然后，在`step`中，我们遍历每个参数组，然后遍历该组中的每个参数，并应用公式20。这里，我们将迭代次数作为与每个参数关联的状态存储：我们首先读取该值，在梯度更新中使用它，然后更新它。API规定用户可能传入可调用的`closure`以在优化器步骤之前重新计算损失。我们不需要为我们将使用的优化器这样做，但我们添加它以符合

要看到它工作，我们可以使用以下最小训练循环示例：

```python
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)

for t in range(100):
    opt.zero_grad()  # 重置所有可学习参数的梯度
    loss = (weights**2).mean()  # 计算标量损失值
    print(loss.cpu().item())
    loss.backward()  # 运行反向传播，计算梯度
    opt.step()  # 运行优化器
```

这是训练循环的典型结构：在每次迭代中，我们将计算损失并运行优化器步骤。在训练语言模型时，我们的可学习参数将来自模型（在PyTorch中，`m.parameters()`给我们这个集合）。损失将在采样的数据批次上计算，但训练循环的基本结构将是

##### 问题（learning_rate_tuning）：调整学习率（1分）

正如我们将看到的，影响训练最多的超参数之一是学习率。让我们在这个玩具示例中看到这一点。使用上述SGD示例，对学习率使用三个其他值：`1e1`、`1e2`和`1e3`，仅进行10次训练迭代。每个学习率的损失会发生什么？它是衰减得更快、更慢，还是发散（即，在训练过程中增加）？

交付物：包含每个学习率行为的一到两句话回答。

## 4.3 AdamW

现代语言模型通常使用比SGD更复杂的优化器进行训练。最近使用的大多数优化器都是Adam优化器 [Kingma和Ba, 2015]的衍生版本。我们将使用AdamW [Loshchilov和Hutter, 2019]，它在近期工作中广泛使用。AdamW提出了对Adam的修改，通过添加权重衰减（在每次迭代中，我们将参数拉向0）来改进正则化

以一种与梯度更新解耦的方式。我们将按照Loshchilov和Hutter [2019]的算法2实现AdamW。

AdamW是有状态的：对于每个参数，它会跟踪其第一矩和第二矩的运行估计。因此，AdamW使用额外的内存来换取更好的稳定性和收敛性。除了学习率$\alpha$外，AdamW还有一对超参数$(\beta_1, \beta_2)$，用于控制矩估计的更新，以及权重衰减率$\lambda$。典型应用将$(\beta_1, \beta_2)$设置为$(0.9, 0.999)$，但像LLaMA [Touvron等人, 2023]和GPT-3 [Brown等人, 2020]这样的大型语言模型通常使用$(0.9, 0.95)$进行训练。该算法可以编写如下，其中$\epsilon$是一个小值（例如，$10^{-8}$），用于在我们得到极小值时提高数值稳定性：

```python
# 算法1 AdamW优化器

初始化(θ)  # 初始化可学习参数
m ← 0      # 第一矩向量的初始值；与θ形状相同
v ← 0      # 第二矩向量的初始值；与θ形状相同

for t = 1, ..., T do
    采样数据批次B_t
    g ← ∇θℓ(θ; B_t)  # 计算当前时间步的损失梯度
    m ← β1 * m + (1 - β1) * g  # 更新第一矩估计
    v ← β2 * v + (1 - β2) * g²  # 更新第二矩估计
    α_t ← α * sqrt(1 - β2^t) / (1 - β1^t)  # 计算迭代t的调整α
    θ ← θ - α_t * m / (sqrt(v) + ε)  # 更新参数
    θ ← θ - α * λ * θ  # 应用权重衰减
end for
```

注意$t$从1开始。现在你将实现这个优化器。

##### 问题（adamw）：实现AdamW（2分）

**交付物**：将AdamW优化器实现为`torch.optim.Optimizer`的子类。你的类应在`__init__`中接受学习率$\alpha$，以及$\beta$、$\epsilon$和$\lambda$超参数。为了帮助你保持状态，基类Optimizer给你字典`self.state`，它将`nn.Parameter`对象映射到一个字典，存储该参数所需的任何信息（对于AdamW，这将是矩估计）。实现`[adapters.get_adamw_cls]`并确保它通过`uv run pytest -k test_adamw`。

##### 问题（adamwAccounting）：使用AdamW训练的资源核算（2分）

让我们计算运行AdamW所需的内存和计算量。假设我们对每个张量使用float32。

**(a)** 运行AdamW需要多少峰值内存？根据参数、激活值、梯度和优化器状态的内存使用情况分解你的答案。用批次大小和模型超参数（vocab_size、context_length、num_layers、d_model、num_heads）表示你的答案。假设$d_{\text{ff}} = 4 \times d_{\text{model}}$。

为了简化，在计算激活值的内存使用时，只考虑以下组件：
- Transformer块
  - RMSNorm(s)
  - 多头自注意力子层：QKV投影、$Q^\top K$矩阵乘法、softmax、值的加权和、输出投影
  - 位置前馈：$W_1$矩阵乘法、SiLU、$W_2$矩阵乘法
- 最终RMSNorm
- 输出嵌入
- 对logits的交叉熵

交付物：每个参数、激活值、梯度和优化器状态的代数表达式，以及总数。

**(b)** 对于GPT-2XL形状的模型，实例化你的答案以得到仅依赖于批次大小的表达式。在80GB内存内可以使用的最大批次大小是多少？

交付物：看起来像$a \cdot \text{batch\_size} + b$的表达式，其中$a$、$b$是数值，以及表示最大批次大小的数字。

**(c)** 运行AdamW的一步需要多少FLOPs？

交付物：代数表达式，附简要说明。

**(d)** 模型FLOPs利用率（MFU）定义为观察到的吞吐量（每秒token数）相对于硬件理论峰值FLOP吞吐量 [Chowdhery等人, 2022]的比率。NVIDIA A100 GPU对float32操作的理论峰值为19.5 teraFLOP/s。假设你能够获得50% MFU，在单个A100上训练GPT-2XL进行400K步和批次大小1024需要多长时间？遵循Kaplan等人[2020]和Hoffmann等人[2022]，假设反向传播的FLOPs是前向传播的两倍。

交付物：训练所需的天数，附简要说明。

## 4.4 学习率调度

在训练期间导致损失最快下降的学习率值通常会变化。在训练Transformer时，通常使用学习率调度，我们从较大的学习率开始，在初期进行更快的更新，并随着模型训练缓慢衰减到较小的值。在本作业中，我们将实现用于训练LLaMA [Touvron等人, 2023]的余弦退火调度。

调度器只是一个函数，它接收当前步数$t$和其他相关参数（如初始和最终学习率），并返回步数$t$的梯度更新应使用的学习率。最简单的调度是常数函数，它会对任何$t$返回相同的学习率。

余弦退火学习率调度接收(i)当前迭代$t$，(ii)最大学习率$\alpha_{\text{max}}$，(iii)最小（最终）学习率$\alpha_{\text{min}}$，(iv)预热迭代次数$T_w$，和(v)余弦退火迭代次数$T_c$。迭代$t$的学习率定义为：

- **预热**：如果$t < T_w$，则$\alpha_t = \frac{t}{T_w} \alpha_{\text{max}}$
- **余弦退火**：如果$T_w \leq t \leq T_c$，则$\alpha_t = \alpha_{\text{min}} + \frac{1}{2}\left(1 + \cos\left(\frac{t - T_w}{T_c - T_w}\pi\right)\right)(\alpha_{\text{max}} - \alpha_{\text{min}})$
- **退火后**：如果$t > T_c$，则$\alpha_t = \alpha_{\text{min}}$

##### 问题（learning_rate_schedule）：实现带预热的余弦学习率调度（2分）

编写一个函数，接收$t$、$\alpha_{\text{max}}$、$\alpha_{\text{min}}$、$T_w$和$T_c$，并根据上述定义的调度器返回学习率$\alpha_t$。然后实现`[adapters.get_lr_cosine_schedule]`并确保它通过`uv run pytest -k test_get_lr_cosine_schedule`。

## 4.5 梯度裁剪

在训练期间，我们有时会遇到产生大梯度的训练样本，这可能会使训练不稳定。为了缓解这一点，实践中经常采用的一种技术是梯度裁剪。其思想是在每次反向传播后、执行优化器步骤之前，对梯度范数施加限制。

给定所有参数的梯度$g$，我们计算其$l2$范数$\|g\|_2$。如果此范数小于最大值$M$，则我们保持$g$不变；否则，我们按因子$\frac{M}{\|g\|_2 + \epsilon}$缩放$g$（其中添加小的$\epsilon$如$10^{-6}$是为了数值稳定性）。注意，结果的范数将略低于$M$。

##### 问题（gradient_clipping）：实现梯度裁剪（1分）

编写一个函数来实现梯度裁剪。你的函数应接收一个参数列表和一个最大$l2$范数。它应就地修改每个参数的梯度。使用$\epsilon = 10^{-6}$（PyTorch默认值）。然后实现适配器`[adapters.run_gradient_clipping]`并确保它通过`uv run pytest -k test_gradient_clipping`。

# 5 训练循环

现在，我们终于将我们构建的主要组件整合到你实现的主训练脚本中。能够轻松启动具有不同超参数的训练运行将会有所回报（例如，通过将它们作为命令行参数），因为你将多次执行这些操作，以研究不同选择如何影响

## 5.1 数据加载器

分词后的数据（例如，你在`tokenizer_experiments`中准备的数据）是一个单一的标记序列$x = (x_1, \ldots, x_n)$。尽管源数据可能由单独的文档组成（例如，不同的网页或源代码文件），但常见的做法是将它们全部连接成一个标记序列，并在它们之间添加分隔符（例如`<|endoftext|>`标记）。

数据加载器将此转换为一批批数据流，其中每个批次由$B$个长度为$m$的序列组成，以及对应的下一个标记，长度也为$m$。例如，对于$B=1, m=3$，$([x_2, x_3, x_4], [x_3, x_4, x_5])$将是一个可能的批次。

以这种方式加载数据简化了训练，原因有几个。首先，任何$1 \leq i < n - m$都提供一个有效的训练序列，因此采样序列是简单的。由于所有训练序列具有相同的长度，无需填充输入序列，这提高了硬件利用率（也通过增加批次大小$B$）。最后，我们也无需完全加载整个数据集来采样训练数据，这使得处理可能无法放入内存的大型数据集变得容易。

##### 问题（data_loading）：实现数据加载（2分）

**交付物**：编写一个函数，接收一个numpy数组`x`（包含标记ID的整数数组）、`batch_size`、`context_length`和一个PyTorch设备字符串（例如`'cpu'`或`'cuda:0'`），并返回一对张量：采样的输入序列和相应的下一个标记目标。两个张量的形状都应为`(batch_size, context_length)`，包含标记ID，并且都应放置在请求的设备上。

为了根据我们提供的测试测试你的实现，你首先需要在`[adapters.run_get_batch]`处实现测试适配器。然后运行`uv run pytest -k test_get_batch`来测试你的实现。

#### 低资源/降规模提示：在CPU或Apple Silicon上进行数据加载

如果你计划在CPU或Apple Silicon上训练你的LM，你需要将数据移动到正确的设备（同样，稍后你也应该对模型使用相同的设备）。

如果你在CPU上，可以使用`'cpu'`设备字符串，在Apple Silicon（M*芯片）上，你可以使用`'mps'`设备字符串。

更多关于MPS的信息，请查看以下资源：
- [https://developer.apple.com/metal/pytorch/](https://developer.apple.com/metal/pytorch/)
- [https://pytorch.org/docs/main/notes/mps.html](https://pytorch.org/docs/main/notes/mps.html)

**如果数据集太大无法加载到内存中怎么办？** 我们可以使用一个名为`mmap`的Unix系统调用，它将磁盘上的文件映射到虚拟内存，并在访问该内存位置时延迟加载文件内容。因此，你可以"假装"将整个数据集加载到内存中。NumPy通过`np.memmap`实现这一点（或者如果你最初使用`np.save`保存数组，可以使用`np.load`的`mmap_mode='r'`标志），这将返回一个类似NumPy数组的对象，在你访问它们时按需加载条目。在训练期间从数据集（即NumPy数组）采样时，请确保使用内存映射模式加载数据集（通过`np.memmap`或`np.load`的`mmap_mode='r'`标志，取决于你如何保存数组）。确保还指定与你加载的数组匹配的dtype。显式验证内存映射的数据看起来正确（例如不包含超出预期词汇表大小的值）可能会有所帮助。

<footer>35</footer>

## 5.2 检查点

除了加载数据，我们还需要在训练时保存模型。当运行任务时，我们经常希望能够恢复因某种原因中途停止的训练运行（例如，由于作业超时、机器故障等）。即使一切顺利，我们也可能希望稍后访问中间模型（例如，事后研究训练动态、从训练不同阶段采样）。

检查点应包含恢复训练所需的所有状态。我们当然至少希望能够恢复模型权重。如果使用有状态优化器（如AdamW），我们还需要保存优化器的状态（例如，对于AdamW，是矩估计）。最后，为了恢复学习率调度，我们需要知道我们停止时的迭代次数。PyTorch使保存所有这些变得容易：每个`nn.Module`都有`state_dict()`方法，返回包含所有可学习权重的字典；稍后我们可以用其姐妹方法`load_state_dict()`恢复这些权重。任何`nn.optim.Optimizer`也是如此。最后，`torch.save(obj, dest)`可以将对象（例如，包含张量在某些值中的字典，但也包含像整数这样的常规Python对象）转储到文件（路径）或类文件对象，然后可以使用`torch.load`加载回内存

##### 问题（checkpointing）：实现模型检查点（1分）

实现以下两个函数来加载和保存检查点：

```python
def save_checkpoint(model, optimizer, iteration, out)
```
应将前三个参数的所有状态转储到类文件对象`out`中。你可以使用`model`和`optimizer`的`state_dict`方法来获取它们的相关状态，并使用`torch.save(obj, out)`将`obj`转储到`out`（PyTorch在这里支持路径或类文件对象）。一个典型的选择是让`obj`成为一个字典，但你可以使用任何你想要的格式，只要你能够加载你的检查点

此函数期望以下参数：
- `model`
- `optimizer`
- `iteration: int`
- `out: str | os.PathLike | typing.BinaryIO`

```python
def load_checkpoint(src, model, optimizer)
```
应从`src`（路径或类文件对象）加载检查点，然后从该检查点恢复`model`和`optimizer`状态。你的函数应返回保存到检查点的迭代次数。你可以使用`torch.load(src)`恢复你在`save_checkpoint`实现中保存的内容，并在`model`和`optimizer`中使用`load_state_dict`方法将它们恢复到之前的状态

此函数期望以下参数：
- `src: str | os.PathLike | typing.BinaryIO`
- `model`
- `optimizer`

实现适配器`[adapters.run_save_checkpoint]`和`[adapters.run_load_checkpoint]`，并确保它们通过`uv run pytest -k test_checkpointing`。

## 5.3 训练循环

现在，终于将你实现的所有组件整合到你的主训练脚本中。能够轻松启动具有不同超参数的训练运行将会得到回报（例如，通过将它们作为命令行参数），因为你将多次执行这些操作，以研究不同选择如何影响

##### 问题（training_together）：整合所有内容（4分）

**交付物**：编写一个脚本，运行训练循环以在用户提供的输入上训练你的模型。特别是，我们建议你训练脚本至少允许以下内容：

- 能够配置和控制各种模型和优化器超参数
- 使用`np.memmap`高效加载训练和验证大型数据集
- 将检查点序列化到用户指定的路径
- 定期记录训练和验证性能（例如，记录到控制台和/或外部服务，如Weights and Biases）

wandb.ai

<footer>37</footer>

# 6 生成文本

现在我们可以训练模型了，我们需要的最后一块是能够从模型生成文本。回顾语言模型接收一个（可能已批处理的）长度为`sequence_length`的整数序列，并生成一个大小为`sequence_length × vocab_size`的矩阵，其中序列的每个元素都是预测该位置之后下一个词的概率分布。现在我们将编写几个函数，将其转换为新序列的采样方案。

**Softmax** 按照标准惯例，语言模型的输出是最终线性层（"logits"），因此我们需要通过softmax操作将其转换为归一化概率，我们之前在公式10中见过这一点。

**解码** 为了从模型生成文本（解码），我们将为模型提供前缀标记序列（"提示"），并要求它生成词汇表上的概率分布，预测序列中的下一个词。然后，我们将从此分布中采样以确定下一个输出标记。

具体来说，解码过程的一步应接收序列$x_{1,\ldots,t}$并通过以下方程返回标记$x_{t+1}$：

$$
\begin{aligned}
P(x_{t+1} = i \mid x_{1\ldots t}) &= \frac{\exp(v_i)}{\sum_j \exp(v_j)} \\
v &= \text{TransformerLM}(x_{1\ldots t})_t \in \mathbb{R}^{\text{vocab_size}}
\end{aligned}
$$

其中TransformerLM是我们的模型，它接收长度为`sequence_length`的序列作为输入，并生成大小为`sequence_length × vocab_size`的矩阵，我们取该矩阵的最后一个元素，因为我们要寻找位置$t$处的下一个词预测。

通过重复从这些单步条件中采样（将我们先前生成的输出标记附加到下一个解码时间步的输入），直到我们生成序列结束标记`<|endoftext|>`（或用户指定的最大生成标记数），这为我们提供了一个基本解码器。

**解码器技巧** 我们将试验小模型，小模型有时会产生质量非常低的文本。两个简单的解码器技巧可以帮助解决这些问题。首先，在**温度缩放**中，我们用温度参数$\tau$修改softmax，新的softmax为

$$\text{softmax}(v, \tau)_i = \frac{\exp(v_i / \tau)}{\sum_{j=1}^{|\text{vocab_size}|} \exp(v_j / \tau)}$$

注意，设置$\tau \to 0$会使$v$中最大元素占主导地位，softmax的输出变为集中在该最大元素上的独热向量。

其次，另一个技巧是**核采样**或**top-p采样**，其中我们通过截断低概率词来修改采样分布。令$q$为我们从（温度缩放的）softmax中得到的概率分布，大小为`vocab_size`。带有超参数$p$的核采样根据以下方程生成下一个标记：

$$P(x_{t+1} = i \mid q) = \begin{cases}
\frac{q_i}{\sum_{j \in V(p)} q_j} & \text{if } i \in V(p) \\
0 & \text{otherwise}
\end{cases}$$

其中$V(p)$是最小的索引集合，使得$\sum_{j \in V(p)} q_j \geq p$。你可以通过首先按大小对概率分布$q$进行排序，并选择最大的词汇元素直到达到目标级别，轻松地计算这个量。

##### 问题（decoding）：解码（3分）

**交付物**：实现从你的语言模型解码的函数。我们建议你支持以下功能：

- 为用户提供的提示生成补全（即，接收一些$x_{1\ldots t}$并采样补全，直到遇到`<|endoftext|>`标记）
- 允许用户控制生成的最大标记数
- 给定所需的温度值，在采样前对预测的下一个词分布应用softmax温度缩放
- 给定用户指定的阈值进行top-p采样（Holtzman等人, 2020；也称为核采样）

<footer>39</footer>

<footer>39</footer>

## 7.1 如何运行实验和交付物

理解Transformer架构组件背后原理的最佳方式是实际修改它并亲自运行。没有什么能替代动手

为此，能够快速、一致地进行实验，并记录你所做的事情非常重要。为了快速实验，我们将在小规模模型（1700万参数）和简单数据集（TinyStories）上运行许多实验。为了一致性，你将以系统的方式消融组件并改变超参数，并要求你提交实验日志和每个实验相关的学习曲线

为了提交损失曲线，请确保定期评估验证损失并记录梯度步数和墙钟时间。你可能会发现像Weights and Biases这样的日志基础设施

### 问题（experiment_log）：实验日志（3分）

对于你的训练和评估代码，创建实验跟踪基础设施，允许你跟踪实验以及相对于梯度步数和墙钟时间的损失曲线

交付物：用于实验的日志基础设施代码，以及本作业下面问题的实验日志（你尝试的所有内容的文档）

## 7.2 TinyStories

我们将从一个非常简单且模型训练快速的数据集开始（TinyStories；Eldan和Li, 2023），并且我们可以看到一些有趣的行为。获取此数据集的说明在第1节。TinyStories数据集的一个示例如下

### 示例（tinystories_example）：TinyStories中的一个示例

```
从前有一个叫本的小男孩。本喜欢探索周围的世界。他看到了许多奇妙的东西，比如商店里陈列的美丽花瓶。一天，本走过商店时，发现了一个非常特别的花瓶。当本看到它时，他惊呆了！他说："哇，那真是一个了不起的花瓶！我能买它吗？"店主微笑着说："当然可以。你可以把它带回家，向所有朋友展示它有多棒！"于是本把花瓶带回家，他为此感到非常自豪！他叫朋友们过来，向他们展示这个了不起的花瓶。所有的朋友都认为花瓶很漂亮，无法相信本有多幸运。这就是本如何在商店里找到一个了不起的花瓶的故事！
```

### 超参数调优

我们将告诉你一些基本的超参数，并要求你找到其他一些有效的设置：

- `vocab_size`：10,000。典型的词汇表大小在数万到数十万之间。你应该改变这个，看看词汇表和模型行为
- `context_length`：256。像TinyStories这样的简单数据集可能不需要很长的序列长度，但对于后面的OpenWebText数据，你可能需要改变这个。尝试改变这个，看看对每次迭代运行时间和最终
- `d_model`：512。这比许多小型Transformer论文中使用的768维略小，但这会使事情更快。
- `d_ff`：1344。这大约是$\frac{8}{3}d_{\text{model}}$，同时是64的倍数，这对GPU性能有好处。
- RoPE theta参数$\Theta$：10,000
- 层数和头数：4层，16个头。一起，这将产生约1700万个非嵌入参数，这是一个相当小的Transformer。
- 处理的总token数：327,680,000（你的批次大小 × 总步数 × 上下文长度应大致等于此值）。

你应该通过试错找到以下其他超参数的良好默认值：学习率、学习率预热、其他AdamW超参数$(\beta_1, \beta_2, \epsilon)$和权重衰减。你可以在Kingma和Ba [2015]中找到此类超参数的一些典型选择。

### 整合在一起

现在你可以通过获取训练好的BPE分词器、对训练数据集进行分词，并在你编写的训练循环中运行它来整合所有内容。重要提示：如果你的实现正确且高效，上述超参数应在1个H100 GPU上产生大约30-40分钟的运行时间。如果你的运行时间要长得多，请检查并确保你的数据加载、检查点或验证损失代码没有成为运行时间的瓶颈，并且你的实现已正确批处理。

### 调试模型架构的技巧

我们强烈建议你熟悉IDE的内置调试器（例如VSCode/PyCharm），与使用print语句进行调试相比，这将为你节省时间。如果你使用文本编辑器，可以使用类似pdb的东西。调试模型架构时还有一些其他好的做法：

- 开发任何神经网络架构时的常见第一步是过拟合单个最小批次。如果你的实现正确，你应该能够快速将训练损失驱动到接近零。
- 在各种模型组件中设置断点，并检查中间张量的形状以确保它们符合你的预期。
- 监控激活值、模型权重和梯度的范数，以确保它们没有爆炸或消失。

##### 问题（learning_rate）：调整学习率（3分）（4 H100小时）

学习率是最重要的超参数之一。使用你训练的基础模型，回答以下问题：

**(a)** 对学习率进行超参数扫描，并报告最终损失（如果优化器发散则注明发散）。

交付物：与多个学习率相关的学习曲线。解释你的超参数搜索策略。

交付物：在TinyStories上验证损失（每个token）至多为1.45的模型

#### 低资源/降规模提示：在CPU或Apple Silicon上训练较少的步数

如果你改为在cpu或mps上运行，你应该将处理的总token数减少到40,000,000，这将足以产生相当流畅的文本。你也可以将目标验证损失从1.45增加到

在M3 Max芯片和36GB RAM上使用调整后的学习率运行我们的解决方案代码，我们使用批次大小 × 总步数 × 上下文长度 = 32×5000×256 = 40,960,000个token，在cpu上需要1小时22分钟，在mps上需要36分钟。在步骤5000时，我们实现了

一些额外的技巧：

- 当使用X训练步骤时，我们建议调整余弦学习率衰减调度，使其在精确步骤X处终止其衰减（即，达到最小学习率）
- 当使用mps时，不要使用TF32内核，即不要设置`torch.set_float32_matmul_precision('high')`，正如你在cuda设备上可能会做的那样。我们尝试在mps上启用TF32内核（torch版本2.6.0）并发现后端会使用静默损坏的内核，导致训练不稳定
- 你可以通过`torch.compile`加快训练速度。具体而言：
  - 在cpu上，用`model = torch.compile(model)`编译你的模型
  - 在mps上，你可以使用以下方式在一定程度上优化反向传递：`model = torch.compile(model, backend="aot_eager")`。截至torch版本，Inductor编译在mps上不受支持

**(b)** 民间智慧是最好的学习率在"稳定性的边缘"。研究学习率发散的点与你最佳学习

交付物：增加学习率的学习曲线，包括至少一个发散运行，并分析它如何与收敛相关

现在让我们改变批次大小，看看训练会发生什么。批次大小很重要——它们允许我们通过进行更大的矩阵乘法从GPU获得更高的效率，但我们总是希望批次大小尽可能大吗？让我们运行一些实验来找到

##### 问题（batch_size_experiment）：批次大小变化（1分）（2 H100小时）

将批次大小从1一直变化到GPU内存限制。尝试至少几个中间的批次大小，包括典型大小如64和

交付物：不同批次大小运行的学习曲线。如果需要，应再次优化学习率

交付物：关于批次大小及其影响的发现的几个句子讨论

使用你手中的解码器，我们现在可以从模型生成文本！我们将从模型生成并看看它有多好。作为参考，你应该至少得到与示例一样好的输出

##### 示例（ts_generate_example）：TinyStories语言模型的样本输出

```
从前有一个叫莉莉的漂亮女孩。她喜欢吃口香糖，尤其是大的黑色那种。一天，莉莉的妈妈请她帮忙做晚饭。莉莉非常兴奋！她喜欢帮助她的妈妈。莉莉的妈妈为晚饭做了一锅大汤。莉莉非常高兴并说："谢谢你，妈妈！我爱你。"她帮助妈妈把汤倒入一个大碗中。晚饭后，莉莉的妈妈做了一些美味的汤。莉莉很喜欢！她说："谢谢你，妈妈！这汤太好喝了！"她的妈妈微笑着说："我很高兴你喜欢，莉莉。"他们做完饭后继续一起做饭。
```

#### 低资源/降规模提示：在CPU或Apple Silicon上生成文本

如果你改为使用处理4000万token的低资源配置，你应该会看到生成结果仍然类似英语，但不如上面流畅。例如，我们在处理4000万token的TinyStories语言模型的样本输出如下：

```
从前有一个叫苏的小女孩。苏有一颗他非常喜欢的牙齿。这是他最好的头。一天，苏去散步并遇到了一只瓢虫！他们成了好朋友，在小路上玩耍

"嘿，波利！我们出去吧！"蒂姆说。苏看着天空，发现找到跳舞闪耀的方式很困难。她微笑着同意帮助说话！"

当苏看着天空移动时，它是什么。她
```

这是精确的问题陈述和我们要求的内容：

##### 问题（generate）：生成文本（1分）

使用你的解码器和训练好的检查点，报告模型生成的文本。你可能需要操作解码器参数（温度、top-p等）以获得流畅

交付物：至少256个token的文本转储（或直到第一个`<|endoftext|>`标记），并对该输出的流畅性以及至少两个影响该输出好坏的因素进行简要评论

## 7.3 消融和架构修改

理解Transformer的最佳方式是实际修改它并观察其行为。我们现在将做一些简单的消融和

**消融1：层归一化** 人们常说层归一化对Transformer训练的稳定性很重要。但也许我们想冒险。让我们从每个Transformer块中移除RMSNorm，看看

##### 问题（layer_norm_ablation）：移除RMSNorm并训练（1分）（1 H100小时）

从Transformer中移除所有RMSNorm并训练。在先前最优学习率下会发生什么？你能通过使用较低的学习率获得稳定性吗？

交付物：移除RMSNorm并训练时的学习曲线，以及最佳学习率下的学习

交付物：关于

影响的几句话评论

让我们现在研究另一个乍一看似乎随意的层归一化选择。预归一化Transformer块定义为

```
z = x + MultiHeadedSelfAttention(RMSNorm(x))
y = z + FFN(RMSNorm(z))
```

这是原始Transformer架构的少数"共识"修改之一，原始架构使用了后归一化方法作为

```
z = RMSNorm(x + MultiHeadedSelfAttention(x))
y = RMSNorm(z + FFN(z))
```

让我们回到后归一化方法，看看会发生什么。

##### 问题（pre_norm_ablation）：实现后归一化并训练（1分）（1 H100小时）

将你的预归一化Transformer实现修改为后归一化版本。用后归一化模型训练并观察会发生什么。

交付物：后归一化Transformer的学习曲线，与预归一化Transformer进行比较。

我们看到层归一化对Transformer的行为有重大影响，甚至层归一化的位置也很重要。

**消融2：位置嵌入** 接下来我们将研究位置嵌入对模型性能的影响。具体来说，我们将比较我们的基础模型（使用RoPE）与不包含任何位置嵌入的模型（NoPE）。事实证明，仅解码器Transformer，即我们实现的具有因果掩码的Transformer，理论上可以在不明确提供位置嵌入的情况下推断相对或绝对位置信息 [Tsai等人, 2019; Kazemnejad等人, 2023]。我们现在将实证测试NoPE与RoPE相比表现如何。

##### 问题（no_pos_emb）：实现NoPE（1分）（1 H100小时）

修改你的使用RoPE的Transformer实现，完全移除位置嵌入信息，并观察会发生什么。

交付物：比较RoPE和NoPE性能的学习曲线。

**消融3：SwiGLU vs SiLU** 接下来，我们将跟随Shazeer[2020]的脚步，通过比较有门控线性单元（GLU）的SwiGLU前馈网络与使用SiLU激活函数但没有门控线性单元（GLU）的前馈网络的性能，来测试前馈网络中门控的重要性：

$$\text{FFN}_{\text{SiLU}}(x) = W_2 \text{SiLU}(W_1 x)$$

回顾在我们的SwiGLU实现中，我们将内层前馈网络的维度设置为大约$d_{\text{ff}} = \frac{8}{3} d_{\text{model}}$（同时确保$d_{\text{ff}} \mod 64 = 0$，以利用GPU张量核心）。在你的$\text{FFN}_{\text{SiLU}}$实现中，你应该设置$d_{\text{ff}} = 4 \times d_{\text{model}}$，以近似匹配SwiGLU前馈网络的参数数量（SwiGLU有三个权重矩阵而不是两个）。

##### 问题（swiglu_ablation）：SwiGLU vs SiLU（1分）（1 H100小时）

**交付物**：比较SwiGLU和SiLU前馈网络性能的学习曲线，参数数量大致匹配。

交付物：讨论你的

#### 低资源/降规模提示：GPU资源有限的在线学生应在TinyStories上测试修改

在作业的剩余部分，我们将转向更大规模、更嘈杂的网络数据集（OpenWebText），尝试架构修改，并（可选）向课程

在OpenWebText上训练LM以达到流畅需要很长时间，因此我们建议GPU访问有限的在线学生继续在TinyStories上测试修改（使用验证损失作为评估

## 7.4 在OpenWebText上运行

我们现在将转向从网络爬取创建的标准预训练数据集。OpenWebText [Gokaslan等人, 2019]的一个小样本也作为单个文本文件提供：见第1节了解如何访问

以下是OpenWebText的一个示例。请注意文本更加真实、复杂和多样化。你可能希望浏览训练数据集，以了解网络爬取训练数据的样子

### 示例（owt_example）：OWT中的一个示例

```
Baseball Prospectus技术总监Harry Pavlidis在雇佣Jonathan时冒了风险。

Baseball Prospectus技术总监Harry Pavlidis在雇佣Jonathan Judge时冒了风险。Pavlidis知道，正如Alan Schwarz在《数字游戏》中写道，"美国文化中没有一个角落比棒球运动员的表现被更精确地统计、更热烈地量化。"点击几下，你就能发现Noah Syndergaard的快速球在到达本垒板的过程中每分钟旋转超过2100次，Nelson Cruz在2016年合格击球手中拥有最高的平均击球速度，以及无数其他似乎来自视频游戏或科幻小说的花絮。数据的海洋赋予了一个越来越重要的角色力量：分析

这种赋能伴随着额外的审查——对测量本身，也对背后的人和出版物。通过Baseball Prospectus，Pavlidis了解定量不完美带来的反弹。他还知道网站的接球指标需要重做，这需要一个有学识的头脑——一个能够处理复杂统计建模问题的人——来完成

"他让我们感到害怕。"Harry Pavlidis

Pavlidis根据后者的写作和他们在网站赞助的球场活动中的互动，直觉上认为Judge"明白了"。不久之后，两人一起喝酒聊天。Pavlidis的直觉得到了验证。Judge适合这个职位——更好的是，他愿意。"我和很多人谈过，"Pavlidis说，"他是唯一一个足够勇敢接受它的人

注意：你可能需要为这个重新调整超参数，如学习率或批次大小

##### 问题（main_experiment）：在OWT上实验（2分）（3 H100小时）

使用与TinyStories相同的模型架构和总训练迭代次数，在OpenWebText上训练你的语言模型。这个模型表现如何？

交付物：你的语言模型在OpenWebText上的学习曲线。描述与TinyStories损失的差异——我们应该如何解释这些损失？

交付物：从OpenWebText LM生成的文本，格式与TinyStories输出相同。这段文本的流畅性如何？为什么即使我们拥有与TinyStories相同的模型和计算预算，输出质量仍然更差？

## 7.5 你自己的修改 + 排行榜

恭喜你达到了这一点。你快完成了！现在你将尝试改进Transformer架构，并看看你的超参数和架构如何与其他学生进行

### 排行榜规则

除了以下几点外，没有其他限制：

**运行时** 你的提交最多可以在H100上运行1.5小时。你可以通过在slurm提交中设置`--time=01:30:00`来强制执行

**数据** 你只能使用我们提供的OpenWebText训练数据集

否则，你可以随心所欲。如果你正在寻找一些实现想法，可以查看以下资源：

- 最先进的开源LLM家族，如Llama 3 [Grattafiori等人, 2024]或Qwen 2.5 [Yang等人, 2024]
- NanoGPT速通仓库（[https://github.com/KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)），社区成员在其中发布了小型语言模型预训练的许多有趣修改以"速通"。例如，一个可以追溯到原始Transformer论文的常见修改是将输入和输出嵌入的权重绑定在一起（参见Vaswani等人[2017]（第3.4节）和Chowdhery等人[2022]（第2节））。如果你尝试权重绑定，你可能需要减小嵌入/LM头的标准差

你可能想在OpenWebText的小子集或TinyStories上测试这些，然后再尝试完整的

需要注意的是，我们确实注意到，你在此排行榜中发现的某些修改可能在更大规模的预训练中效果不佳。我们将在课程的扩展定律单元中进一步探讨这个想法

### 问题（leaderboard）：排行榜（6分）（10 H100小时）

你将在上述排行榜规则下训练一个模型，目标是在

交付物：在此记录的最终验证损失，相关的学习曲线清楚地显示墙钟时间x轴小于1.5小时，以及你所做工作的描述。我们期望排行榜提交至少击败5.0损失的朴素基线。在此提交到排行榜：

<footer>46</footer>

# 参考文献

Ronen Eldan和Yuanzhi Li. TinyStories: How small can language models be and still speak coherent English?,

Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, 和 Stefanie Tellex. OpenWebText语料库。http://

Rico Sennrich, Barry Haddow, 和 Alexandra Birch. Neural machine translation of rare words with subword units. In Proc. of ACL,

Changhan Wang, Kyunghyun Cho, 和 Jiatao Gu. Neural machine translation with byte-level subwords,

Philip Gage. A new algorithm for data compression. C Users Journal, 12(2):23–38, February 1994. ISSN

Philip

Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, 和 Ilya Sutskever. Language models are unsupervised multitask learners,

Alec Radford, Karthik Narasimhan, Tim Salimans, 和 Ilya Sutskever. Improving language understanding by generative pre-training,

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, 和 Illia Polosukhin. Attention is all you need. In Proc. of NeurIPS,

Toan Q. Nguyen 和 Julian Salazar. Transformers without tears: Improving the normalization of self-attention. In Proc. of IWSWLT,

Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, 和 Tie-Yan Liu. On layer normalization in the transformer architecture. In Proc. of ICML,

Jimmy Lei Ba, Jamie Ryan Kiros, 和 Geoffrey E. Hinton. Layer normalization,

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, 和 Guillaume Lample. Llama: Open and efficient foundation language models,

Biao Zhang 和 Rico Sennrich. Root mean square layer normalization. In Proc. of NeurIPS,

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad AlDahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emili Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzmán, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu

<footer>48</footer>

Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todar Mihaylov, Tong Xiao, Ujjzal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vítor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Moham-

<footer>48</footer>

mad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, Zhiyu Ma, The llama 3 herd of models, 2024. URL

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zihan Qiu, Qwen2.5 technical report, arXiv preprint

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, Noah Fiedel. PaLM: Scaling language modeling with pathways,

Dan Hendrycks 和 Kevin Gimpel. Bridging nonlinearities and stochastic regularizers with gaussian error linear units,

Stefan Elfwing, Eiji Uchibe, 和 Kenji Doya. Sigmoid-weighted linear units for neural network function approximation in reinforcement learning, 2017. URL

Yann N. Dauphin, Angela Fan, Michael Auli, 和 David Grangier. Language modeling with gated convolutional networks, 2017. URL

Noam Shazeer. GLU variants improve transformer,

Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, 和 Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding,

<footer>49</footer>

Diederik P. Kingma 和 Jimmy Ba. Adam: A method for stochastic optimization. In Proc. of ICLR,

Ilya Loshchilov 和 Frank Hutter. Decoupled weight decay regularization. In Proc. of ICLR,

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Jeffrey Wu, 和 Dario Amodei. Language models are few-shot learners. In Proc. of NeurIPS,

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, 和 Dario Amodei. Scaling laws for neural language models,

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, 和 Laurent Sifre. Training compute-optimal large language models,

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, 和 Yejin Choi. The curious case of neural text degeneration. In Proc. of ICLR,

Yao-Hung Hubert Tsai, Shaojie Bai, Makoto Yamada, Louis-Philippe Morency, 和 Ruslan Salakhutdinov. Transformer dissection: An unified understanding for transformer's attention via the lens of kernel. In Kentaro Inui, Jing Jiang, Vincent Ng, 和 Xiaojun Wan, editors, Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4344–4353, Hong Kong, China, November 2019. Association for Computational Linguistics. doi:10.18653/v1/D19-1443. URL

Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan, Payel Das, 和 Siva Reddy. The impact of positional encoding on length generalization in transformers. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL

<footer>50</footer>

