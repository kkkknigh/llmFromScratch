# **CS336 春季2025 作业1：基础**

## **1 作业概述**

在本作业中，您将从零开始构建训练标准 Transformer 语言模型（LM）所需的所有组件，并训练一些模型。

### **您将实现的内容**
1.  **字节对编码（BPE）分词器**（第2节）
2.  **Transformer 语言模型（LM）**（第3节）
3.  **交叉熵损失函数和 AdamW 优化器**（第4节）
4.  **训练循环**，支持序列化和加载模型及优化器状态（第5节）

### **您将运行的内容**
1.  在 TinyStories 数据集上训练一个 BPE 分词器。
2.  使用训练好的分词器在数据集上运行，将其转换为整数 ID 序列。
3.  在 TinyStories 数据集上训练一个 Transformer LM。
4.  使用训练好的 Transformer LM 生成样本并评估困惑度。
5.  在 OpenWebText 上训练模型，并将获得的困惑度提交到排行榜。

### **您可以使用的工具**
我们期望您从零开始构建这些组件。具体来说，**您不能使用** `torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的任何定义，**除了以下例外**：
*   `torch.nn.Parameter`
*   `torch.nn` 中的容器类（例如，`Module`、`ModuleList`、`Sequential` 等）
*   `torch.optim.Optimizer` 基类

您可以使用任何其他 PyTorch 定义。如果您想使用某个函数或类但不确定是否被允许，欢迎在 Slack 上提问。如果有疑问，请考虑使用它是否会损害作业“从零开始”的精神。

### **关于 AI 工具的声明**
允许使用 ChatGPT 等大型语言模型来询问低级编程问题或关于语言模型的高级概念性问题，但**禁止直接使用它来解决问题**。

我们强烈建议您在完成作业时，禁用 IDE 中的 AI 自动补全功能（例如 Cursor Tab、GitHub CoPilot）（不过，非 AI 的自动补全，例如自动补全函数名，是完全没问题的）。我们发现，AI 自动补全使您更难深入理解内容。

### **代码结构**
所有作业代码以及本文档都在 GitHub 上提供：
`github.com/stanford-cs336/assignment1-basics`

请 `git clone` 该仓库。如果有任何更新，我们会通知您，以便您 `git pull` 获取最新版本。

1.  `cs336_basics/*`：这是您编写代码的地方。注意，这里没有代码——您可以从头开始做任何想做的事情！
2.  `adapters.py`：有一组您的代码必须具备的功能。对于每个功能（例如，缩放点积注意力），通过简单地调用您的代码来填写其实现（例如，`run_scaled_dot_product_attention`）。**注意**：您对 `adapters.py` 的更改不应包含任何实质性逻辑；这是粘合代码。
3.  `test_*.py`：这包含您必须通过的所有测试（例如，`test_scaled_dot_product_attention`），这些测试将调用在 `adapters.py` 中定义的钩子。**不要编辑测试文件**。

### **如何提交**
您将向 Gradescope 提交以下文件：
*   `writeup.pdf`：回答所有书面问题。请排版您的答案。
*   `code.zip`：包含您编写的所有代码。

要提交到排行榜，请向以下仓库提交 PR（Pull Request）：
`github.com/stanford-cs336/assignment1-basics-leaderboard`

请参阅排行榜仓库中的 `README.md` 获取详细的提交说明。

### **数据集来源**
本作业将使用两个预处理过的数据集：**TinyStories** [Eldan and Li, 2023] 和 **OpenWebText** [Gokaslan et al., 2019]。这两个数据集都是单个大型纯文本文件。
*   如果您是与班级一起完成作业，您可以在任何非头节点机器的 `/data` 目录下找到这些文件。
*   如果您在家自学，可以在 `README.md` 中找到下载这些文件的命令。

### **低资源/降尺度提示：初始化**
在整个课程的作业材料中，我们将为在较少或没有 GPU 资源的情况下完成作业的部分提供建议。例如，我们有时会建议缩小数据集或模型规模，或解释如何在 MacOS 集成 GPU 或 CPU 上运行训练代码。您将在蓝色框（像这个一样）中找到这些“**低资源提示**”。即使您是拥有课程机器访问权限的斯坦福在校学生，这些提示也可能帮助您更快地迭代并节省时间，因此我们建议您阅读它们！

---

## **2 字节对编码（BPE）分词器**

在作业的第一部分，我们将训练并实现一个**字节级字节对编码（BPE）分词器** [Sennrich 等人，2016；Wang 等人，2019]。具体来说，我们将任意（Unicode）字符串表示为字节序列，并在此字节序列上训练我们的 BPE 分词器。之后，我们将使用该分词器将文本（字符串）编码为用于语言建模的标记（整数序列）。

### **2.1 Unicode 标准**
Unicode 是一种文本编码标准，它将字符映射到整数码点。截至 Unicode 16.0（2024年9月发布），该标准定义了 168 种文字中的 154,998 个字符。例如，字符 "s" 的码点是 115（通常表示为 U+0073，其中 U+ 是约定前缀，0073 是 115 的十六进制），字符 "𗈛" 的码点是 29275。在 Python 中，您可以使用 `ord()` 函数将单个 Unicode 字符转换为其整数表示。`chr()` 函数则将整数 Unicode 码点转换为包含对应字符的字符串。
```
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

#### **问题 (unicode1)：理解 Unicode (1 分)**
**(a)** `chr(0)` 返回什么 Unicode 字符？
> '\x00'

**(b)** 该字符的字符串表示（`__repr__()`）与其打印表示有何不同？

> 打印为不可见字符，字符串表示\0

**(c)** 当这个字符出现在文本中时会发生什么？在您的 Python 解释器中尝试以下操作可能有助于观察是否符合您的预期：

> ```
> >>> chr(0)
> >>> print(chr(0))
> >>> "this is a test" + chr(0) + "string"
> >>> print("this is a test" + chr(0) + "string")
> ```
> ~~~
> >>> repr(0)
> '0'
> >>> chr(0)
> '\x00'
> >>> print(chr(0))
> 
> >>> "this is a test" + chr(0) + "string"
> 'this is a test\x00string'
> >>> print("this is a test" + chr(0) + "string")
> this is a teststring
> ~~~
>
> 以\_str\_打印出来会被忽略，以\_repr\_调用会显示\x00

### **2.2 Unicode 编码**
虽然 Unicode 标准定义了从字符到码点（整数）的映射，但直接在 Unicode 码点上训练分词器是不切实际的，因为词汇量会过大（约 15 万个条目）且稀疏（因为许多字符非常罕见）。相反，我们将使用一种 **Unicode 编码**，将 Unicode 字符转换为字节序列。Unicode 标准本身定义了三种编码：UTF-8、UTF-16 和 UTF-32，其中 UTF-8 是互联网上占主导地位的编码（超过 98% 的网页）。

要在 Python 中将 Unicode 字符串编码为 UTF-8，我们可以使用 `encode()` 函数。要访问 Python `bytes` 对象底层的字节值，我们可以对其进行迭代（例如，调用 `list()`）。最后，我们可以使用 `decode()` 函数将 UTF-8 字节字符串解码为 Unicode 字符串。

### **2.3 为什么要使用字节级 BPE？**
对文本进行建模有两种常见方法：**字符级**建模和**词级**建模。词级建模的问题在于词汇量大（许多语言有超过 10 万个常用词），并且无法处理未登录词（OOV）。字符级建模不会遇到这些 OOV 问题，因为词汇量很小（例如，一个字母表可能只有 50-100 个字符）。然而，字符级建模的缺点是会产生更长的序列，处理起来更昂贵。例如，一个包含 10 个单词的句子在词级语言模型中可能只有 10 个标记长，但在字符级模型中可能长达 50 个或更多标记（取决于单词的长度）。处理这些更长的序列需要在模型的每个步骤进行更多计算。此外，对字节序列进行语言建模很困难，因为更长的输入序列会在数据中产生长期依赖关系。

**子词分词** 是词级分词器和字节级分词器之间的折中方案。请注意，**字节级分词器**的**词汇表**有 **256** 个条目（字节值为 0 到 255）。子词分词器以更大的词汇量为代价，更好地压缩输入字节序列。例如，如果字节序列 `b'the'` 经常出现在我们的原始文本训练数据中，将其作为条目添加到词汇表中会把这个 3 个标记的序列减少为单个标记。

我们如何选择这些要添加到词汇表中的子词单元？Sennrich 等人 [2016] 提出使用**字节对编码（BPE；Gage, 1994）**，这是一种迭代地用单个新的未使用索引替换（"合并"）最常见的字节对的压缩算法。请注意，该算法向我们的词汇表中添加子词标记，以最大化输入序列的压缩——如果一个单词在我们的输入文本中出现足够多次，它将被表示为一个子词单元。

通过 BPE 构建词汇表的子词分词器通常被称为 **BPE 分词器**。在本作业中，我们将实现一个**字节级 BPE 分词器**，其中的词汇项是字节或合并的字节序列，这在处理未登录词和管理输入序列长度方面让我们两全其美。构建 BPE 分词器词汇表的过程被称为"训练" BPE 分词器。

### **2.4 BPE 分词器训练**
BPE 分词器训练过程包括三个主要步骤。

1.  **词汇表初始化**：分词器词汇表是从字节串标记到整数 ID 的一一映射。由于我们要训练一个字节级 BPE 分词器，我们的初始词汇表就是所有字节的集合。因为有 256 个可能的字节值，我们的初始词汇表大小为 256。

2.  **预分词**：一旦有了词汇表，原则上，您可以计算字节在文本中彼此相邻出现的频率，并从最常见的字节对开始合并它们。然而，这在计算上相当昂贵，因为每次合并我们都必须对整个语料库进行一次完整扫描。此外，直接跨语料库合并字节可能会导致仅标点符号不同的标记（例如，`dog!` 与 `dog.`）。这些标记将获得完全不同的标记 ID，即使它们很可能具有高度的语义相似性（因为它们仅在标点符号上不同）。
    为了避免这种情况，我们对语料库进行**预分词**。您可以将此视为对语料库进行的粗粒度分词，帮助我们统计字符对出现的频率。例如，单词 'text' 可能是一个出现 10 次的预分词单元。在这种情况下，当我们统计字符 't' 和 'e' 相邻出现的次数时，我们会看到单词 'text' 中 't' 和 'e' 是相邻的，并且我们可以将它们的计数增加 10，而不是遍历整个语料库。由于我们训练的是字节级 BPE 模型，每个预分词单元都表示为 UTF-8 字节序列。
    Sennrich 等人 [2016] 的原始 BPE 实现仅按空白字符（即 `s.split(" ")`）进行预分词。相反，我们将使用一个基于正则表达式的预分词器（GPT-2 使用；Radford 等人，2019），来自 github.com/openai/tiktoken/pull/234/files:
    ```
    >>> PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    ```
    交互式地用这个预分词器拆分一些文本，以便更好地理解其行为可能是有用的：
    ```
    >>> # requires 'regex' package
    >>> import regex as re
    >>> re.findall(PAT, "some text that i'll pre-tokenize")
    ['some', 'text', 'that', 'i', 'll', 'pre', '-', 'tokenize']
    ```
    然而，在您的代码中使用它时，您应该使用 `re.finditer`，以避免在构建从预分词单元到其计数的映射时存储预分词后的单词。

3.  **计算 BPE 合并**：现在我们已经将输入文本转换为预分词单元，并将每个预分词单元表示为 UTF-8 字节序列，我们可以计算 **BPE 合并**（即训练 BPE 分词器）。概括地说，BPE 算法迭代地计算每个字节对，并识别出频率最高的对 ("A", "B")。然后，这个最常见对 ("A", "B") 的每次出现都被合并，即替换为一个新的标记 "AB"。这个新的合并标记被添加到我们的词汇表中；因此，BPE 训练后的最终词汇表大小是初始词汇表大小（在我们的例子中是 256）加上训练期间执行的 BPE 合并操作的数量。为了提高 BPE 训练效率，我们不考虑跨越预分词单元边界的对。在计算合并时，通过优先选择字典序更大的对来确定性解决对的频率并列问题。例如，如果对 ("A", "B")、("A", "C")、("B", "ZZ") 和 ("BA", "A") 都具有最高频率，我们将合并 ("BA", "A")：
    ```
    >>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
    ('BA', 'A')
    ```

**特殊标记**：通常，某些字符串（例如 `<|endoftext|>`）用于编码元数据（例如，文档之间的边界）。在编码文本时，通常希望将某些字符串视为"特殊标记"，它们永远不应被拆分为多个标记（即，将始终作为单个标记保留）。例如，序列结束字符串 `<|endoftext|>` 应始终作为单个标记（即单个整数 ID）保留，以便我们知道何时停止从语言模型中生成。这些特殊标记必须添加到词汇表中，以便它们有对应的固定标记 ID。

Sennrich 等人 [2016] 的算法 1 包含了一个低效的 BPE 分词器训练实现（基本上遵循了我们上面概述的步骤）。作为第一个练习，实现并测试这个函数可能有助于测试您的理解。

#### **示例 (bpe_example)：BPE 训练示例**
这是来自 Sennrich 等人 [2016] 的一个程式化示例。考虑一个由以下文本组成的语料库：
```
low low low low lower lower widest widest widest newest newest newest newest newest
```
并且词汇表有一个特殊标记 `<|endoftext|>`。

**词汇表**：我们用特殊标记 `<|endoftext|>` 和 256 个字节值初始化我们的词汇表。

**预分词**：为简单起见并专注于合并过程，我们在此示例中假设预分词只是按空白字符拆分。当我们进行预分词和计数时，我们得到频率表：
`{low: 5, lower: 2, widest: 3, newest: 6}`

方便地，这可以表示为一个 `dict[tuple[bytes], int]`，例如 `{(1,o,w): 5 ...}`。注意，即使在 Python 中单个字节也是一个 `bytes` 对象。Python 中没有 `byte` 类型来表示单个字节，就像没有 `char` 类型来表示单个字符一样。

**合并**：我们首先查看每个连续的字节对，并汇总它们出现的单词的频率 `{1o: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}`。对 ('es') 和 ('st') 并列，所以我们取字典序更大的对 ('st')。然后我们会合并预分词单元，最终得到 `{(1,o,w): 5, (1,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}`。
在第二轮中，我们看到 (e, st) 是最常见的对（计数为 9），我们会将其合并为 `{(1,o,w): 5, (1,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}`。继续这个过程，最终得到的合并序列将是 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'love r']`。
如果我们进行 6 次合并，我们有 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']`，我们的词汇表元素将是 `[<|endoftext|>, [...256 BYTE CHARS], st, est, ow, low, west, ne]`。使用这个词汇表和合并集合，单词 `newest` 将被分词为 `[ne, west]`。

### **2.5 实验 BPE 分词器训练**
让我们在 TinyStories 数据集上训练一个字节级 BPE 分词器。查找/下载数据集的说明可以在第 1 节找到。在您开始之前，我们建议看一下 TinyStories 数据集，了解其中的内容。

**并行化预分词**：您会发现一个主要瓶颈是预分词步骤。您可以使用内置库 `multiprocessing` 并行化您的代码来加速预分词。具体来说，我们建议在并行预分词实现中，对语料库进行分块，同时确保您的块边界出现在特殊标记的开头。您可以直接使用以下链接的起始代码来获取块边界，然后您可以用它来跨进程分配工作：
https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/protekenization_example.py

这种分块方式始终有效，因为我们永远不想跨文档边界合并。就本作业而言，您总是可以这样拆分。不用担心收到一个非常大且不包含 `<|endoftext|>` 的语料库的边缘情况。

**预分词前移除特殊标记**：在使用正则表达式模式（使用 `re.finditer`）运行预分词之前，您应该从语料库（或您的块，如果使用并行实现）中去除所有特殊标记。确保在特殊标记上拆分，使得在它们分隔的文本之间不会发生合并。例如，如果您有一个像 `[Doc 1]<|endoftext|>[Doc 2]` 这样的语料库（或块），您应该在特殊标记 `<|endoftext|>` 处拆分，并分别对 `[Doc 1]` 和 `[Doc 2]` 进行预分词，这样就不会跨越文档边界发生合并。这可以使用 `re.split` 并以 `"|".join(special_tokens)` 作为分隔符来完成（需要小心使用 `re.escape`，因为 `|` 可能出现在特殊标记中）。测试 `test_train_bpe_special_tokens` 将对此进行测试。

**优化合并步骤**：上面程式化示例中 BPE 训练的天真实现速度很慢，因为对于每次合并，它都会遍历所有字节对以识别最频繁的对。然而，每次合并后唯一改变的对计数是与合并对重叠的那些对。因此，可以通过索引所有对的计数并增量更新这些计数来改进 BPE 训练速度，而不是显式地遍历每个字节对来统计对频率。您可以通过这种缓存过程获得显著的加速，不过我们注意到，BPE 训练的合并部分在 Python 中无法并行化。

#### **低资源/降尺度提示：性能分析**
您应该使用像 `cProfile` 或 `scalene` 这样的性能分析工具来识别实现中的瓶颈，并专注于优化它们。

#### **低资源/降尺度提示："降尺度"**
我们建议您不要直接跳到在完整的 TinyStories 数据集上训练分词器，而是首先在数据的一个小子集上训练：一个"调试数据集"。例如，您可以在 TinyStories 验证集上训练分词器，它是 22K 个文档而不是 2.12M 个。这说明了尽可能降尺度以加速开发的一般策略：例如，使用更小的数据集、更小的模型尺寸等。选择调试数据集的大小或超参数配置需要仔细考虑：您希望您的调试集足够大，以拥有与完整配置相同的瓶颈（这样您做的优化就能推广），但又不能大到需要很长时间才能运行。

#### **问题 (train_bpe)：BPE 分词器训练 (15 分)**
**交付物：** 编写一个函数，给定输入文本文件的路径，训练一个（字节级）BPE 分词器。您的 BPE 训练函数应至少处理以下输入参数：
*   `input_path: str`：包含 BPE 分词器训练数据的文本文件路径。
*   `vocab_size: int`：一个正整数，定义最终的最大词汇表大小（包括初始字节词汇表、合并产生的词汇表项以及任何特殊标记）。
*   `special_tokens: list[str]`：要添加到词汇表中的字符串列表。这些特殊标记不会以其他方式影响 BPE 训练。

您的 BPE 训练函数应返回生成的词汇表和合并列表：
*   `vocab: dict[int, bytes]`：分词器词汇表，一个从整数（词汇表中的标记 ID）到字节（标记字节）的映射。
*   `merges: list[tuple[bytes, bytes]]`：训练产生的 BPE 合并列表。每个列表项是一个字节元组 (`<token1>`, `<token2>`)，表示 `<token1>` 与 `<token2>` 被合并。合并应按创建顺序排序。

要针对我们提供的测试来测试您的 BPE 训练函数，您需要首先在 `[adapters.run_train_bpe]` 处实现测试适配器。然后，运行 `uv run pytest tests/test_train_bpe.py`。您的实现应该能够通过所有测试。
可选地（这可能是一个耗时的投资），您可以使用某些系统语言（例如 C++（考虑使用 cppyy）或 Rust（使用 PyO3））来实现训练方法的关键部分。如果您这样做，请注意哪些操作需要复制与直接从 Python 内存读取，并确保留下构建说明，或确保仅使用 `pyproject.toml` 即可构建。另请注意，GPT-2 正则表达式在大多数正则表达式引擎中不受良好支持，并且在大多数支持它的引擎中速度太慢。我们已经验证 Oniguruma 速度合理且支持负向先行断言，但 Python 中的 `regex` 包速度甚至更快。

**(a)** 在 TinyStories 数据集上训练一个字节级 BPE 分词器，最大词汇表大小为 10,000。确保将 TinyStories 的特殊标记 `<|endoftext|>` 添加到词汇表中。将生成的词汇表和合并序列化到磁盘以供进一步检查。训练花费了多少小时和多少内存？词汇表中最长的标记是什么？这合理吗？
> **资源要求：** ≤ 30 分钟（无 GPU），≤ 30GB RAM
> **提示**：您应该能够在预分词期间使用多进程以及以下两个事实，在 2 分钟内完成 BPE 训练：
>     a) `<|endoftext|>` 标记在数据文件中分隔文档。
>     b) `<|endoftext|>` 标记在应用 BPE 合并之前作为特殊情况处理。
> **交付物：** 一到两句话的回答。

**(b)** 对您的代码进行性能分析。分词器训练过程的哪个部分花费时间最多？
> **交付物：** 一到两句话的回答。

接下来，我们将尝试在 OpenWebText 数据集上训练一个字节级 BPE 分词器。和之前一样，我们建议看一下数据集以更好地理解其内容。

#### **问题 (train_bpe_expts_out)：OpenWebText 上的 BPE 训练 (2 分)**
**(a)** 在 OpenWebText 数据集上训练一个字节级 BPE 分词器，最大词汇表大小为 32,000。将生成的词汇表和合并序列化到磁盘以供进一步检查。词汇表中最长的标记是什么？这合理吗？
> **资源要求：** ≤ 12 小时（无 GPU），≤ 100GB RAM
> **交付物：** 一到两句话的回答。

**(b)** 比较并对比在 TinyStories 与 OpenWebText 上训练得到的分词器。
> **交付物：** 一到两句话的回答。

### **2.6 BPE 分词器：编码和解码**
在作业的前一部分，我们实现了一个函数来训练输入文本上的 BPE 分词器，以获得分词器词汇表和 BPE 合并列表。现在，我们将实现一个 BPE 分词器，它加载提供的词汇表和合并列表，并使用它们将文本编码为标记 ID 并从标记 ID 解码文本。

#### **2.6.1 编码文本**
BPE 编码文本的过程与训练 BPE 词汇表的方式类似。有几个主要步骤。

**步骤 1：预分词**。我们首先对序列进行预分词，并将每个预分词单元表示为 UTF-8 字节序列，就像在 BPE 训练中所做的那样。我们将在每个预分词单元内将这些字节合并为词汇表元素，独立处理每个预分词单元（不跨越预分词单元边界合并）。

**步骤 2：应用合并**。然后，我们获取 BPE 训练期间创建的词汇表元素合并序列，并以相同的创建顺序将其应用到我们的预分词单元上。

##### **示例 (bpe_encoding)：BPE 编码示例**
例如，假设我们的输入字符串是 `'the cat ate'`，我们的词汇表是 `{0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b'at'}`，我们学习到的合并是 `[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]`。
首先，我们的预分词器会将此字符串拆分为 `['the', ' cat', ' ate']`。
然后，我们将查看每个预分词单元并应用 BPE 合并。
第一个预分词单元 `'the'` 最初表示为 `[b't', b'h', b'e']`。查看我们的合并列表，我们识别出第一个适用的合并是 `(b't', b'h')`，并使用它将预分词单元转换为 `[b'th', b'e']`。然后，我们回到合并列表，识别下一个适用的合并是 `(b'th', b'e')`，它将预分词单元转换为 `[b'the']`。最后，回顾合并列表，我们看到没有更多的合并适用于该字符串（因为整个预分词单元已合并为单个标记），所以我们完成了 BPE 合并的应用。对应的整数序列是 `[9]`。
对剩余的预分词单元重复此过程，我们看到预分词单元 `' cat'` 在应用 BPE 合并后表示为 `[b' c', b'a', b't']`，变为整数序列 `[7, 1, 5]`。最后一个预分词单元 `' ate'` 在应用 BPE 合并后是 `[b' at', b'e']`，变为整数序列 `[10, 3]`。因此，编码我们输入字符串的最终结果是 `[9, 7, 1, 5, 10, 3]`。

**特殊标记**。您的分词器在编码文本时应能正确处理用户定义的特殊标记（在构造分词器时提供）。

**内存考虑**。假设我们想要对一个无法装入内存的大文本文件进行分词。
为了有效地对这个大文件（或任何其他数据流）进行分词，我们需要将其分解成可管理的块，并依次处理每个块，使得内存复杂度是常数而不是与文本大小成线性关系。在这样做时，我们需要确保一个标记不会跨越块边界，否则我们将得到与将整个序列在内存中进行分词的天真方法不同的分词结果。

#### **2.6.2 解码文本**
要将整数标记 ID 序列解码回原始文本，我们可以简单地查找每个 ID 在词汇表中对应的条目（一个字节序列），将它们连接在一起，然后将字节解码为 Unicode 字符串。
注意，输入的 ID 不能保证映射到有效的 Unicode 字符串（因为用户可以输入任何整数 ID 序列）。如果输入的标记 ID 不能产生有效的 Unicode 字符串，您应该用官方的 Unicode 替换字符 U+FFFD³ 替换格式错误的字节。`bytes.decode` 的 `errors` 参数控制如何处理 Unicode 解码错误，使用 `errors='replace'` 会自动用替换标记替换格式错误的数据。

³ 有关 Unicode 替换字符的更多信息，请参见 en.wikipedia.org/wiki/Specials_(Unicode_block)#Replacement_character。

#### **问题 (tokenizer)：实现分词器 (15 分)**
**交付物：** 实现一个 `Tokenizer` 类，给定一个词汇表和一个合并列表，将文本编码为整数 ID，并将整数 ID 解码为文本。您的分词器还应支持用户提供的特殊标记（如果它们尚不存在，则将其附加到词汇表中）。我们建议以下接口：

*   `def __init__(self, vocab, merges, special_tokens=None)`：根据给定的词汇表、合并列表和（可选的）特殊标记列表构造一个分词器。此函数应接受以下参数：
    *   `vocab: dict[int, bytes]`
    *   `merges: list[tuple[bytes, bytes]]`
    *   `special_tokens: list[str] | None = None`
*   `def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)`：类方法，从序列化的词汇表和合并列表（格式与您的 BPE 训练代码输出相同）以及（可选的）特殊标记列表构造并返回一个 `Tokenizer`。此方法应接受以下额外参数：
    *   `vocab_filepath: str`
    *   `merges_filepath: str`
    *   `special_tokens: list[str] | None = None`
*   `def encode(self, text: str) -> list[int]`：将输入文本编码为标记 ID 序列。
*   `def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]`：给定一个字符串的可迭代对象（例如，一个 Python 文件句柄），返回一个惰性生成标记 ID 的生成器。这对于对我们无法直接加载到内存中的大文件进行内存高效的分词是必需的。
*   `def decode(self, ids: list[int]) -> str`：将标记 ID 序列解码为文本。

要针对我们提供的测试来测试您的 `Tokenizer`，您需要首先在 `[adapters.get_tokenizer]` 处实现测试适配器。然后，运行 `uv run pytest tests/test_tokenizer.py`。您的实现应该能够通过所有测试。

### **2.7 实验**

#### **问题 (tokenizer_experiments)：分词器实验 (4 分)**
**(a)** 从 TinyStories 和 OpenWebText 中各采样 10 个文档。使用您之前训练的 TinyStories 和 OpenWebText 分词器（分别为 10K 和 32K 词汇量），将这些采样的文档编码为整数 ID。每个分词器的压缩比（字节/标记）是多少？
> **交付物：** 一到两句话的回答。

**(b)** 如果您用 TinyStories 分词器对您的 OpenWebText 样本进行分词会发生什么？比较压缩比和/或定性描述发生的情况。
> **交付物：** 一到两句话的回答。

**(c)** 估计您的分词器的吞吐量（例如，字节/秒）。对 Pile 数据集（825GB 文本）进行分词需要多长时间？
> **交付物：** 一到两句话的回答。

**(d)** 使用您的 TinyStories 和 OpenWebText 分词器，将相应的训练和开发数据集编码为整数标记 ID 序列。我们稍后将使用这个来训练我们的语言模型。我们建议将标记 ID 序列化为 `uint16` 数据类型的 NumPy 数组。为什么 `uint16` 是一个合适的选择？

---

## **3 Transformer 语言模型架构**

语言模型将一批整数标记 ID 序列（即形状为 `(batch_size, sequence_length)` 的 `torch.Tensor`）作为输入，并返回一个（批处理的）在词汇表上的归一化概率分布（即形状为 `(batch_size, sequence_length, vocab_size)` 的 PyTorch 张量），其中预测的分布是针对每个输入标记的下一个词。在训练语言模型时，我们使用这些下一个词预测来计算实际下一个词与预测下一个词之间的交叉熵损失。在推理期间从语言模型生成文本时，我们从最后的时间步（即序列中的最后一项）获取预测的下一个词分布，以生成序列中的下一个标记（例如，通过取概率最高的标记、从分布中采样等），将生成的标记添加到输入序列，并重复此过程。

在作业的这一部分，您将从头开始构建这个 Transformer 语言模型。我们将从模型的高层描述开始，逐步详细介绍各个组件。

### **3.1 Transformer LM**
给定一个标记 ID 序列，Transformer 语言模型使用输入嵌入将标记 ID 转换为密集向量，通过 `num_layers` 个 Transformer 块传递嵌入的标记，然后应用一个学习的线性投影（"输出嵌入"或"LM 头"）以产生预测的下一个标记 logits。请参见图 1 的示意图表示。

#### **3.1.1 标记嵌入**
在第一步中，Transformer 将（批处理的）标记 ID 序列嵌入到一个包含标记身份信息的向量序列中（图 1 中的红色块）。

更具体地说，给定一个标记 ID 序列，Transformer 语言模型使用一个标记嵌入层来产生一个向量序列。每个嵌入层接收一个形状为 `(batch_size, sequence_length)` 的整数张量，并产生一个形状为 `(batch_size, sequence_length, d_model)` 的向量序列。

### **3.2 输出归一化和嵌入**
经过 `num_layers` 个 Transformer 块后，我们将获取最终的激活并将其转换为在词汇表上的分布。

我们将实现"预归一化" Transformer 块（详见第 3.5 节），这另外要求在最终的 Transformer 块之后使用层归一化（详见下文），以确保其输出被正确缩放。

在此归一化之后，我们将使用一个标准的学习线性变换，将 Transformer 块的输出转换为预测的下一个标记 logits（参见，例如，Radford 等人 [2018] 方程 2）。

### **3.3 备注：批处理、Einsum 和高效计算**
在整个 Transformer 中，我们将对许多类似批处理的输入应用相同的计算。以下是一些示例：
*   **批中的元素**：我们对每个批元素应用相同的 Transformer 前向操作。
*   **序列长度**：像 RMSNorm 和前馈这样的"逐位置"操作在序列的每个位置上完全相同地操作。
*   **注意力头**：注意力操作在"多头"注意力操作中跨注意力头进行批处理。

拥有一种既完全利用 GPU，又易于阅读和理解的方式来执行此类操作是很有用的。许多 PyTorch 操作可以在张量的开头接受额外的"批状"维度，并高效地跨这些维度重复/广播操作。

例如，假设我们正在进行一个逐位置的、批处理的操作。我们有一个形状为 `(batch_size, sequence_length, d_model)` 的"数据张量" \(D\)，并且我们想要对一个形状为 `(d_model, d_model)` 的矩阵 \(A\) 进行批处理的向量-矩阵乘法。在这种情况下，`D @ A` 将执行一个批处理矩阵乘法，这是 PyTorch 中的一个高效原语，其中 `(batch_size, sequence_length)` 维度是批处理过的。

因此，假设您的函数可能会被赋予额外的批状维度，并将这些维度保持在 PyTorch 形状的开头是有帮助的。为了以便于这种批处理的方式组织张量，可能需要使用许多步的 `view`、`reshape` 和 `transpose` 来调整它们的形状。这可能有点麻烦，并且常常难以阅读代码在做什么以及张量的形状是什么。

一个更符合人体工程学的选择是在 `torch.einsum` 中使用 einsum 表示法，或者使用与框架无关的库，如 `einops` 或 `einx`。两个关键操作是 `einsum`（可以对输入张量的任意维度进行张量收缩）和 `rearrange`（可以重新排序、连接和拆分任意维度）。事实证明，机器学习中的几乎所有操作都是维度调整和张量收缩的某种组合，偶尔还有（通常是逐点的）非线性函数。这意味着，当使用 einsum 表示法时，您的很多代码可以更具可读性和灵活性。

我们强烈建议为本课程学习并使用 einsum 表示法。以前未接触过 einsum 表示法的学生应使用 `einops`（文档在此），已经熟悉 `einops` 的学生应学习更通用的 `einx`（在此）。这两个包都已安装在我们提供的环境中。

我们在这里给出一些如何使用 einsum 表示法的示例。这些是对 `einops` 文档的补充，您应该先阅读它。

**示例 (einstein_example1)：使用 einops.einsum 进行批处理矩阵乘法**
```python
import torch
from einops import rearrange, einsum

# 基本实现 Y = D @ A.T
# 难以分辨输入和输出形状及其含义。
# D 和 A 可以有什么形状？其中哪些有意外行为？

# Einsum 是自文档化且健壮的
# D @ A -> Y
Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")
# 或者，一个批处理版本，其中 D 可以有任何前导维度，但 A 是受约束的。
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
```

**示例 (einstein_example2)：使用 einops.rearrange 进行广播操作**
我们有一批图像，对于每张图像，我们希望基于某个缩放因子生成 10 个变暗版本：
```python
images = torch.randn(64, 128, 128, 3) # (batch, height, width, channel)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)

# 重塑和相乘
dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images = images_rearr + dim_value

# 或者一步完成：
dimmed_images = einsum(images, dim_by, "batch height width channel, dim_value -> batch dim_value height width channel")
```

**示例 (einstein_example3)：使用 einops.rearrange 进行像素混合**
假设我们有一批图像，表示为形状为 `(batch, height, width, channel)` 的张量，我们想对图像的所有像素执行线性变换，但该变换应对每个通道独立进行。我们的线性变换表示为一个形状为 `(height × width, height × width)` 的矩阵 \(B\)。
```python
channels_last = torch.randn(64, 32, 32, 3) # (batch, height, width, channel)
B = torch.randn(32*32, 32*32)

# 使用 einops：
height = width = 32
# Rearrange 替代了笨拙的 torch view + transpose
channels_first = rearrange(channels_last, "batch height width channel -> batch channel (height width)")
channels_first_transformed = einsum(channels_first, B, "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out")
channels_last_transformed = rearrange(channels_first_transformed, "batch channel (height width) -> batch height width channel", height=height, width=width)

# 或者，如果您想更简洁：使用 einx.dot（einx 中等价于 einops.einsum）一步完成
height = width = 32
channels_last_transformed = einx.dot("batch row_in col_in channel, (row_out col_out) (row_in col_in) -> batch row_out col_out channel", channels_last, B, col_in=width, col_out=width)
```

第一个实现可以通过在前后添加注释来指示输入和输出形状，但这样很笨拙且容易出错。有了 einsum 表示法，**文档就是实现本身！**

Einsum 表示法可以处理任意输入批处理维度，还具有**自文档化**的关键优势。在使用 einsum 表示法的代码中，输入和输出张量的相关形状要清晰得多。对于其余的运算，您可以考虑使用张量类型提示，例如使用 `jaxtyping` 库（不特定于 Jax）。

我们将在作业 2 中更多讨论使用 einsum 表示法的性能影响，但现在要知道它们几乎总是比替代方案更好！

#### **3.3.1 数学符号和内存顺序**
许多机器学习论文在其表示法中使用行向量，这与 NumPy 和 PyTorch 默认使用的行主序内存布局配合得很好。对于行向量，线性变换看起来像：
\[y = xW^{\top}, \quad (1)\]
其中 \(W\in \mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}\) 是行主序的，\(x\in \mathbb{R}^{1\times d_{\mathrm{in}}}\) 是行向量。

在线性代数中，通常更常见的是使用列向量，其中线性变换看起来像：
\[y = Wx, \quad (2)\]
给定一个行主序的 \(W\in \mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}\) 和列向量 \(x\in \mathbb{R}^{d_{\mathrm{in}}}\)。在本作业中，我们将**在数学表示法中使用列向量**，因为这样通常更容易理解数学。您应该记住，如果您想使用普通的矩阵乘法表示法，您将不得不使用行向量约定来应用矩阵，因为 PyTorch 使用行主序内存排序。如果您使用 einsum 进行矩阵运算，这应该不是问题。

### **3.4 基本构建块：线性和嵌入模块**

#### **3.4.1 参数初始化**
有效训练神经网络通常需要仔细初始化模型参数——糟糕的初始化可能导致不良行为，例如梯度消失或爆炸。预归一化 Transformer 对初始化异常鲁棒，但它们仍然会对训练速度和收敛产生重大影响。由于本作业已经很长，我们将把细节留到作业 3，现在只给您一些在大多数情况下应该效果良好的近似初始化。目前，请使用：
*   **线性权重**：\(\mathcal{N}\left(\mu = 0,\sigma^{2} = \frac{2}{d_{\mathrm{in}} + d_{\mathrm{out}}}\right)\)，在 \([- 3\sigma ,3\sigma ]\) 处截断。
*   **嵌入**：\(\mathcal{N}\left(\mu = 0,\sigma^{2} = 1\right)\)，在 \([- 3,3]\) 处截断。
*   **RMSNorm**：初始化为 1。

您应该使用 `torch.nn.init.trunc_normal_` 来初始化截断正态权重。

#### **3.4.2 线性模块**
线性层是 Transformer 和一般神经网络的基本构建块。首先，您将实现自己的继承自 `torch.nn.Module` 的 `Linear` 类，并执行线性变换：
\[y = Wx. \quad (3)\]
注意，我们**不包含偏置项**，遵循大多数现代 LLM 的做法。

#### **问题 (linear)：实现线性模块 (1 分)**
**交付物：** 实现一个继承自 `torch.nn.Module` 并执行线性变换的 `Linear` 类。您的实现应遵循 PyTorch 内置 `nn.Linear` 模块的接口，除了没有偏置参数或参数。我们建议以下接口：
*   `def __init__(self, in_features, out_features, device=None, dtype=None)`：构造一个线性变换模块。此函数应接受以下参数：
    *   `in_features: int`：输入的最终维度
    *   `out_features: int`：输出的最终维度
    *   `device: torch.device | None = None`：存储参数的设备
    *   `dtype: torch.dtype | None = None`：参数的数据类型
*   `def forward(self, x: torch.Tensor) -> torch.Tensor`：对输入应用线性变换。

确保：
*   子类化 `nn.Module`
*   调用超类构造函数
*   将参数构造并存储为 \(W\)（而不是 \(W^{\top}\)）以符合内存顺序，将其放入 `nn.Parameter`
*   当然，不要使用 `nn.Linear` 或 `nn.functional.linear`

对于初始化，请使用上面的设置以及 `torch.nn.init.trunc_normal_` 来初始化权重。

要测试您的 `Linear` 模块，请在 `[adapters.run_linear]` 处实现测试适配器。适配器应将给定的权重加载到您的 `Linear` 模块中。您可以为此使用 `Module.load_state_dict`。然后，运行 `uv run pytest -k test_linear`。

#### **3.4.3 嵌入模块**
如上所述，Transformer 的第一层是一个嵌入层，它将整数标记 ID 映射到维度为 `d_model` 的向量空间中。我们将实现一个继承自 `torch.nn.Module` 的自定义 `Embedding` 类（因此您不应使用 `nn.Embedding`）。前向方法应通过使用形状为 `(batch_size, sequence_length)` 的 `torch.LongTensor` 标记 ID 索引到形状为 `(vocab_size, d_model)` 的嵌入矩阵中，为每个标记 ID 选择嵌入向量。

#### **问题 (embedding)：实现嵌入模块 (1 分)**
**交付物：** 实现继承自 `torch.nn.Module` 并执行嵌入查找的 `Embedding` 类。您的实现应遵循 PyTorch 内置 `nn.Embedding` 模块的接口。我们建议以下接口：
*   `def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)`：构造一个嵌入模块。此函数应接受以下参数：
    *   `num_embeddings: int`：词汇表大小
    *   `embedding_dim: int`：嵌入向量的维度，即 \(d_{\mathrm{model}}\)
    *   `device: torch.device | None = None`：存储参数的设备
    *   `dtype: torch.dtype | None = None`：参数的数据类型
*   `def forward(self, token_ids: torch.Tensor) -> torch.Tensor`：查找给定标记 ID 的嵌入向量。

确保：
*   子类化 `nn.Module`
*   调用超类构造函数
*   将您的嵌入矩阵初始化为 `nn.Parameter`
*   以 `d_model` 作为最终维度存储嵌入矩阵
*   当然，不要使用 `nn.Embedding` 或 `nn.functional.embedding`

再次，使用上面的设置进行初始化，并使用 `torch.nn.init.trunc_normal_` 来初始化权重。

要测试您的实现，请在 `[adapters.run_embedding]` 处实现测试适配器。然后，运行 `uv run pytest -k test_embedding`。

### **3.5 预归一化 Transformer 块**
每个 Transformer 块有两个子层：一个**多头自注意力机制**和一个**逐位置前馈网络** (Vaswani 等人, 2017, 第 3.1 节)。

在原始的 Transformer 论文中，模型在两个子层周围使用残差连接，然后是层归一化。这种架构通常被称为"**后归一化**" Transformer，因为层归一化应用于子层的输出。然而，一系列工作发现，将层归一化从每个子层的输出移动到每个子层的输入（在最终的 Transformer 块之后还有一个额外的层归一化）可以提高 Transformer 的训练稳定性 [Nguyen and Salazar, 2019, Xiong 等人, 2020]——请参见图 2 中这种"**预归一化**" Transformer 块的可视化表示。然后，每个 Transformer 块子层的输出通过残差连接添加到子层输入（Vaswani 等人, 2017, 第 5.4 节）。预归一化的一个直观理解是，从输入嵌入到 Transformer 的最终输出有一个清晰的"残差流"，没有任何归一化，这据称可以改善梯度流。这种预归一化 Transformer 现在是当今语言模型中使用的标准（例如，GPT-3、LLaMA、PaLM 等），因此我们将实现这种变体。我们将逐步介绍预归一化 Transformer 块的每个组件，按顺序实现它们。

#### **3.5.1 均方根层归一化**
Vaswani 等人 [2017] 的原始 Transformer 实现使用层归一化 [Ba 等人, 2016] 来归一化激活。遵循 Touvron 等人 [2023]，我们将使用**均方根层归一化（RMSNorm；Zhang and Sennrich, 2019, 方程 4）** 进行层归一化。给定激活向量 \(a \in \mathbb{R}^{d_{\mathrm{model}}}\)，RMSNorm 将按以下方式重新缩放每个激活 \(a_i\)：
\[\mathrm{RMSNorm}(a_i) = \frac{a_i}{\mathrm{RMS}(a)} g_i, \quad (4)\]
其中 \(\mathrm{RMS}(a) = \sqrt{\frac{1}{d_{\mathrm{model}}}\sum_{i = 1}^{d_{\mathrm{model}}}a_i^2 + \epsilon}\)。这里，\(g_i\) 是一个可学习的"增益"参数（总共有 \(d_{\mathrm{model}}\) 个这样的参数），\(\epsilon\) 是一个通常固定为 1e-5 的超参数。

您应将输入上转换为 `torch.float32`，以防止平方输入时溢出。总体而言，您的前向方法应如下所示：
```python
in_dtype = x.dtype
x = x.to(torch.float32)
# 您在此处执行 RMSNorm 的代码
result = ... # 将结果转换回原始数据类型
return result.to(in_dtype)
```

#### **问题 (rmsnorm)：均方根层归一化 (1 分)**
**交付物：** 将 RMSNorm 实现为 `torch.nn.Module`。我们建议以下接口：
*   `def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)`：构造 RMSNorm 模块。此函数应接受以下参数：
    *   `d_model: int`：模型的隐藏维度
    *   `eps: float = 1e-5`：数值稳定性的 epsilon 值
    *   `device: torch.device | None = None`：存储参数的设备
    *   `dtype: torch.dtype | None = None`：参数的数据类型
*   `def forward(self, x: torch.Tensor) -> torch.Tensor`：处理形状为 `(batch_size, sequence_length, d_model)` 的输入张量并返回相同形状的张量。

**注意：** 如上所述，在执行归一化之前，请记住将输入上转换为 `torch.float32`（之后下转换回原始数据类型）。要测试您的实现，请在 `[adapters.run_rmsnorm]` 处实现测试适配器。然后，运行 `uv run pytest -k test_rmsnorm`。

#### **3.5.2 逐位置前馈网络**
在原始 Transformer 论文（Vaswani 等人 [2017] 第 3.3 节）中，Transformer 前馈网络由两个线性变换组成，中间有一个 ReLU 激活（ \(\mathrm{ReLU}(x) = \max (0,x)\) ）。内部前馈层的维度通常是输入维度的 4 倍。

然而，现代语言模型与这个原始设计相比，往往包含两个主要变化：它们使用另一种激活函数并采用门控机制。具体来说，我们将实现在 Llama 3 [Grattafiori 等人, 2024] 和 Qwen 2.5 [Yang 等人, 2024] 等 LLM 中采用的"**SwiGLU**"激活函数，它将 SiLU（通常称为 Swish）激活与称为门控线性单元（GLU）的门控机制结合起来。我们还将省略有时在线性层中使用的偏置项，遵循 PaLM [Chowdhery 等人, 2022] 和 LLaMA [Touvron 等人, 2023] 之后的大多数现代 LLM 的做法。

**SiLU 或 Swish 激活函数** [Hendrycks and Gimpel, 2016, Elfwing 等人, 2017] 定义如下：
\[\mathrm{SiLU}(x) = x\cdot \sigma (x) = \frac{x}{1 + e^{-x}} \quad (5)\]
如图 3 所示，SiLU 激活函数类似于 ReLU 激活函数，但在零处是平滑的。

**门控线性单元（GLU）** 最初由 Dauphin 等人 [2017] 定义为一个通过 sigmoid 函数的线性变换与另一个线性变换的逐元素乘积：
\[\mathrm{GLU}(x,W_1,W_2) = \sigma (W_1x)\odot W_2x, \quad (6)\]
其中 \(\odot\) 表示逐元素乘法。门控线性单元被认为可以"通过为梯度提供线性路径同时保留非线性能力，减少深度架构的梯度消失问题。"

将 SiLU/Swish 和 GLU 结合在一起，我们得到了 **SwiGLU**，我们将用它作为我们的前馈网络：
\[\mathrm{FFN}(x) = \mathrm{SwiGLU}(x,W_1,W_2,W_3) = W_2(\mathrm{SiLU}(W_1x)\odot W_3x), \quad (7)\]
其中 \(x\in \mathbb{R}^{d_{\mathrm{model}}}\)， \(W_{1},W_{3}\in \mathbb{R}^{d_{\mathrm{ff}}\times d_{\mathrm{model}}}\)， \(W_{2}\in \mathbb{R}^{d_{\mathrm{model}}\times d_{\mathrm{ff}}}\)，并且规范地，\(d_{\mathrm{ff}} = \frac{8}{3} d_{\mathrm{model}}\)。

Shazeer [2020] 首次提出将 SiLU/Swish 激活与 GLU 结合，并进行了实验，表明 SwiGLU 在语言建模任务上优于像 ReLU 和 SiLU（无门控）这样的基线。稍后在作业中，您将比较 SwiGLU 和 SiLU。尽管我们提到了这些组件的一些启发式论点（论文中提供了更多支持证据），但保持经验主义的观点是好的：Shazeer 论文中现在有一句名言是：
> 我们没有解释为什么这些架构似乎有效；我们将它们的成功，以及其他一切，归功于神的恩典。

#### **问题 (positionwise_feedforward)：实现逐位置前馈网络 (2 分)**
**交付物：** 实现由 SiLU 激活函数和 GLU 组成的 SwiGLU 前馈网络。

**注意：** 在这种情况下，为了数值稳定性，您可以随意在实现中使用 `torch.sigmoid`。

您应该在实现中将 \(d_{\mathrm{ff}}\) 设置为大约 \(\frac{8}{3}\times d_{\mathrm{model}}\)，同时确保内部前馈层的维度是 64 的倍数，以便充分利用您的硬件。要针对我们提供的测试测试您的实现，您需要在 `[adapters.run_swiglu]` 处实现测试适配器。然后，运行 `uv run pytest -k test_swiglu` 来测试您的实现。

#### **3.5.3 相对位置嵌入**
为了将位置信息注入模型，我们将实现**旋转位置嵌入** [Su 等人, 2021]，通常称为 **RoPE**。对于标记位置 \(i\) 处的给定查询标记 \(q^{(i)} = W_{q}x^{(i)}\in \mathbb{R}^{d}\)，我们将应用一个成对旋转矩阵 \(R^i\)，得到 \(q^{(i)} = R^i q^{(i)} = R^i W_q x^{(i)}\)。这里，\(R^i\) 将嵌入元素对 \(q_{2k - 1:2k}^{(i)}\) 作为 2D 向量旋转角度 \(\theta_{i,k} = \frac{i}{10000^{2k / d}}\)，其中 \(k\in \{1,\ldots ,d / 2\}\)，\(d\) 是嵌入维度。因此，我们可以将 \(R^i\) 视为一个大小为 \(d\times d\) 的块对角矩阵，块 \(R_k^i\) 对应 \(k\in \{1,\ldots ,d / 2\}\)，其中：
\[R_{k}^{i} = \left[ \begin{array}{ccc}\cos (\theta_{i,k}) & -\sin (\theta_{i,k})\\ \sin (\theta_{i,k}) & \cos (\theta_{i,k}) \end{array} \right]. \quad (8)\]
因此我们得到完整的旋转矩阵：
\[R^{i} = \left[ \begin{array}{ccccc}R_{1}^{i} & 0 & 0 & \ldots & 0\\ 0 & R_{2}^{i} & 0 & \ldots & 0\\ 0 & 0 & R_{3}^{i} & \ldots & 0\\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \ldots & R_{d / 2}^{i} \end{array} \right], \quad (9)\]
其中 0 表示 \(2\times 2\) 零矩阵。虽然可以构造完整的 \(d\times d\) 矩阵，但一个好的解决方案应该利用此矩阵的特性来更高效地实现变换。由于我们只关心给定序列内标记的相对旋转，我们可以在不同层和不同批次中重用为 \(\cos (\theta_{i,k})\) 和 \(\sin (\theta_{i,k})\) 计算的值。如果您想优化它，可以使用一个由所有层引用的单一 RoPE 模块，并且它可以有一个在初始化期间创建的 sin 和 cos 值的 2D 预计算缓冲区，使用 `self.register_buffer(persistent=False)`，而不是 `nn.Parameter`（因为我们不想学习这些固定的余弦和正弦值）。然后对 \(k^{(j)}\) 执行与我们为 \(q^{(i)}\) 所做的完全相同的旋转过程，旋转相应的 \(R^i\)。注意，此层没有可学习参数。

#### **问题 (rope)：实现 RoPE (2 分)**
**交付物：** 实现一个将 RoPE 应用于输入张量的 `RotaryPositionalEmbedding` 类。建议以下接口：
*   `def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)`：构造 RoPE 模块，并在需要时创建缓冲区。
    *   `theta: float`：RoPE 的 \(\Theta\) 值
    *   `d_k: int`：查询和关键向量的维度
    *   `max_seq_len: int`：将要输入的最大序列长度
    *   `device: torch.device | None = None`：存储缓冲区的设备
*   `def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor`：处理形状为 `(..., seq_len, d_k)` 的输入张量并返回相同形状的张量。注意，您应该容忍具有任意数量批维度的 \(x\)。您应该假设标记位置是一个形状为 `(..., seq_len)` 的张量，指定了 \(x\) 沿序列维度的标记位置。

您应该使用标记位置沿序列维度切片您的（可能预计算的）cos 和 sin 张量。

要测试您的实现，完成 `[adapters.run_rope]` 并确保通过 `uv run pytest -k test_rope`。

#### **3.5.4 缩放点积注意力**
我们现在将实现 Vaswani 等人 [2017]（第 3.2.1 节）中描述的缩放点积注意力。作为初步步骤，注意力操作的定义将使用 **softmax**，这是一个获取未归一化的分数向量并将其转换为归一化分布的操作：
\[\mathrm{softmax}(v)_i = \frac{\exp(v_i)}{\sum_{j = 1}^{n}\exp(v_j)}. \quad (10)\]
注意，对于较大的值，\(\exp (v_i)\) 可能变为 inf（然后，\(\mathrm{inf} / \mathrm{inf} = \mathrm{NaN}\)）。我们可以通过注意到 softmax 操作对向所有输入添加任何常数 \(c\) 保持不变来避免这种情况。我们可以利用这个性质来提高数值稳定性——通常，我们将从 \(o_i\) 的所有元素中减去 \(o_i\) 的最大项，使新的最大项为 0。您现在将实现 softmax，使用这个技巧来提高数值稳定性。

#### **问题 (softmax)：实现 softmax (1 分)**
**交付物：** 编写一个函数，在张量上应用 softmax 操作。您的函数应接受两个参数：一个张量和一个维度 \(i\)，并对输入张量的第 \(i\) 维应用 softmax。输出张量应与输入张量形状相同，但其第 \(i\) 维现在将具有归一化的概率分布。使用减去第 \(i\) 维中所有元素的最大值的技巧来避免数值稳定性问题。

要测试您的实现，完成 `[adapters.run_softmax]` 并确保通过 `uv run pytest -k test_softmax_matches_pytorch`。

我们现在可以如下数学定义注意力操作：
\[\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{Q^{\top}K}{\sqrt{d_k}}\right)V \quad (11)\]

**因果掩码**。您的实现应防止模型关注序列中的未来标记。换句话说，如果模型给定一个标记序列 \(t_1,\ldots ,t_n\)，并且我们想要计算前缀 \(t_1,\ldots ,t_i\)（其中 \(i< n\)）的下一个词预测，模型应该不能访问（关注）位置 \(t_{i + 1},\ldots ,t_n\) 的标记表示，因为在推理期间生成文本时，它将无法访问这些标记（并且这些未来标记泄露了真实下一个词的身份，使语言建模预训练目标变得平凡）。对于输入标记序列 \(t_1,\ldots ,t_n\)，我们可以通过运行多头自注意力 \(n\) 次（针对序列中的 \(n\) 个唯一前缀）来简单地防止访问未来标记。相反，我们将使用**因果注意力掩码**，它允许标记 \(i\) 关注序列中所有位置 \(j\leq i\)。您可以使用 `torch.triu` 或广播索引比较来构造此掩码，并且您应该利用您在 §3.5.4 中的缩放点积注意力实现已经支持注意力掩码这一事实。

**应用 RoPE**。RoPE 应应用于查询和关键向量，但不应用于值向量。此外，头维度应作为批处理维度处理，因为在多头注意力中，注意力对每个头独立应用。这意味着应**完全相同**的 RoPE 旋转应用于每个头的查询和关键向量。

#### **问题 (multhead_self_attention)：实现因果多头自注意力 (5 分)**
**交付物：** 将因果多头自注意力实现为 `torch.nn.Module`。您的实现应至少接受以下参数：
*   `d_model: int`：Transformer 块输入的维度。
*   `num_heads: int`：多头自注意力中使用的头数。

遵循 Vaswani 等人 [2017]，设置 \(d_k = d_v = d_{\mathrm{model}} / h\)。要针对我们提供的测试测试您的实现，请在 `[adapters.run_multhead_self_attention]` 处实现测试适配器。然后，运行 `uv run pytest -k test_multhead_self_attention` 来测试您的实现。

### **3.6 完整的 Transformer LM**
让我们首先组装 Transformer 块（回头参考图 2 会有所帮助）。一个 Transformer 块包含两个'子层'，一个用于多头自注意力，另一个用于前馈网络。在每个子层中，我们首先执行 RMSNorm，然后是主要操作（MHA/FF），最后添加残差连接。

具体来说，Transformer 块的前半部分（第一个'子层'）应实现以下更新集，以从输入 \(x\) 产生输出 \(y\)，
\[y = x + \mathrm{MultiHeadSelfAttention}(\mathrm{RMSNorm}(x)). \quad (15)\]

#### **问题 (transformer_block)：实现 Transformer 块 (3 分)**
实现 §3.5 中描述并在图 2 中说明的预归一化 Transformer 块。您的 Transformer 块应至少接受以下参数。
*   `d_model: int`：Transformer 块输入的维度。
*   `num_heads: int`：多头自注意力中使用的头数。
*   `d_ff: int`：逐位置前馈内部层的维度。

要测试您的实现，请在 `[adapters.run_transformer_block]` 处实现适配器。然后运行 `uv run pytest -k test_transformer_block` 来测试您的实现。
**交付物：** 通过提供测试的 Transformer 块代码。

现在我们将这些块组合起来，遵循图 1 中的高级图表。遵循我们在第 3.1.1 节中对嵌入的描述，将其输入到 `num_layers` 个 Transformer 块中，然后将其传递到三个输出层以获得在词汇表上的分布。

#### **问题 (transformer lm)：实现 Transformer LM (3 分)**
是时候把它们整合在一起了！实现 §3.1 中描述并在图 1 中说明的 Transformer 语言模型。至少，您的实现应接受 Transformer 块的所有上述构造参数，以及这些附加参数：
*   `vocab_size: int`：词汇表大小，用于确定标记嵌入矩阵的维度。
*   `context_length: int`：最大上下文长度，用于确定位置嵌入矩阵的维度。
*   `num_layers: int`：要使用的 Transformer 块的数量。

要针对我们提供的测试测试您的实现，您需要首先在 `[adapters.run_transformer_lm]` 处实现测试适配器。然后，运行 `uv run pytest -k test_transformer_lm` 来测试您的实现。
**交付物：** 通过上述测试的 Transformer LM 模块。

**资源核算**。能够理解 Transformer 的各个部分如何消耗计算和内存是有用的。我们将逐步完成一些基本的"FLOPs 核算"。Transformer 中的绝大多数 FLOPs 是矩阵乘法，因此我们的核心方法很简单：
1.  写下 Transformer 前向传递中的所有矩阵乘法。
2.  将每个矩阵乘法转换为所需的 FLOPs。

对于第二步，以下事实将有用：
**规则**：给定 \(A\in \mathbb{R}^{m\times n}\) 和 \(B\in \mathbb{R}^{n\times p}\)，矩阵-矩阵乘积 \(AB\) 需要 \(2mnp\) 次 FLOPs。

要看到这一点，请注意 \((AB)[i,j] = A[i,:]\cdot B[:,j]\)，并且这个点积需要 \(n\) 次加法和 \(n\) 次乘法（2n 次 FLOPs）。然后，由于矩阵-矩阵乘积 \(AB\) 有 \(m\times p\) 个条目，总 FLOPs 数为 \((2n)(mp) = 2mnp\)。

现在，在您做下一个问题之前，遍历您的 Transformer 块和 Transformer LM 的每个组件，并列出所有矩阵乘法及其相关的 FLOPs 成本可能会有所帮助。

#### **问题 (transformer accounting)：Transformer LM 资源核算 (5 分)**
**(a)** 考虑 GPT-2 XL，其配置如下：
    vocab_size : 50,257
    context_length : 1,024
    num_layers : 48
    d_model : 1,600
    num_heads : 25
    d_ff : 6,400

假设我们使用此配置构建模型。我们的模型将有多少可训练参数？假设每个参数使用单精度浮点数表示，仅加载此模型需要多少内存？
> **交付物：** 一到两句话的回答。

**(b)** 识别完成我们 GPT-2 XL 形状模型的前向传递所需的矩阵乘法。这些矩阵乘法总共需要多少 FLOPs？假设我们的输入序列有 `context_length` 个标记。
> **交付物：** 矩阵乘法列表（带描述）以及所需的总 FLOPs 数。

**(c)** 根据您上面的分析，模型的哪些部分需要最多的 FLOPs？
> **交付物：** 一到两句话的回答。

**(d)** 对 GPT-2 small（12 层，768 d_model，12 头）、GPT-2 medium（24 层，1024 d_model，16 头）和 GPT-2 large（36 层，1280 d_model，20 头）重复您的分析。随着模型规模增加，Transformer LM 的哪些部分占总 FLOPs 的比例变得更大或更小？
> **交付物：** 对于每个模型，提供模型组件的分解及其相关的 FLOPs（作为一次前向传递所需总 FLOPs 的比例）。此外，提供一到两句话的描述，说明改变模型规模如何改变每个组件 FLOPs 的比例。

**(e)** 以 GPT-2 XL 为例，将上下文长度增加到 16,384。一次前向传递的总 FLOPs 如何变化？模型组件 FLOPs 的相对贡献如何变化？
> **交付物：** 一到两句话的回答。

---

## **4 训练一个 Transformer LM**

我们现在已经有了预处理数据（通过分词器）和模型（Transformer）的步骤。剩下的是构建所有支持训练的代码。这包括以下内容：
*   **损失**：我们需要定义损失函数（交叉熵）。
*   **优化器**：我们需要定义优化器来最小化此损失（AdamW）。
*   **训练循环**：我们需要所有加载数据、保存检查点和管理训练的支持基础设施。

### **4.1 交叉熵损失**
回想一下，Transformer 语言模型为每个长度为 \(m + 1\) 的序列 \(x\) 和 \(i = 1, \ldots , m\) 定义了一个分布 \(p_{\theta}(x_{i + 1} \mid x_{1:i})\)。给定一个由长度为 \(m\) 的序列组成的训练集 \(D\)，我们定义标准的交叉熵（负对数似然）损失函数：
\[\ell (\theta ;D) = \frac{1}{|D|m}\sum_{x\in D}\sum_{i = 1}^{m} - \log p_{\theta}(x_{i + 1}\mid x_{1:i}). \quad (16)\]
（注意，Transformer 中的一次前向传递为所有 \(i = 1, \ldots , m\) 产生 \(p_{\theta}(x_{i + 1} \mid x_{1:i})\)。）具体来说，Transformer 为每个位置 \(i\) 计算 logits \(o_i \in \mathbb{R}^{\text{vocab\_size}}\)，这导致：
\[p(x_{i + 1}\mid x_{1:i}) = \mathrm{softmax}(o_i)[x_{i + 1}] = \frac{\exp(o_i[x_{i + 1}])}{\sum_{a = 1}^{\text{vocab\_size}}\exp(o_i[a])}. \quad (17)\]
交叉熵损失通常相对于 logits 向量 \(o_i \in \mathbb{R}^{\text{vocab\_size}}\) 和目标 \(x_{i + 1}\) 定义。

实现交叉熵损失需要像在 softmax 情况下一样注意数值问题。

#### **问题 (cross entropy)：实现交叉熵**
**交付物：** 编写一个函数来计算交叉熵损失，该函数接收预测的 logits \((o_i)\) 和目标 \((x_{i + 1})\)，并计算交叉熵 \(\ell_i = - \log \mathrm{softmax}(o_i)[x_{i + 1}]\)。您的函数应处理以下问题：
*   减去最大元素以提高数值稳定性。
*   尽可能抵消 log 和 exp。
*   处理任何额外的批维度，并返回批上的平均值。与第 3.3 节一样，我们假设批状维度总是出现在词汇大小维度之前。

实现 `[adapters.run_cross_entropy]`，然后运行 `uv run pytest -k test_cross_entropy` 来测试您的实现。

**困惑度** 交叉熵足以进行训练，但当我们评估模型时，我们还想报告困惑度。对于一个长度为 \(m\) 且遭受交叉熵损失 \(\ell_1, \ldots , \ell_m\) 的序列：
\[\mathrm{perplexity} = \exp \left(\frac{1}{m}\sum_{i = 1}^{m}\ell_{i}\right). \quad (18)\]

### **4.2 SGD 优化器**
现在我们有了损失函数，我们将开始探索优化器。最简单的基于梯度的优化器是**随机梯度下降（SGD）**。我们从随机初始化的参数 \(\theta_0\) 开始。然后对于每个步骤 \(t = 0,\ldots ,T - 1\)，我们执行以下更新：
\[\theta_{t + 1}\leftarrow \theta_t - \alpha_t\nabla L(\theta_t;B_t), \quad (19)\]
其中 \(B_{t}\) 是从数据集 \(D\) 中随机采样的一批数据，学习率 \(\alpha_{t}\) 和批量大小 \(|B_{t}|\) 是超参数。

#### **4.2.1 在 PyTorch 中实现 SGD**
为了实现我们的优化器，我们将子类化 PyTorch 的 `torch.optim.Optimizer` 类。一个 `Optimizer` 子类必须实现两个方法：

*   `def __init__(self, params, ...)` 应该初始化您的优化器。这里，`params` 将是要优化的参数集合（或参数组，如果用户希望为模型的不同部分使用不同的超参数，例如学习率）。确保将 `params` 传递给基类的 `__init__` 方法，该方法将存储这些参数以供 `step` 使用。您可以根据优化器接受额外参数（例如，学习率是一个常见参数），并将它们作为字典传递给基类构造函数，其中键是您为这些参数选择的名称（字符串）。

*   `def step(self)` 应该对参数进行一次更新。在训练循环期间，这将在反向传播之后调用，因此您可以访问最后一批的梯度。此方法应遍历每个参数张量 `p` 并就地修改它们，即设置 `p.data`（保存与该参数关联的张量），基于梯度 `p.grad`（如果存在）（表示损失相对于该参数的梯度的张量）。

PyTorch 优化器 API 有一些微妙之处，所以最好通过一个例子来解释。为了使我们的示例更丰富，我们将实现一个稍加变化的 SGD，其中学习率随训练衰减，从初始学习率 \(\alpha\) 开始，并随着时间的推移采取越来越小的步长：
\[\theta_{t + 1} = \theta_t - \frac{\alpha}{\sqrt{t + 1}}\nabla L(\theta_t;B_t) \quad (20)\]

让我们看看这个版本的 SGD 如何作为 PyTorch `Optimizer` 实现：
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
            lr = group["lr"] # 获取学习率。
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # 获取与 p 关联的状态。
                t = state.get("t", 0) # 从状态获取迭代次数，或初始值。
                grad = p.grad.data # 获取损失相对于 p 的梯度。
                p.data = -lr / math.sqrt(t + 1) * grad # 就地更新权重张量。
                state["t"] = t + 1 # 增加迭代次数。
        return loss
```

在 `__init__` 中，我们将参数以及默认超参数传递给基类构造函数（参数可能分组，每组具有不同的超参数）。如果参数只是 `torch.nn.Parameter` 对象的单个集合，基构造函数将创建单个组并为其分配默认超参数。然后，在 `step` 中，我们遍历每个参数组，然后遍历该组中的每个参数，并应用方程 20。在这里，我们将迭代次数保持为与每个参数关联的状态：我们首先读取此值，在梯度更新中使用它，然后更新它。API 规定用户可能会传递一个可调用的 `closure` 来在优化器步骤之前重新计算损失。我们不需要这个用于我们将使用的优化器，但我们添加它以符合 API。

要看到这个工作，我们可以使用以下训练循环的最小示例：
```python
weights = torch.nn.Parameter(5 * torch.randn(10, 10))
opt = SGD([weights], lr=1)

for t in range(100):
    opt.zero_grad() # 重置所有可学习参数的梯度。
    loss = (weights**2).mean() # 计算标量损失值。
    print(loss.cpu().item())
    loss.backward() # 运行反向传播，计算梯度。
    opt.step() # 运行优化器步骤。
```

这是训练循环的典型结构：在每次迭代中，我们将计算损失并运行优化器的一个步骤。训练语言模型时，我们的可学习参数将来自模型（在 PyTorch 中，`m.parameters()` 给了我们这个集合）。损失将在一批采样数据上计算，但训练循环的基本结构是相同的。

#### **问题 (learning_rate_tuning)：调整学习率 (1 分)**
正如我们将看到的，影响训练的最重要的超参数之一是学习率。让我们在玩具示例中实际看看。使用另外三个学习率值运行上面的 SGD 示例：1e1、1e2 和 1e3，仅进行 10 次训练迭代。对于每个学习率，损失会发生什么变化？它衰减得更快、更慢，还是发散（即，在训练过程中增加）？
> **交付物：** 一个一到两句话的回答，描述您观察到的行为。

### **4.3 AdamW**
现代语言模型通常使用更复杂的优化器进行训练，而不是 SGD。最近使用的大多数优化器都是 Adam 优化器 [Kingma and Ba, 2015] 的衍生品。我们将使用 AdamW [Loshchilov and Hutter, 2019]，这在最近的工作中被广泛使用。AdamW 提出对 Adam 进行修改，通过添加权重衰减来改进正则化（在每次迭代中，我们将参数向 0 拉回），详见 Loshchilov 和 Hutter [2019] 的第 2 节。

AdamW 是**有状态的**：对于每个参数，它跟踪其第一和第二矩的运行估计。因此，AdamW 使用额外的内存来换取改进的稳定性和收敛性。除了学习率 \(\alpha\) 之外，AdamW 还有一对控制矩估计更新的超参数 \((\beta_{1},\beta_{2})\)，以及一个权重衰减率 \(\lambda\)。典型的应用将 \((\beta_{1},\beta_{2})\) 设置为 \((0.9,0.999)\)，但像 LLaMA [Touvron 等人, 2023] 和 GPT-3 [Brown 等人, 2020] 这样的大型语言模型通常使用 \((0.9,0.95)\) 进行训练。该算法可以写如下，其中 \(\epsilon\) 是一个小值（例如，\(10^{- 8}\)），用于在 \(v\) 中得到极小的值时提高数值稳定性：

```
# 算法 1 AdamW 优化器
初始化(0)（初始化可学习参数）
m <- 0（第一矩向量的初始值；与 θ 形状相同）
v <- 0（第二矩向量的初始值；与 θ 形状相同）
for t = 1,...,T do
    采样数据批次 B_t
    g <- ∇_θ L(θ ; B_t)（计算当前时间步损失的梯度）
    m <- β_1 m + (1 - β_1) g（更新第一矩估计）
    v <- β_2 v + (1 - β_2) g^2（更新第二矩估计）
    α_t <- α * (sqrt(1 - β_2^t) / (1 - β_1^t))（计算迭代 t 的调整后 α）
    θ <- θ - α_t * m / (sqrt(v) + ε)（更新参数）
    θ <- θ - α * λ * θ（应用权重衰减）
end for
```

注意 \(t\) 从 1 开始。您现在将实现这个优化器。

#### **问题 (adamw)：实现 AdamW (2 分)**
**交付物：** 将 AdamW 优化器实现为 `torch.optim.Optimizer` 的子类。您的类应在 `__init__` 中接受学习率 \(\alpha\)，以及 \(\beta\)、\(\epsilon\) 和 \(\lambda\) 超参数。为了帮助您保持状态，基 `Optimizer` 类为您提供了一个字典 `self.state`，它将 `nn.Parameter` 对象映射到一个字典，该字典存储该参数所需的任何信息（对于 AdamW，这将是矩估计）。实现 `[adapters.get_adamw_cls]` 并确保通过 `uv run pytest -k test_adamw`。

#### **问题 (adamwAccounting)：使用 AdamW 训练的资源核算 (2 分)**
让我们计算运行 AdamW 需要多少内存和计算。假设我们对每个张量使用 `float32`。

**(a)** 运行 AdamW 需要多少峰值内存？根据参数、激活、梯度和优化器状态的内存使用情况分解您的答案。用 `batch_size` 和模型超参数（`vocab_size`、`context_length`、`num_layers`、`d_model`、`num_heads`）表示您的答案。假设 `d_ff = 4 x d_model`。

为简化起见，在计算激活的内存使用时，仅考虑以下组件：
*   Transformer 块
    *   RMSNorm(s)
    *   多头自注意力子层：\(QKV\) 投影、\(Q^{\top}K\) 矩阵乘法、softmax、值的加权和、输出投影。
    *   逐位置前馈：\(W_{1}\) 矩阵乘法、SiLU、\(W_{2}\) 矩阵乘法
*   最终 RMSNorm
*   输出嵌入
*   logits 上的交叉熵

> **交付物：** 参数、激活、梯度和优化器状态的代数表达式，以及总和。

**(b)** 针对 GPT-2 XL 形状的模型实例化您的答案，以获得一个仅依赖于 `batch_size` 的表达式。在 80GB 内存内可以使用的最大批大小是多少？
> **交付物：** 一个看起来像 \(a\) `batch_size` + \(b\) 的表达式，其中 \(a, b\) 是数值，以及一个表示最大批大小的数字。

**(c)** 运行 AdamW 一步需要多少 FLOPs？
> **交付物：** 一个代数表达式，以及简要的理由。

**(d)** **模型 FLOPs 利用率（MFU）** 定义为观察到的吞吐量（标记/秒）相对于硬件理论峰值 FLOP 吞吐量的比率 [Chowdhery 等人, 2022]。NVIDIA A100 GPU 的 `float32` 操作理论峰值为 19.5 teraFLOP/s。假设您能够达到 50% 的 MFU，在单个 A100 上以 1024 的批大小训练 GPT-2 XL 400K 步需要多长时间？遵循 Kaplan 等人 [2020] 和 Hoffmann 等人 [2022]，假设反向传播的 FLOPs 是前向传播的两倍。
> **交付物：** 训练所需的天数，以及简要的理由。

### **4.4 学习率调度**
导致损失最快下降的学习率值通常在训练期间变化。在训练 Transformers 时，通常使用**学习率调度**，我们从较大的学习率开始，在开始时进行更快的更新，并随着模型的训练缓慢将其衰减到较小的值。在本作业中，我们将实现用于训练 LLaMA [Touvron 等人, 2023] 的余弦退火调度。

调度器只是一个函数，它接收当前步骤 \(t\) 和其他相关参数（例如初始和最终学习率），并返回步骤 \(t\) 用于梯度更新的学习率。最简单的调度是**常数函数**，它将为任何 \(t\) 返回相同的学习率。

**余弦退火学习率调度** 接收 (i) 当前迭代 \(t\)，(ii) 最大学习率 \(\alpha_{\mathrm{max}}\)，(iii) 最小（最终）学习率 \(\alpha_{\mathrm{min}}\)，(iv) 预热迭代次数 \(T_{w}\)，和 (v) 余弦退火迭代次数 \(T_{c}\)。迭代 \(t\) 的学习率定义为：
*   **(预热)** 如果 \(t < T_{w}\)，则 \(\alpha_{t} = \frac{t}{T_{w}}\alpha_{\mathrm{max}}\)。
*   **(余弦退火)** 如果 \(T_{w} \leq t \leq T_{c}\)，则 \(\alpha_{t} = \alpha_{\mathrm{min}} + \frac{1}{2} \left(1 + \cos \left(\frac{t - T_{w}}{T_{c} - T_{w}} \pi\right)\right) (\alpha_{\mathrm{max}} - \alpha_{\mathrm{min}})\)。
*   **(退火后)** 如果 \(t > T_{c}\)，则 \(\alpha_{t} = \alpha_{\mathrm{min}}\)。

#### **问题 (learning_rate_schedule)：实现带预热的余弦学习率调度**
编写一个函数，接收 \(t\)、\(\alpha_{\mathrm{max}}\)、\(\alpha_{\mathrm{min}}\)、\(T_{w}\) 和 \(T_{c}\)，并根据上面定义的调度器返回学习率 \(\alpha_{t}\)。然后实现 `[adapters.get_lr_cosine_schedule]` 并确保通过 `uv run pytest -k test_get_lr_cosine_schedule`。

### **4.5 梯度裁剪**
在训练期间，我们有时会遇到产生大梯度的训练样本，这可能使训练不稳定。为了缓解这种情况，实践中经常采用的一种技术是**梯度裁剪**。其思想是在每次反向传播后、采取优化器步骤之前，对梯度的范数实施限制。

给定（所有参数的）梯度 \(g\)，我们计算其 \(\ell_2\)-范数 \(\| g\| _2\)。如果此范数小于最大值 \(M\)，则我们保持 \(g\) 不变；否则，我们将 \(g\) 按因子 \(\frac{M}{\|g\|_2 + \epsilon}\) 缩小（其中添加了一个小的 \(\epsilon\)，如 \(10^{- 6}\)，用于数值稳定性）。注意，结果范数将刚好低于 \(M\)。

#### **问题 (gradient_clipping)：实现梯度裁剪 (1 分)**
编写一个实现梯度裁剪的函数。您的函数应接收一个参数列表和一个最大 \(\ell_2\)-范数。它应该就地修改每个参数的梯度。使用 \(\epsilon = 10^{- 6}\)（PyTorch 默认值）。然后，实现适配器 `[adapters.run_gradient_clipping]` 并确保通过 `uv run pytest -k test_gradient_clipping`。

---

## **5 训练循环**

我们现在终于将迄今为止构建的主要组件整合在一起：分词后的数据、模型和优化器。

### **5.1 数据加载器**
分词后的数据（例如，您在 `tokenizer_experiments` 中准备的数据）是一个单一的标记序列 \(x = (x_{1},\ldots ,x_{n})\)。即使源数据可能由单独的文档组成（例如，不同的网页或源代码文件），常见的做法是将所有这些连接成一个单一的标记序列，在它们之间添加分隔符（例如 `<|endoftext|>` 标记）。

数据加载器将其转换为批流，其中每批由 \(B\) 个长度为 \(m\) 的序列组成，与相应的下一个标记配对，长度也为 \(m\)。例如，对于 \(B = 1, m = 3\)，\(([x_{2},x_{3},x_{4}],[x_{3},x_{4},x_{5}])\) 将是一个潜在的批次。

以这种方式加载数据简化了训练，原因有很多。首先，任何 \(1\leq i< n - m\) 都给出一个有效的训练序列，因此采样序列是微不足道的。由于所有训练序列具有相同的长度，不需要填充输入序列，这提高了硬件利用率（也通过增加批大小 \(B\)）。最后，我们也不需要完全加载整个数据集来采样训练数据，使得处理可能无法装入内存的大型数据集变得容易。

#### **问题 (data_loading)：实现数据加载 (2 分)**
**交付物：** 编写一个函数，接收一个 numpy 数组 \(x\)（包含标记 ID 的整数数组）、`batch_size`、`context_length` 和一个 PyTorch 设备字符串（例如，'cpu' 或 'cuda:0'），并返回一对张量：采样的输入序列和相应的下一个标记目标。两个张量都应具有形状 `(batch_size, context_length)` 包含标记 ID，并且都应放置在请求的设备上。要针对我们提供的测试测试您的实现，您需要首先在 `[adapters.run_get_batch]` 处实现测试适配器。然后，运行 `uv run pytest -k test_get_batch` 来测试您的实现。

#### **低资源/降尺度提示：在 CPU 或 Apple Silicon 上加载数据**
如果您计划在 CPU 或 Apple Silicon 上训练 LM，您需要将数据移动到正确的设备（同样，稍后应使用相同的设备用于您的模型）。
*   如果您在 CPU 上，可以使用 'cpu' 设备字符串。
*   在 Apple Silicon（M* 芯片）上，可以使用 'mps' 设备字符串。

有关 MPS 的更多信息，请查看这些资源：
*   https://developer.apple.com/metal/pytorch/
*   https://pytorch.org/docs/main/notes/mps.html

**如果数据集太大无法装入内存怎么办？** 我们可以使用名为 `mmap` 的 Unix 系统调用，它将磁盘上的文件映射到虚拟内存，并在访问该内存位置时惰性加载文件内容。因此，您可以"假装"整个数据集都在内存中。Numpy 通过 `np.memmap` 实现此功能（或 `np.load` 的标志 `mmap_mode='r'`，如果您最初使用 `np.save` 保存数组），它将返回一个类似 numpy 数组的对象，在您访问条目时按需加载它们。在训练期间从数据集（即 numpy 数组）采样时，请务必以内存映射模式（通过 `np.memmap` 或 `np.load` 的标志 `mmap_mode='r'`，取决于您保存数组的方式）加载数据集。确保您还指定了与您要加载的数组匹配的 `dtype`。显式验证内存映射数据看起来正确（例如，不包含超出预期词汇表大小的值）可能会有所帮助。

### **5.2 检查点**
除了加载数据，我们还需要在训练时保存模型。运行作业时，我们通常希望能够恢复因某种原因中途停止的训练运行（例如，由于作业超时、机器故障等）。即使一切顺利，我们可能也希望以后能够访问中间模型（例如，事后研究训练动态，从训练不同阶段的模型采样等）。

一个检查点应具有我们恢复训练所需的所有状态。当然，我们希望至少能够恢复模型权重。如果使用有状态的优化器（如 AdamW），我们还需要保存优化器的状态（例如，在 AdamW 的情况下，矩估计）。最后，为了恢复学习率调度，我们需要知道我们停止时的迭代次数。PyTorch 使得保存所有这些变得容易：每个 `nn.Module` 都有一个 `state_dict()` 方法，返回一个包含所有权重学习的字典；稍后我们可以使用姊妹方法 `load_state_dict()` 恢复这些权重。任何 `nn.optim.Optimizer` 也是如此。最后，`torch.save(obj, dest)` 可以将一个对象（例如，一个在某些值中包含张量但也包含常规 Python 对象如整数的字典）转储到文件（路径）或类似文件的对象，然后可以使用 `torch.load(src)` 加载回内存。

#### **问题 (checkpointing)：实现模型检查点 (1 分)**
实现以下两个函数以加载和保存检查点：
*   `def save_checkpoint(model, optimizer, iteration, out)` 应将前三个参数的所有状态转储到文件类对象 `out` 中。您可以使用模型和优化器的 `state_dict` 方法获取它们的相关状态，并使用 `torch.save(obj, out)` 将 `obj` 转储到 `out`（PyTorch 支持路径或文件类对象）。典型的选择是让 `obj` 是一个字典，但只要您以后可以加载您的检查点，您可以使用任何格式。

此函数期望以下参数：
*   `model: torch.nn.Module`
*   `optimizer: torch.optim.Optimizer`
*   `iteration: int`
*   `out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`

*   `def load_checkpoint(src, model, optimizer)` 应从 `src`（路径或文件类对象）加载检查点，然后从该检查点恢复模型和优化器状态。您的函数应返回保存到检查点的迭代次数。您可以使用 `torch.load(src)` 恢复您在 `save_checkpoint` 实现中保存的内容，并使用模型和优化器中的 `load_state_dict` 方法将它们恢复到之前的状态。

此函数期望以下参数：
*   `src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`
*   `model: torch.nn.Module`
*   `optimizer: torch.optim.Optimizer`

实现 `[adapters.run_save_checkpoint]` 和 `[adapters.run_load_checkpoint]` 适配器，并确保它们通过 `uv run pytest -k test_checkpointing`。

### **5.3 训练循环**
现在，终于到了将您实现的所有组件整合到您的主训练脚本中的时候了。使使用不同超参数轻松启动训练运行（例如，通过将它们作为命令行参数）将是值得的，因为您稍后将多次执行这些操作以研究不同选择如何影响训练。

#### **问题 (training together)：整合一切 (4 分)**
**交付物：** 编写一个脚本，运行训练循环以在用户提供的输入上训练您的模型。特别是，我们建议您的训练脚本允许（至少）以下功能：
*   能够配置和控制各种模型和优化器超参数。
*   使用 `np.memmap` 内存高效地加载大型训练和验证数据集。
*   将检查点序列化到用户提供的路径。
*   定期记录训练和验证性能（例如，到控制台和/或像 Weights and Biases 这样的外部服务）。

---

## **6 生成文本**

既然我们可以训练模型，我们需要的最后一部分是从模型生成文本的能力。回想一下，语言模型接收一个（可能批处理的）长度为 `sequence_length` 的整数序列，并产生一个大小为 `(sequence_length × vocab_size)` 的矩阵，其中序列的每个元素是预测该位置之后下一个词的概率分布。我们现在将编写一些函数，将其转换为新序列的采样方案。

**Softmax** 按照标准约定，语言模型输出是最终线性层（"logits"）的输出，因此我们必须通过 softmax 操作将其转换为归一化概率，我们之前在方程 10 中看到过。

**解码** 要从模型生成文本（解码），我们将向模型提供一系列前缀标记（"提示"），并要求它产生在词汇表上的概率分布，以预测序列中的下一个词。然后，我们将从此词汇表分布中采样以确定下一个输出标记。

具体来说，解码过程的一步应接收序列 \(x_{1\dots t}\) 并通过以下方程返回标记 \(x_{t + 1}\)，
\[P(x_{t + 1} = i\mid x_{1\dots t}) = \frac{\exp(v_i)}{\sum_j\exp(v_j)}\]
\[\qquad v = \mathrm{TransformerLM}(x_{1\dots t})_t\in \mathbb{R}^{\mathrm{vocab\_size}}\]
其中 `TransformerLM` 是我们的模型，它接收长度为 `sequence_length` 的序列作为输入，并产生大小为 `(sequence_length × vocab_size)` 的矩阵，我们取此矩阵的最后一个元素，因为我们在寻找第 \(t\) 个位置的下一个词预测。

这通过重复从这些一步条件概率中采样（将我们之前生成的输出标记附加到下一个解码时间步的输入中），直到我们生成序列结束标记 `<|endoftext|>`（或用户指定的要生成的最大标记数），为我们提供了一个基本解码器。

**解码器技巧** 我们将使用小模型进行实验，而小模型有时会生成质量很低的文本。两个简单的解码器技巧可以帮助解决这些问题。首先，在**温度缩放**中，我们使用温度参数 \(\tau\) 修改我们的 softmax，其中新的 softmax 是：
\[\mathrm{softmax}(v,\tau)_i = \frac{\exp(v_i / \tau)}{\sum_{j = 1}^{|\mathrm{vocab\_size}|}\exp(v_j / \tau)}. \quad (24)\]
注意，设置 \(\tau \to 0\) 使得 \(v\) 的最大元素占主导地位，softmax 的输出变为集中于该最大元素的一个独热向量。

其次，另一个技巧是**核心或 top-\(p\) 采样**，我们通过截断低概率词来修改采样分布。设 \(q\) 是从大小（vocab_size）的（温度缩放的）softmax 得到的概率分布。具有超参数 \(p\) 的核心采样根据以下方程产生下一个标记：
\[P_{\text{nucleus}}(x_{t+1} = i \mid x_{1\dots t}) = \begin{cases} q_i / \sum_{j \in V(p)} q_j, & \text{if } i \in V(p) \\ 0, & \text{otherwise} \end{cases}\]
其中 \(V(p)\) 是最小的索引集合，使得 \(\textstyle \sum_{j\in V(p)}q_{j}\geq p\)。您可以首先按大小对概率分布 \(q\) 进行排序，然后选择最大的词汇表元素直到达到目标水平 \(\alpha\)，从而轻松计算此量。

#### **问题 (decoding)：解码 (3 分)**
**交付物：** 实现一个从您的语言模型解码的函数。我们建议您支持以下功能：
*   为用户提供的提示生成补全（即，接收一些 \(x_{1\dots t}\) 并采样补全，直到遇到 `<|endoftext|>` 标记）。
*   允许用户控制生成的最大标记数。
*   给定所需的温度值，在采样之前对预测的下一个词分布应用 softmax 温度缩放。
*   **Top-p 采样** (Holtzman 等人, 2020; 也称为核心采样)，给定用户指定的阈值。

---

## **7 实验**

现在是时候将所有内容整合起来，并在预训练数据集上训练（小型）语言模型了。

### **7.1 如何运行实验和交付物**
理解 Transformer 架构组件背后原理的最佳方法是实际修改并自己运行它。动手经验是无法替代的。

为此，重要的是能够快速、一致地进行实验，并保留所做工作的记录。为了快速实验，我们将在小规模模型（1700 万参数）和简单数据集（TinyStories）上运行许多实验。为了保持一致，您将系统地消融组件和改变超参数，为了保留记录，我们将要求您提交实验日志和与每个实验相关的学习曲线。

为了能够提交损失曲线，请确保定期评估验证损失，并记录步数和挂钟时间。您可能会发现像 Weights and Biases 这样的日志记录基础设施很有帮助。

#### **问题 (experiment_log)：实验日志记录 (3 分)**
为您的训练和评估代码创建实验跟踪基础设施，允许您跟踪实验以及相对于梯度步数和挂钟时间的损失曲线。
> **交付物：** 用于实验的日志记录基础设施代码，以及本部分下面作业问题的实验日志（记录您尝试过的所有内容）。

### **7.2 TinyStories**
我们将从一个非常简单的数据集（TinyStories；Eldan and Li, 2023）开始，模型将在此数据集上快速训练，并且我们可以看到一些有趣的行为。获取此数据集的说明在第 1 节。以下是该数据集的一个示例。

#### **示例 (tinystories_example)：TinyStories 的一个例子**
> 从前有一个名叫本的小男孩。本喜欢探索他周围的世界。他看到了许多令人惊奇的东西，比如商店里陈列的漂亮花瓶。有一天，本在商店里走着，偶然发现了一个非常特别的花瓶。当本看到它时，他惊呆了！他说："哇，那真是一个非常棒的花瓶！我能买下它吗？"店主笑着说："当然可以。你可以把它带回家，向你所有的朋友展示它有多棒！"于是本把花瓶带回家，他为此感到非常自豪！他叫来他的朋友们，向他们展示了这个令人惊奇的花瓶。他所有的朋友都认为这个花瓶很漂亮，不敢相信本有多幸运。这就是本如何在商店里找到一个令人惊奇的花瓶的故事！

**超参数调整** 我们将告诉您一些非常基本的超参数作为开始，并要求您找到其他一些运行良好的设置。
*   `vocab_size`：10000。典型的词汇表大小在数万到数十万之间。您应该改变这个值，看看词汇表和模型行为如何变化。
*   `context_length`：256。像 TinyStories 这样的简单数据集可能不需要长序列长度，但对于后来的 OpenWebText 数据，您可能希望改变这个值。尝试改变这个值，看看对每次迭代运行时间和最终困惑度的影响。
*   `d_model`：512。这比许多小型 Transformer 论文中使用的 768 维度略小，但这会使速度更快。
*   `d_ff`：1344。这大约是 3×`d_model`，同时是 64 的倍数，这对 GPU 性能有好处。
*   RoPE theta 参数 \(\Theta\)：10000。
*   层数和头数：4 层，16 头。总共这将产生大约 1700 万个非嵌入参数，这是一个相当小的 Transformer。
*   处理的总标记数：327,680,000（您的批大小 × 总步数 × 上下文长度应大致等于此值）。

您应该进行一些试验和错误，为以下其他超参数找到良好的默认值：学习率、学习率预热、其他 AdamW 超参数 \((\beta_{1},\beta_{2},\epsilon)\) 和权重衰减。您可以在 Kingma 和 Ba [2015] 中找到此类超参数的一些典型选择。

**整合一切** 现在，您可以通过获取训练好的 BPE 分词器、对训练数据集进行分词，并将其放入您编写的训练循环中，将所有内容整合在一起。
**重要提示**：如果您的实现正确且高效，上述超参数应在 1 个 H100 GPU 上产生大约 30-40 分钟的运行时。如果您的运行时间远长于此，请检查并确保您的数据加载、检查点或验证损失代码没有成为瓶颈，并且您的实现已正确批处理。

**调试模型架构的技巧** 我们强烈建议熟悉您 IDE 的内置调试器（例如，VSCode/PyCharm），与使用 print 语句调试相比，这将节省您的时间。如果您使用文本编辑器，可以使用像 pdb 这样的工具。调试模型架构时其他一些好的做法是：
*   开发任何神经网络架构时，常见的第一步是**过拟合到单个小批次**。如果您的实现是正确的，您应该能够快速将训练损失驱动到接近零。
*   在各个模型组件中设置调试断点，并检查中间张量的形状，确保它们符合您的预期。
*   监控激活、模型权重和梯度的范数，确保它们没有爆炸或消失。

#### **问题 (learning_rate)：调整学习率 (3 分) (4 H100 小时)**
学习率是最重要的需要调整的超参数之一。使用您训练的基础模型，回答以下问题：

**(a)** 对学习率执行超参数扫描，并报告最终损失（或注意优化器是否发散）。
> **交付物：** 多个学习率相关的学习曲线。解释您的超参数搜索策略。
> **交付物：** 在 TinyStories 上验证损失（每标记）至多为 1.45 的模型。

#### **低资源/降尺度提示：在 CPU 或 Apple Silicon 上进行少量步骤的训练**
如果您在 cpu 或 mps 上运行，您应该将处理的总标记数减少到 40,000,000，这足以产生相当流畅的文本。您也可以将目标验证损失从 1.45 增加到 2.00。

使用调优的学习率运行我们的解决方案代码，在 M3 Max 芯片和 36 GB RAM 上，我们使用批大小 × 总步数 × 上下文长度 = \(32\times 5000\times 256 = 40,960,000\) 个标记，这在 cpu 上需要 1 小时 22 分钟，在 mps 上需要 36 分钟。在 5000 步时，我们实现了 1.80 的验证损失。

一些额外提示：
*   当使用 \(X\) 个训练步时，我们建议调整余弦学习率衰减调度，使其衰减在恰好第 \(X\) 步终止（即，达到最小学习率）。
*   当使用 mps 时，不要使用 TF32 内核，即不要设置 `torch.set_float32_matmul_precision('high')`，您可能会在 cuda 设备上这样做。我们尝试启用 mps 的 TF32 内核（torch 版本 2.6.0），发现后端会使用静默损坏的内核，导致训练不稳定。
*   您可以通过使用 `torch.compile` JIT 编译模型来加速训练。具体来说：
    *   在 cpu 上，使用 `model = torch.compile(model)` 编译您的模型。
    *   在 mps 上，您可以使用 `model = torch.compile(model, backend="aot_eager")` 在一定程度上优化反向传播。截至 torch 版本 2.6.0，Inductor 编译在 mps 上不受支持。

**(b)** 民间智慧认为最佳学习率是"在稳定性的边缘"。研究学习率发散的点与您的最佳学习率之间的关系。
> **交付物：** 增加学习率的学习曲线，其中包括至少一个发散运行，以及分析这与收敛速度的关系。

现在让我们改变批大小，看看训练会发生什么。批大小很重要——它们通过做更大的矩阵乘法让我们从 GPU 获得更高的效率，但我们总是希望批大小很大吗？让我们运行一些实验来找出答案。

#### **问题 (batch_size_experiment)：批大小变化 (1 分) (2 H100 小时)**
将批大小从 1 变化到 GPU 内存限制。至少尝试几个中间的批大小，包括典型的大小，如 64 和 128。
> **交付物：** 不同批大小的学习曲线。如果需要，应再次优化学习率。
> **交付物：** 几句话讨论您关于批大小及其对训练影响的发现。

有了您的解码器，我们现在可以生成文本了！我们将从模型生成文本，看看它有多好。作为参考，您应该获得至少和下面示例一样好的输出。

#### **示例 (ts_generate_example)：来自 TinyStories 语言模型的样本输出**
> 从前，有一个漂亮的女孩名叫莉莉。她喜欢吃口香糖，尤其是大的黑色那种。一天，莉莉的妈妈叫她帮忙做晚饭。莉莉非常兴奋！她喜欢帮助妈妈。莉莉的妈妈做了一大锅汤当晚餐。莉莉非常高兴，说："谢谢你，妈妈！我爱你。"她帮助妈妈把汤倒进一个大碗里。晚饭后，莉莉的妈妈做了一些美味的汤。莉莉很喜欢！她说："谢谢你，妈妈！这汤真好吃！"她妈妈笑着说："我很高兴你喜欢，莉莉。"他们做完饭，继续一起做饭。结束。

#### **低资源/降尺度提示：在 CPU 或 Apple Silicon 上生成文本**
如果您使用了处理 4000 万标记的低资源配置，您应该看到仍然像英语但不如上面流利的生成。例如，我们处理 4000 万标记的 TinyStories 语言模型的样本输出如下：
> 从前，有一个名叫苏的小女孩。苏有一颗她非常喜欢的牙齿。这是他最好的头。一天，苏去散步，遇到了一只瓢虫！他们成了好朋友，在小径上一起玩耍。
> "嘿，波莉！我们出去吧！"蒂姆说。苏看着天空，发现很难找到一种闪闪发光的方式跳舞。她笑着同意帮助这个谈话！"
> 当苏看着天空移动时，它是什么。她

以下是精确的问题陈述和我们要求的内容：

#### **问题 (generate)：生成文本 (1 分)**
使用您的解码器和训练好的检查点，报告您的模型生成的文本。您可能需要操作解码器参数（温度、top-p 等）以获得流畅的输出。
> **交付物：** 至少 256 个标记的文本转储（或直到第一个 `<|endoftext|>` 标记），并简要评论此输出的流畅性以及影响此输出好坏的两个因素。

### **7.3 消融和架构修改**
理解 Transformer 的最佳方法是实际修改它并观察其行为。我们现在将进行一些简单的消融和修改。

**消融 1：层归一化** 通常认为层归一化对 Transformer 训练的稳定性很重要。但也许我们想冒险。让我们从每个 Transformer 块中移除 RMSNorm，看看会发生什么。

#### **问题 (layer_norm_ablation)：移除 RMSNorm 并进行训练 (1 分) (1 H100 小时)**
从 Transformer 中移除所有 RMSNorm 并进行训练。在先前的最佳学习率下会发生什么？您可以通过使用较低的学习率来获得稳定性吗？
> **交付物：** 当您移除 RMSNorm 并进行训练时的学习曲线，以及最佳学习率的学习曲线。
> **交付物：** 关于 RMSNorm 影响的几句话评论。

现在让我们研究另一个乍一看似乎很随意的层归一化选择。预归一化 Transformer 块定义为：
\[z = x + \mathrm{MultiHeadedSelfAttention}(\mathrm{RMSNorm}(x))\]
\[y = z + \mathrm{FFN}(\mathrm{RMSNorm}(z)).\]
这是对原始 Transformer 架构为数不多的"共识"修改之一，原始架构使用后归一化方法：
\[z = \mathrm{RMSNorm}(x + \mathrm{MultiHeadedSelfAttention}(x))\]
\[y = \mathrm{RMSNorm}(z + \mathrm{FFN}(z)).\]

让我们恢复到后归一化方法，看看会发生什么。

#### **问题 (pre_norm_ablation)：实现后归一化并进行训练 (1 分) (1 H100 小时)**
将您的预归一化 Transformer 实现修改为后归一化。使用后归一化模型进行训练，看看会发生什么。
> **交付物：** 后归一化 Transformer 的学习曲线，与预归一化的进行比较。

我们看到层归一化对 Transformer 的行为有重大影响，甚至层归一化的位置也很重要。

**消融 2：位置嵌入** 接下来，我们将研究位置嵌入对模型性能的影响。具体来说，我们将比较我们的基础模型（带 RoPE）与完全不包含位置嵌入（NoPE）。事实证明，仅解码器 Transformer，即我们已实现的带有因果掩码的 Transformer，理论上可以在不显式提供位置嵌入的情况下推断相对或绝对位置信息 [Tsai 等人, 2019, Kazemnejad 等人, 2023]。我们现在将实证测试 NoPE 与 RoPE 相比表现如何。

#### **问题 (no_pos_emb)：实现 NoPE (1 分) (1 H100 小时)**
修改您带有 RoPE 的 Transformer 实现，完全移除位置嵌入信息，看看会发生什么。
> **交付物：** 比较 RoPE 和 NoPE 性能的学习曲线。

**消融 3：SwiGLU vs. SiLU** 接下来，我们将遵循 Shazeer [2020]，通过比较 SwiGLU 前馈网络与使用 SiLU 激活但没有门控线性单元（GLU）的前馈网络的性能，来测试前馈网络中门控的重要性：
\[\mathrm{FFN}_{\mathrm{SiLU}}(x) = W_2\mathrm{SiLU}(W_1x). \quad (25)\]
回想一下，在我们的 SwiGLU 实现中，我们将内部前馈层的维度设置为大约 \(d_{\mathrm{ff}} = \frac{8}{3}\times d_{\mathrm{model}}\)（同时确保 \(d_{\mathrm{ff}}\) mod \(64 = 0\)，以利用 GPU 张量核心）。在您的 \(\mathrm{FFN}_{\mathrm{SiLU}}\) 实现中，您应将 \(d_{\mathrm{ff}}\) 设置为 \(4\times d_{\mathrm{model}}\)，以大致匹配 SwiGLU 前馈网络的参数计数（后者有三个而不是两个权重矩阵）。

#### **问题 (swigl_ablation)：SwiGLU vs. SiLU (1 分) (1 H100 小时)**
**交付物：** 比较 SwiGLU 和 SiLU 前馈网络性能的学习曲线，具有大致匹配的参数计数。

#### **低资源/降尺度提示：GPU 资源有限的在线学生应在 TinyStories 上测试修改**
在作业的剩余部分，我们将转向一个更大规模、更嘈杂的网络数据集（OpenWebText），实验架构修改，并（可选）提交到课程排行榜。

在 OpenWebText 上将 LM 训练到流畅需要很长时间，因此我们建议 GPU 访问有限的在线学生继续在 TinyStories 上测试修改（使用验证损失作为评估性能的指标）。

### **7.4 在 OpenWebText 上运行**
我们现在将转向一个更标准的从网络爬取创建的预训练数据集。OpenWebText [Gokaslan 等人, 2019] 的一个小样本也作为单个文本文件提供：请参阅第 1 节了解如何访问此文件。

以下是来自 OpenWebText 的一个例子。注意文本如何更加真实、复杂和多样。您可能想浏览训练数据集，以了解网络抓取语料库的训练数据是什么样子。

#### **示例 (ovt_example)：OWT 的一个例子**
> 棒球展望网站技术总监哈里·帕夫利季斯在雇佣乔纳森·贾奇时冒了风险。帕夫利季斯知道，正如艾伦·施瓦茨在《数字游戏》中所写，"美国文化中没有哪个角落比棒球运动员的表现被更精确地计算、更热情地量化。"点击几下，您就可以发现诺阿·辛德加德的快速球在到达本垒板的过程中每分钟旋转超过 2,100 次，纳尔逊·克鲁兹在 2016 年合格击球手中拥有比赛中最高的平均出球速度，以及无数其他似乎来自电子游戏或科幻小说的花絮。不断增长的数据海洋赋予了棒球文化中一个日益重要的角色力量：分析爱好者。
> 这种赋权伴随着额外的审查——不仅针对测量，也针对背后的人员和出版物。对于棒球展望网站，帕夫利季斯完全了解伴随定量不完美的反弹。他也知道该网站的捕手指标需要重新设计，并且需要一个博学的头脑——一个能够处理复杂统计建模问题的人——来完成这项工作。
> "他让我们抓狂。" 哈里·帕夫利季斯
> 帕夫利季斯有一种直觉，基于后者的文章以及他们在网站赞助的棒球场活动中的互动，贾奇"明白了"。不久之后，两人边喝酒边交谈。帕夫利季斯的直觉得到了验证。贾奇适合这个职位——更妙的是，他是一个自愿适合的人。"我和很多人谈过，"帕夫利季斯说，"他是唯一有勇气接受它的人。" [...]

**注意：** 您可能必须为此实验重新调整超参数，例如学习率或批大小。

#### **问题 (main_experiment)：在 OWT 上实验 (2 分) (3 H100 小时)**
在 OpenWebText 上使用与 TinyStories 相同的模型架构和总训练迭代次数训练您的语言模型。这个模型表现如何？
> **交付物：** 您在 OpenWebText 上的语言模型的学习曲线。描述与 TinyStories 相比的损失差异——我们应如何解释这些损失？
> **交付物：** 来自 OpenWebText LM 的生成文本，格式与 TinyStories 输出相同。此文本的流畅性如何？为什么即使我们拥有与 TinyStories 相同的模型和计算预算，输出质量更差？

### **7.5 您自己的修改 + 排行榜**
恭喜您达到了这一点。您快要完成了！您现在将尝试改进 Transformer 架构，并看看您的超参数和架构与班上其他学生相比如何。

**排行榜规则** 除了以下规定外，没有其他限制：
*   **运行时间**：您的提交在 H100 上最多可以运行 1.5 小时。您可以通过在 slurm 提交脚本中设置 `--time=01:30:00` 来强制执行此限制。
*   **数据**：您只能使用我们提供的 OpenWebText 训练数据集。

除此之外，您可以自由地做任何您想做的事情。

如果您正在寻找一些实现的思路，可以查看以下资源：
*   最先进的开源 LLM 家族，如 Llama 3 [Grattafiori 等人, 2024] 或 Qwen 2.5 [Yang 等人, 2024]。
*   NanoGPT 速通仓库 (https://github.com/KellerJordan/modded-nanogpt)，社区成员在此发布了许多有趣的修改，用于"速通"小规模语言模型预训练。例如，可以追溯到原始 Transformer 论文的一个常见修改是**绑定输入和输出嵌入的权重**（参见 Vaswani 等人 [2017]（第 3.4 节）和 Chowdhery 等人 [2022]（第 2 节））。如果您尝试权重绑定，您可能需要减小嵌入/LM 头初始化的标准差。

在尝试完整的 1.5 小时运行之前，您需要在 OpenWebText 的一个小子集或 TinyStories 上测试这些。

作为警告，我们注意到，您可能发现在此排行榜上运行良好的某些修改可能无法推广到更大规模的预训练。我们将在课程的缩放定律单元中进一步探讨这个想法。

#### **问题 (leaderboard)：排行榜 (6 分) (10 H100 小时)**
您将根据上述排行榜规则训练一个模型，目标是在 1.5 H100 小时内最小化语言模型的验证损失。
> **交付物：** 记录的最终验证损失，一个清晰显示挂钟时间 x 轴小于 1.5 小时的相关学习曲线，以及您所做工作的描述。我们期望排行榜提交至少能击败 5.0 损失的简单基线。在此处提交到排行榜：https://github.com/stanford-cs336/assignment1-basics-leaderboard。

---

## **参考文献**

1.  Ronen Eldan and Yuanzhi Li. TinyStories: How small can language models be and still speak coherent English?, 2023. arXiv:2305.07759.
2.  Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, and Stefanie Tellex. OpenWebText corpus. http://Skylion007.github.io/OpenWebTextCorpus, 2019.
3.  Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. In Proc. of ACL, 2016.
4.  Changhan Wang, Kyunghyun Cho, and Jiatao Gu. Neural machine translation with byte-level subwords, 2019. arXiv:1909.03341.
5.  Philip Gage. A new algorithm for data compression. C Users Journal, 12(2):23-38, February 1994. ISSN 0898-9788.
6.  Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners, 2019.
7.  Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training, 2018.
8.  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, L ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Proc. of NeurIPS, 2017.
9.  Toan Q. Nguyen and Julian Salazar. Transformers without tears: Improving the normalization of self-attention. In Proc. of IWSLT, 2019.
10. Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tie-Yan Liu. On layer normalization in the Transformer architecture. In Proc. of ICML, 2020.
11. Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization, 2016. arXiv:1607.06450.
12. Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023. arXiv:2302.13971.
13. Biao Zhang and Rico Sennrich. Root mean square layer normalization. In Proc. of NeurIPS, 2019.
14. Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhari, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzman, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnston, Joshua Saxe, Junteng Jia, Kalyan Vasudean Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur Celebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, VISH Vogeti, Vitor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaele Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuna Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon Clvin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ Howes, Rutty Rinott, Sachin Mehta, Sachin Sibly, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamar Best, Thilo Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.
15. An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.
16. Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. PaLM: Scaling language modeling with pathways, 2022. arXiv:2204.02311.
17. Dan Hendrycks and Kevin Gimpel. Bridging nonlinearities and stochastic regularizers with gaussian error linear units, 2016. arXiv:1606.08415.
18. Stefan Elfwing, Eiji Uchibe, and Kenji Doya. Sigmoid-weighted linear units for neural network function approximation in reinforcement learning, 2017. URL https://arxiv.org/abs/1702.03118.
19. Yann N. Dauphin, Angela Fan, Michael Auli, and David Grangier. Language modeling with gated convolutional networks, 2017. URL https://arxiv.org/abs/1612.08083.
20. Noam Shazeer. GLU variants improve transformer, 2020. arXiv:2002.05202.
21. Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding, 2021.
22. Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proc. of ICLR, 2015.
23. Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In Proc. of ICLR, 2019.
24. Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Proc. of NeurIPS, 2020.
25. Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models, 2020. arXiv:2001.08361.
26. Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. Training compute-optimal large language models, 2022. arXiv:2203.15556.
27. Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. In Proc. of ICLR, 2020.
28. Yao-Hung Hubert Tsai, Shaojie Bai, Makoto Yamada, Louis-Philippe Morency, and Ruslan Salakhutdinov. Transformer dissection: An unified understanding for transformer's attention via the lens of kernel. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4344-4353, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1443. URL https://aclanthology.org/D19-1443/.
29. Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan, Payel Das, and Siva Reddy. The impact of positional encoding on length generalization in transformers. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=Drrl2gcjz1.

