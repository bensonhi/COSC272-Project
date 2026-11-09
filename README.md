# 科学声明核验数据集

模型的输入是一条 claim 和它引用的一篇或多篇摘要，输出三个标签之一：`SUPPORT`、`CONTRADICT`、`NEI`。摘要来自 `data/corpus.jsonl`，已经按句切开。证据用的是句子在原文中的编号，从 0 开始。

最终训练和评测使用 `data/train` 与 `data/test`。它们由扩充后的训练集和开发集合成，再按 7:3 随机划分。官方 `data/claims_test.jsonl` 没有使用。

## 目录

| 路径 | 作用 |
|---|---|
| `data/corpus.jsonl` | 5183 篇摘要 |
| `data/claims_train.jsonl` | 原始训练集，809 条，保留金标证据的原始分组 |
| `data/claims_dev.jsonl` | 原始开发集，300 条 |
| `data/claims_train_more.jsonl` | 扩充并整理过格式的训练集，809 条 |
| `data/claims_test.jsonl` | 官方测试集，300 条，只有 `id` 和 `claim`，本次未使用 |
| `data/train/{id}.json` | 最终训练集，776 条 claim |
| `data/test/{id}.json` | 最终测试集，333 条 claim |
| `data/cross_validation/` | 原始 train 与 dev 的五折划分，不是本次使用的 7:3 划分 |
| `build_claim_splits.py` | 生成 `data/train` 和 `data/test` |

## 原始语料和标注

`corpus.jsonl` 每行一篇文档：

- `doc_id`：整数，PubMed id
- `title`：标题
- `abstract`：句子列表
- `structured`：是否为带 `BACKGROUND`、`METHODS` 等前缀的结构化摘要

5183 篇里 977 篇是结构化摘要。摘要最短 3 句，中位数 8 句，最长 367 句。

`claims_train.jsonl` 和 `claims_dev.jsonl` 每行一条 claim：

```json
{
  "id": 12,
  "claim": "...",
  "cited_doc_ids": [33409100],
  "evidence": {
    "33409100": [
      {"sentences": [8], "label": "SUPPORT"},
      {"sentences": [12], "label": "SUPPORT"}
    ]
  }
}
```

`cited_doc_ids` 是整数。`evidence` 的键是字符串形式的文档编号。同一个文档下可以有多组 rationale。一组 rationale 里的多句合在一起才构成一套证据。`evidence` 为空对象时，这条 claim 是 NEI，没有单独的标签字段。

同一条 claim 的金标极性是统一的，不会同时出现 `SUPPORT` 和 `CONTRADICT`。

| 标签 | Train | Dev | 合计 |
|---|---:|---:|---:|
| SUPPORT | 332 | 124 | 456 |
| CONTRADICT | 173 | 64 | 237 |
| NEI | 304 | 112 | 416 |
| 合计 | 809 | 300 | 1109 |

Train 有 60 组多句金标 rationale，分布在 58 条 claim 里。Dev 有 25 组，分布在 24 条 claim 里。其余金标 rationale 都只有一句。

## 处理过程

### 1. 扩充训练集证据

`augment_train_evidence.py` 读取 `claims_train.jsonl`。NEI 原样保留。对已经有金标的摘要，模型另外标出能够单独得出同一结论的句子，写成 `SUPPORT_more` 或 `CONTRADICT_more`。结果保存在 `claims_train_more.jsonl`，接口缓存位于 `data/cache/train_evidence_more/`。

扩充不改变 claim 级标签。809 条里仍是 SUPPORT 332、CONTRADICT 173、NEI 304。其中 210 条非 NEI claim 增加了 `_more` 句子，295 条没有找到额外句子。

### 2. 整理 `claims_train_more.jsonl` 的格式

`fix_train_more_format.py` 做了两件事。

有证据的 claim，`cited_doc_ids` 改成和 `evidence` 中的文档完全一致，编号仍是整数，顺序跟 `evidence` 的键一致。NEI 的 `cited_doc_ids` 不从这里清空，而是按相同 `id` 从 `claims_train.jsonl` 抄回原来的引用列表。

存储格式上，每个 `sentences` 数组只留一个编号。原来一组里有多句时，拆成多条，`label` 不变。因此 `claims_train_more.jsonl` 里有 1363 条 rationale，且每条都只有一句：

| 标签 | rationale 数 |
|---|---:|
| SUPPORT | 661 |
| SUPPORT_more | 184 |
| CONTRADICT | 364 |
| CONTRADICT_more | 154 |

生成最终数据时，金标的多句分组仍从 `claims_train.jsonl` 和 `claims_dev.jsonl` 恢复。`_more` 每一句单独作为一组，因为它们是按“单句即可判断”标出来的。

### 3. 合并后按 7:3 划分

`build_claim_splits.py` 合并 `claims_train_more.jsonl` 与 `claims_dev.jsonl`，共 1109 条，`id` 不重复。先把 `id` 排序，再用种子 42 打乱，取前 776 条作为 train，其余 333 条作为 test。划分没有按标签分层。同一条 claim 的所有删句版本都留在 train 文件里，不会进入 test。

| 来源 | Train | Test |
|---|---:|---:|
| `claims_train_more.jsonl` | 560 | 249 |
| `claims_dev.jsonl` | 216 | 84 |
| 合计 | 776 | 333 |

原始标签在两个文件夹中的分布：

| 标签 | Train | Test |
|---|---:|---:|
| SUPPORT | 319 | 137 |
| CONTRADICT | 167 | 70 |
| NEI | 290 | 126 |

### 4. 只在 train 上删句

证据按组删除。金标的一组 rationale 是一组；每个 `_more` 句子是一组。一组要么整组留下，要么整组删除。

一条有 m 组证据的 train claim 产生 m+1 组数据：

1. `original`：文章不删任何句子，标签是原来的 SUPPORT 或 CONTRADICT。
2. `drop_1_groups` 到 `drop_{m-1}_groups`：每种删除数量随机抽一个组的子集，至少留下一整组。留下至少一整组时，标签保持原来的 SUPPORT 或 CONTRADICT。
3. `drop_all_evidence`：所有证据组都删除，标签改为 NEI。每条有证据的 train claim 都有这一组。

原本就是 NEI 的 claim 没有证据组，只保留 `original` 一组，引用文章保持全文。

除 `original` 以外，每篇摘要再删除大约 15% 的非证据句，删除数量不超过非证据句的一半。仍然保留的证据句，其原文中的前一句和后一句不参与这次删除。受这个限制，实际删掉的非证据句可能少于 15%。证据句只通过整组删除去掉，不计入这 15%。

`corpus.jsonl` 不改。删掉的句子记在每组数据的 `deleted_sentences` 里。处理后的句子在该篇摘要内从 0 重新编号，字段是 `new_index`。

## 最终数据统计

776 个 train 文件共有 1901 组数据。其中 776 组是原文，1125 组是新生成的。

新生成的 1125 组：

| 内容 | 组数 |
|---|---:|
| 证据全部删除后的 NEI | 486 |
| 仍为 SUPPORT | 397 |
| 仍为 CONTRADICT | 242 |

290 个 train 文件只有原文一组，对应原来的 NEI。其余 486 个文件含有删句后的新组。单个文件中的组数从 1 到 16。组数为 1 的文件有 290 个，组数为 2 的有 193 个，组数为 3 的有 137 个。

Test 不删句。333 条里 311 条只有一篇引用摘要，其余为 2 到 5 篇。126 条 NEI 的 `evidence` 是空数组。

## 文件结构

### `data/train/{id}.json`

`n_groups` 等于 `groups` 的长度，也就是原文加上新生成的组数。`origin` 为 `train_more` 或 `dev`。

每一组包含：

- `variant`：`original`、`drop_k_groups` 或 `drop_all_evidence`
- `dropped_group_ids`：这一组删掉的证据组
- `claim`：声明原文
- `evidence`：全部原始证据句。`kept` 表示处理后是否还在，`new_index` 是它在处理后摘要中的编号；被删掉时 `new_index` 为 `null`
- `articles`：处理后的摘要。多篇时用 `doc_id` 区分，并带有标题
- `deleted_sentences`：删掉的句子。`kind` 为 `evidence` 或 `non_evidence`
- `label`：这组数据的最终标签

`evidence` 里的 `group_id` 表示这些句子属于同一套证据。`evidence_label` 保留 `SUPPORT_more`、`CONTRADICT_more` 等句子级标签。模型要预测的是组上的 `label`，只有 `SUPPORT`、`CONTRADICT`、`NEI`。

```json
{
  "id": 1028,
  "origin": "train_more",
  "n_groups": 3,
  "groups": [
    {
      "variant": "original",
      "dropped_group_ids": [],
      "claim": "...",
      "evidence": [
        {
          "doc_id": 13923140,
          "group_id": "g0",
          "evidence_label": "SUPPORT",
          "original_index": 2,
          "sentence": "...",
          "kept": true,
          "new_index": 2
        }
      ],
      "articles": [
        {
          "doc_id": 13923140,
          "title": "...",
          "sentences": [
            {"new_index": 0, "original_index": 0, "sentence": "..."}
          ]
        }
      ],
      "deleted_sentences": [],
      "label": "SUPPORT"
    }
  ]
}
```

以 `data/train/1028.json` 为例，三组标签依次是 SUPPORT、SUPPORT、NEI。第二组把 `g0` 的第 2 句和第 3 句一起删掉，`g1` 仍完整保留，所以标签仍是 SUPPORT。第三组把证据全部删掉，标签是 NEI。

### `data/test/{id}.json`

Test 使用完整摘要，句子编号就是原文编号。

```json
{
  "id": 683,
  "origin": "train_more",
  "claim": "...",
  "evidence": [
    {
      "doc_id": 9315213,
      "group_id": "g0",
      "evidence_label": "CONTRADICT",
      "original_index": 4,
      "sentence": "..."
    }
  ],
  "articles": [
    {
      "doc_id": 9315213,
      "title": "...",
      "sentences": [
        {"original_index": 0, "sentence": "..."}
      ]
    }
  ],
  "label": "CONTRADICT"
}
```

NEI 的 `evidence` 为 `[]`，`articles` 仍是原来的引用摘要。

## 复现

整理扩充文件并重新生成两个文件夹：

```bash
python3 fix_train_more_format.py
python3 build_claim_splits.py
```

第二条命令会清空并重写 `data/train` 和 `data/test`。划分种子是 42。
