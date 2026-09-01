### Experiment 1 - Search mode (chunk=256, no rerank)

| Configuration | mrr_at_5 | recall_at_1 | recall_at_3 | ndcg_at_5 |
|---|---|---|---|---|
| Dense only | 0.883 | 0.800 | 0.960 | 0.913 |
| Hybrid (dense + BM25) | 0.907 | 0.840 | 1.000 | 0.930 |

### Experiment 2 - Cross-encoder reranking (hybrid, chunk=256)

| Configuration | mrr_at_5 | recall_at_1 | recall_at_3 | ndcg_at_5 |
|---|---|---|---|---|
| Hybrid, no rerank | 0.907 | 0.840 | 1.000 | 0.930 |
| Hybrid + rerank | 1.000 | 1.000 | 1.000 | 1.000 |

### Experiment 3 - Chunk size (hybrid, no rerank)

| Configuration | mrr_at_5 | recall_at_1 | recall_at_3 | ndcg_at_5 |
|---|---|---|---|---|
| 128 tokens | 0.863 | 0.760 | 0.960 | 0.898 |
| 256 tokens | 0.907 | 0.840 | 1.000 | 0.930 |
| 512 tokens | 0.930 | 0.880 | 0.960 | 0.948 |

### Experiment 4 - Chunking method at matched size (hybrid, no rerank)

| Configuration | mrr_at_5 | recall_at_1 | recall_at_3 | ndcg_at_5 |
|---|---|---|---|---|
| Word-split (~256 tok) | 0.893 | 0.800 | 1.000 | 0.921 |
| Sentence-aware (256 tok) | 0.907 | 0.840 | 1.000 | 0.930 |
