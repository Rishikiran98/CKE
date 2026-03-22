# Ablation: Retrieval Budget

## hotpotqa

### RAG baseline (k ablation)

| Config | EM | F1 | Median tokens | Median latency (ms) |
|--------|----|----|---------------|---------------------|
| RAG k=5 | 0.0000 | 0.1087 | 534 | 1.1 |
| RAG k=10 | 0.0000 | 0.1122 | 1091 | 1.2 |

### CKE-lite (N ablation)

| Config | EM | F1 | Median tokens | Median latency (ms) | Avg statements |
|--------|----|----|---------------|---------------------|----------------|
| CKE N=8 | 0.0000 | 0.1114 | 59 | 11.1 | n/a |
| CKE N=12 | 0.0000 | 0.1114 | 59 | 11.1 | n/a |
| CKE N=20 | 0.0000 | 0.1114 | 59 | 11.1 | n/a |

### Hybrid (graph + dense fallback)

| Config | EM | F1 | Median tokens | Median latency (ms) |
|--------|----|----|---------------|---------------------|
| Hybrid N=12 | 0.0000 | 0.1250 | 61 | 11.4 |

## wiki2

### RAG baseline (k ablation)

| Config | EM | F1 | Median tokens | Median latency (ms) |
|--------|----|----|---------------|---------------------|
| RAG k=5 | 0.0000 | 0.1092 | 538 | 1.2 |
| RAG k=10 | 0.0000 | 0.1122 | 1107 | 1.2 |

### CKE-lite (N ablation)

| Config | EM | F1 | Median tokens | Median latency (ms) | Avg statements |
|--------|----|----|---------------|---------------------|----------------|
| CKE N=8 | 0.0000 | 0.1114 | 59 | 11.6 | n/a |
| CKE N=12 | 0.0000 | 0.1114 | 59 | 11.6 | n/a |
| CKE N=20 | 0.0000 | 0.1114 | 59 | 11.6 | n/a |

### Hybrid (graph + dense fallback)

| Config | EM | F1 | Median tokens | Median latency (ms) |
|--------|----|----|---------------|---------------------|
| Hybrid N=12 | 0.0000 | 0.1250 | 61 | 11.9 |

## combined

### RAG baseline (k ablation)

| Config | EM | F1 | Median tokens | Median latency (ms) |
|--------|----|----|---------------|---------------------|
| RAG k=5 | 0.0000 | 0.1089 | 537 | 1.1 |
| RAG k=10 | 0.0000 | 0.1122 | 1099 | 1.2 |

### CKE-lite (N ablation)

| Config | EM | F1 | Median tokens | Median latency (ms) | Avg statements |
|--------|----|----|---------------|---------------------|----------------|
| CKE N=8 | 0.0000 | 0.1114 | 59 | 11.3 | n/a |
| CKE N=12 | 0.0000 | 0.1114 | 59 | 11.3 | n/a |
| CKE N=20 | 0.0000 | 0.1114 | 59 | 11.3 | n/a |

### Hybrid (graph + dense fallback)

| Config | EM | F1 | Median tokens | Median latency (ms) |
|--------|----|----|---------------|---------------------|
| Hybrid N=12 | 0.0000 | 0.1250 | 61 | 11.6 |
