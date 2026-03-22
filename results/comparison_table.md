# RAG vs CKE-lite Comparison Table

## hotpotqa

| Metric | RAG k=5 | RAG k=10 | CKE N=8 | CKE N=12 | CKE N=20 | Hybrid N=12 |
|--------|-------|-------|-------|-------|-------|-------|
| Answer EM | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Answer F1 | 0.1087 | 0.1122 | 0.1114 | 0.1114 | 0.1114 | 0.1250 |
| Median prompt tokens | 534 | 1091 | 59 | 59 | 59 | 61 |
| Median latency (ms) | 1.1 | 1.2 | 11.1 | 11.1 | 11.1 | 11.4 |
| Token reduction vs RAG k=10 | 2.0× | 1.0× | 18.5× | 18.5× | 18.5× | 17.9× |

## wiki2

| Metric | RAG k=5 | RAG k=10 | CKE N=8 | CKE N=12 | CKE N=20 | Hybrid N=12 |
|--------|-------|-------|-------|-------|-------|-------|
| Answer EM | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Answer F1 | 0.1092 | 0.1122 | 0.1114 | 0.1114 | 0.1114 | 0.1250 |
| Median prompt tokens | 538 | 1107 | 59 | 59 | 59 | 61 |
| Median latency (ms) | 1.2 | 1.2 | 11.6 | 11.6 | 11.6 | 11.9 |
| Token reduction vs RAG k=10 | 2.1× | 1.0× | 18.8× | 18.8× | 18.8× | 18.1× |

## combined

| Metric | RAG k=5 | RAG k=10 | CKE N=8 | CKE N=12 | CKE N=20 | Hybrid N=12 |
|--------|-------|-------|-------|-------|-------|-------|
| Answer EM | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Answer F1 | 0.1089 | 0.1122 | 0.1114 | 0.1114 | 0.1114 | 0.1250 |
| Median prompt tokens | 537 | 1099 | 59 | 59 | 59 | 61 |
| Median latency (ms) | 1.1 | 1.2 | 11.3 | 11.3 | 11.3 | 11.6 |
| Token reduction vs RAG k=10 | 2.0× | 1.0× | 18.6× | 18.6× | 18.6× | 18.0× |
