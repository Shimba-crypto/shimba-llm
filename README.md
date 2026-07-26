# Shimba LLM

A complete, portable LLM training system in a single Python file. Train, serve, and chat with transformer language models — CPU-only, runs anywhere.

## Quick Start

```bash
git clone https://github.com/Shimba-crypto/shimba-llm.git
cd shimba-llm
./install.sh                    # one-command setup
python shimba.py serve --model BaseModel.pth
```

Open **http://localhost:8000** for the chat UI.

## Commands

| Command | Description |
|---|---|
| `python shimba.py train --data file.txt --out model.pth` | Train a model |
| `python shimba.py serve --model model.pth` | Start API + web UI |
| `python shimba.py chat --model model.pth` | Terminal chat |
| `python shimba.py benchmark --model model.pth` | Run benchmarks |
| `python dataset.py prepare` | Convert chat exports to training data |
| `python dataset.py stats` | Show dataset statistics |
| `python shimbasdk.py` | Python client SDK (CLI mode) |

## Web UI

![Shimba Chat UI](https://img.shields.io/badge/UI-Claude%20Style-8B5CF6)

Served at `/` — dark theme, streaming, chat history, temperature/top-k controls, markdown rendering, file upload, RAG, prompt templates, model comparison, Python code execution, conversation branching, semantic search, PDF export, PWA installable.

## Features

- Single-file design (`shimba.py`) — zero external deps beyond PyTorch
- GPT-style transformer with RoPE, GELU, multi-head attention
- Character-level tokenizer (built from your data)
- OpenAI-compatible API: `/v1/completions`, `/v1/chat/completions`, `/v1/models`, SSE streaming
- Tools: Calculator, Web Search, Python code execution
- RAG: upload documents, search web, semantic search over chats
- Python SDK (`shimbasdk.py`): `ShimbaClient(base_url, api_key)`
- Any-cloud deploy: `./deploy.sh colab`, `./deploy.sh gcp-run`, `./deploy.sh aws`
- Docker: `docker compose up`
- Google Colab: `shimba_colab.ipynb` with "Open in Colab" badge

## Train a Real Model

```bash
# Collect training data from chat exports
python dataset.py prepare --input chats/ --output training_data.txt

# Train (50M param example — ~3 min on Colab GPU)
python shimba.py train --data training_data.txt --out my_model.pth \
  --n_embd 512 --n_layer 12 --n_head 8 --block_size 512 \
  --batch_size 8 --max_iters 5000

# Serve it
python shimba.py serve --model my_model.pth
```

## Deploy

```bash
./deploy.sh colab        # Google Colab (free GPU)
./deploy.sh gcp-run      # Google Cloud Run
./deploy.sh aws          # AWS EC2
./deploy.sh azure        # Azure VM
./deploy.sh huggingface  # HuggingFace Spaces
```

## Architecture

| Component | File |
|---|---|
| GPT model + training | `shimba.py` |
| API server (OpenAI-compatible) | `shimba.py` (`ShimbaAPI` class) |
| Web UI | `index.html` |
| Python SDK | `shimbasdk.py` |
| Data pipeline | `dataset.py` |
| Multi-file module | `llm/` (model.py, tokenizer.py, data.py, train.py, generate.py) |
| Deploy scripts | `deploy.sh`, `Dockerfile`, `docker-compose.yml`, `cloudbuild.yaml` |
| Colab notebook | `shimba_colab.ipynb` |
| Installers | `install.sh`, `install.bat`, `setup.sh`, `setup.bat` |

## License

MIT
