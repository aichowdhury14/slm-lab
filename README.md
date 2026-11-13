# SLM Lab: TaxBot

A Bangladesh salary-tax assistant that runs **entirely on this PC** with a small language model (SLM).
Ask in English or Bangla; no internet and no API key needed, and nothing leaves the machine.

It is also a hands-on course: each part of the code teaches one technique that real SLM products use.

## How it works

```
question ──► 1. EXTRACT   SLM + JSON schema    "80k", monthly, general, ...
                 │          (Python validates: amounts must appear in the question,
                 │           yes/no flags need supporting words)
                 ▼
             2. CALCULATE  taxbot/tax.py       exact tax, tested (no model arithmetic)
                 ▼
             3. RETRIEVE   bge-m3 embeddings   top 3 rule sections (works across Bangla/English)
                 ▼
             4. ANSWER     SLM, streamed        explains using only 2 + 3, cites [1] [2]
```

The design rule: **the model handles language and code handles numbers.** Small models can
read a question well but they slip on arithmetic and make up facts, so every number comes
from code or from the rule book.

## Lessons learned building it (real failures, all fixed in code)

| What qwen3:1.7b did | Fix |
|---|---|
| Set `new_taxpayer: true` for "My salary is 80k a month" | A yes/no flag only counts with supporting words in the question (`bot.py`) |
| Answered a Bangla question with "tax must be paid" when the calculator said Tk 0 | The verdict sentence is written by code (`headline()`); the model only explains |
| Repeated the verdict even after being told not to | Code drops the repeat; small models often ignore "don't" instructions |
| Ran on for 5,000+ tokens in a repetition loop | Output cap (`num_predict`) + Qwen3's recommended sampling (temperature 0.7, presence_penalty 1.5), not temperature ~0 |
| Copied "বার্ষিক" (the word "annual") as the salary instead of "৯ লাখ" | If the copied text has no number, use the only amount in the question (`find_amounts`) |
| Out of memory when loading qwen3:4b; models swapping in and out | Smaller context, retrieve before extracting so models swap once per question (see Memory below) |

| File | What it teaches |
|---|---|
| `taxbot/llm.py` | Talking to Ollama over HTTP: chat, streaming, structured output, embeddings, speed stats |
| `taxbot/knowledge.py` | RAG: chunking, embedding, cosine search, rebuilding the index when the knowledge changes |
| `taxbot/bot.py` | Pipeline design, prompt writing, validating model output |
| `taxbot/amounts.py` | Doing in code what the model is bad at (lakh/crore, Bangla digits) |
| `taxbot/tax.py`, `taxbot/data/tax_rules.json` | Tax engine copied from bd-salary-compass (Finance Act 2026) |
| `knowledge/*.md` | The bot's handwritten knowledge. Add a `## section` and it is indexed on the next run |
| `eval/` | Measuring quality so you can compare models and prompts with numbers |

## Setup (already done on this PC)

1. Ollama 0.40 installed; models are stored in `E:\ollama-models` (user env var `OLLAMA_MODELS`).
2. Models: `qwen3:1.7b` (default), `qwen3:4b` (better, needs more memory), `bge-m3` (embeddings).
3. Python env: `.venv` (`python -m venv .venv` then `.venv\Scripts\pip install -r requirements.txt`).

## Run it

```powershell
cd "E:\Sample Projects\slm-lab"
.venv\Scripts\activate

python ask.py --debug "My salary is 80k a month. How much tax?"     # terminal
python -m uvicorn app:app --port 8000                                # web app at http://127.0.0.1:8000
pytest                                                               # unit tests (no model needed)
python eval\run_eval.py --model qwen3:1.7b                           # quality score (~1 min per question)
```

Switch model for everything: `$env:TAXBOT_MODEL = "qwen3:4b"`.

## Learning path

Do the lessons in order; each takes a few minutes to run.

1. `python learn\01_hello_slm.py`: first local model call; tokens per second; 1.7B vs 4B.
2. `python learn\02_embeddings.py`: how a Bangla question finds an English rule.
3. `python learn\03_structured_output.py`: forcing JSON and why the bot copies amounts instead of converting them.
4. `python learn\04_why_rag.py`: the same model with and without the rule book; watch it make numbers up.
5. Read `taxbot/bot.py` top to bottom, then run `ask.py --debug` and match each line of output to a step.
6. Run the eval, change one thing (a prompt line, `k` in retrieval, the model), run it again, compare.

Exercises once that makes sense:
- Add a knowledge file (for example `knowledge/provident-fund.md`) and ask about it.
- Add keyword (BM25) search next to embeddings and merge the results ("hybrid search").
- Try a newer small model (`ollama pull <name>`), run the eval, and decide whether to switch, with numbers.
- Make the answer step use Ollama's tool calling instead of the fixed pipeline, and compare eval scores.

## Making it fast (lesson 5: `python learn\05_where_time_goes.py`)

Measure first. On this laptop the time went to model swapping (~17 s), reading a long prompt
(~20 s) and writing the answer (~14 s). Fixes, biggest first:

| Change | Calculation question | General question |
|---|---|---|
| Before | 64 s | 39 s |
| Calculator writes the explanation; no search, no generation (`--explain` / "AI explanation" brings the model back) | **7.9 s** | |
| Questions without an amount skip extraction; 2 sources instead of 3; answers capped at 250 tokens | | **20.6 s** |

The general idea: **the fastest model call is the one you skip.** Use the model only for what code can't do.

Still on the table:
- **Plug in the charger and choose "Best performance"** (Settings → System → Power). On battery the CPU throttles.
- **Enlarge the page file** (see Memory below) so the chat and embedding models stay loaded together; that removes ~5–10 s of reloading from general questions.
- **Try `qwen3:0.6b` for extraction** (`ollama pull qwen3:0.6b`), which is about 2x faster. Run the eval to see whether accuracy holds.
- Keep prompts short: every 100 prompt tokens costs ~1.2 s on this CPU.

## Next stage: fine-tuning (on Google Colab)

This PC has no NVIDIA GPU, so training happens on a free Colab T4 and only the result comes back:

1. Collect 300–1000 good question → answer pairs (generate drafts with the bot, fix them by hand, and include Bangla).
2. Fine-tune `Qwen3-1.7B` with LoRA (Unsloth or Hugging Face TRL) on Colab: about 20–40 minutes.
3. Export to GGUF (Q4_K_M), copy it here, and create the model with `ollama create taxbot -f Modelfile`.
4. Run `python eval\run_eval.py --model taxbot` and keep it only if it beats the base model.

Good targets for fine-tuning here: fluent, non-repetitive Bangla answers, and more reliable
extraction from the 1.7B model, which would let it match 4B quality at 1.7B speed.

## Memory on this PC (why the 4B model may fail)

The laptop has 16 GB RAM, but Windows caps total memory at RAM + page file (currently a fixed
4 GB on E:). With Chrome and VS Code open, under 3 GB was left, and `qwen3:4b` failed with
"out-of-memory ... kv cache". To use the 4B model:

- Close Chrome and other heavy apps, then `ollama stop qwen3:1.7b`; or
- Enlarge the page file: *System → About → Advanced system settings → Performance Settings →
  Advanced → Virtual memory*, set E: to "System managed size" (or 16384 MB), then restart.

`TAXBOT_CTX` (default 4096) sets the context window; smaller means less memory for the KV cache.

## Limits

Salary income of resident individuals only, Finance Act 2026 (assessment year 2026-27). Some
category thresholds differ between published guides; see `confidence_notes` in
`taxbot/data/tax_rules.json`. Estimates only; verify with nbr.gov.bd or a tax practitioner.
