# Self-hosting an open-weight LLM on Hetzner

The agent in `agent/ollama_agent.py` talks to an Ollama endpoint. Locally that
is `http://localhost:11434`. This is how to make it a box in Falkenstein.

## 1. Pick the machine

Hetzner sells GPUs only in the **dedicated** (Robot) line, not Hetzner Cloud —
Cloud CPX/CCX instances have no GPU. As of 2026:

| Server | GPU | VRAM | ~Price/mo | Comfortably runs |
|---|---|---|---|---|
| GEX44 | RTX 4000 SFF Ada | 20 GB | €184 | 8–14B at q4, 24GB-class at q3 (often listed unavailable) |
| GEX130 | RTX 6000 Ada | 48 GB | €838 + €79 setup | 32B at q4/q8, 70B at q4 |
| GEX131 | RTX PRO 6000 Blackwell Max-Q | 96 GB | €889 | 70B at q8, 120B at q4 |

**VRAM is the only spec that matters at first.** Rough rule for a q4 quant:
GB of VRAM needed ≈ billions of parameters × 0.6, plus 2–4 GB for the KV cache
at a long context. So a 32B q4 model wants ~22 GB and fits a GEX130 easily, a
GEX44 not at all.

Two things to know before you commit:

- **Dedicated servers are monthly, not hourly.** The GEX130 has an hourly
  option; the GEX44 does not. There is no scale-to-zero — an idle box bills the
  same as a busy one. If your agent runs a few minutes a day, a per-token API
  is cheaper, and that is a real answer, not a cop-out.
- **Setup fees and availability vary.** GEX44 has been intermittently listed as
  unavailable. Check the current Robot order page before planning around it.

Start with the CPU-only path below on a cheap Cloud box if you only want to
learn the mechanics; move to a GPU when the latency actually bothers you.

## 2. Install Ollama

On a fresh Ubuntu 24.04 dedicated server:

```bash
# NVIDIA driver first — check it sees the card
sudo apt update && sudo apt install -y nvidia-driver-550
sudo reboot
nvidia-smi

curl -fsSL https://ollama.com/install.sh | sh
systemctl status ollama
```

Ollama binds `127.0.0.1:11434` by default. That default is doing real security
work — see step 4 before you change it.

## 3. Pull a model that can actually call tools

Most open-weight models cannot reliably emit tool calls. Trained-for-tool-use
families as of 2026: **Qwen3** (the safe default), Llama 3.1+, Mistral-Nemo,
and Llama-3-Groq-Tool-Use. Instruct-tuned models without tool training will
happily write a JSON blob into their prose instead of a real tool call, and
your loop will never fire.

```bash
ollama pull qwen3:8b     # ~5 GB, fits a 20 GB card with room for context
ollama pull qwen3:32b    # ~20 GB, better at multi-step chains; needs GEX130+
ollama list
```

Verify tool calling works at all before blaming your agent:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3:8b",
  "messages": [{"role":"user","content":"List the python files"}],
  "tools": [{"type":"function","function":{
      "name":"list_files",
      "description":"List files matching a glob",
      "parameters":{"type":"object","properties":{"pattern":{"type":"string"}},"required":["pattern"]}}}],
  "stream": false
}'
```

A `message.tool_calls` array in the response means you are in business. Prose
describing the function call means the model is not tool-trained — change models.

## 4. Reach it from your laptop, safely

**Do not** just set `OLLAMA_HOST=0.0.0.0` and open port 11434. Ollama has no
authentication whatsoever. An open endpoint is an open invitation to have your
GPU mine someone else's tokens, and these get scanned for continuously.

Pick one:

**a. SSH tunnel (simplest, use this first).** Nothing is exposed publicly.

```bash
ssh -N -L 11434:localhost:11434 user@your-hetzner-box
# then, unchanged:
OLLAMA_HOST=http://localhost:11434 .venv/Scripts/python.exe agent/ollama_agent.py "..."
```

**b. WireGuard / Tailscale.** Bind Ollama to the VPN interface only:

```bash
sudo systemctl edit ollama
# [Service]
# Environment="OLLAMA_HOST=100.x.x.x:11434"    # the tailnet IP, not 0.0.0.0
sudo systemctl restart ollama
```

**c. Caddy with auth, if it must be public.** Terminate TLS, require a token,
and keep Ollama on loopback:

```
llm.example.com {
    basic_auth { you <bcrypt-hash> }
    reverse_proxy 127.0.0.1:11434
}
```

Also set a firewall regardless — Hetzner Robot servers get a public IP with
everything reachable by default:

```bash
sudo ufw allow OpenSSH && sudo ufw enable
```

## 5. Keep the model warm

Ollama unloads a model after 5 minutes idle; the next request then pays the
full load time (tens of seconds for a 32B). For an agent that is called
sporadically:

```bash
sudo systemctl edit ollama
# [Service]
# Environment="OLLAMA_KEEP_ALIVE=-1"       # never unload
# Environment="OLLAMA_NUM_PARALLEL=2"
```

`-1` pins VRAM permanently. Fine on a dedicated box you own; not fine if you
share the GPU with anything else.

## 6. Point the agent at it

```bash
OLLAMA_HOST=http://localhost:11434 \
OLLAMA_MODEL=qwen3:32b \
  .venv/Scripts/python.exe agent/ollama_agent.py "What TODOs are outstanding?"
```

Note where the MCP server runs in this setup: **on your laptop, not the GPU
box.** The agent process launches `mcp_server/devtools.py` as a local
subprocess and only sends the model text over the wire. Your files never leave
your machine. If you want the server to live next to the model instead, switch
it to `transport="streamable-http"` and connect with a URL rather than
`StdioServerParameters`.

## When to skip all of this

Self-hosting wins on privacy, on fixed-cost high volume, and on not being
rate-limited. It loses on capability per euro — a €838/mo GEX130 running a 32B
model is weaker at multi-step tool use than a frontier API model you'd pay far
less to use at low volume. Choose it because you want the data locality or the
control, not because you assume it is cheaper.

## Sources

- [Hetzner GPU server docs](https://docs.hetzner.com/robot/dedicated-server/server-lines/gpu-server/)
- [Hetzner GEX130 announcement](https://www.hetzner.com/pressroom/gpu-server-gex130/)
- [Hetzner GPU review 2026 — pricing](https://gpuhosted.com/en/hetzner-gpu-review/)
- [Best local models for tool calling 2026](https://www.promptquorum.com/power-local-llm/best-local-models-tool-calling-2026)
- [Best local LLMs for tool & function calling](https://localaimaster.com/blog/best-ollama-models-tool-calling)
