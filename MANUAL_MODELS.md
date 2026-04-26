# Manual Testing Targets

Agents whose system prompt is controlled by the platform operator and **cannot** be overridden via API. For these, the CAID benchmark must be performed manually by sending the 50 prompts (5 combos × 5 pressure levels × 2 conditions) to the agent's native chat interface.

These agents are tested **only in the native condition** — we measure their built-in behaviour, because there is no way to swap the platform's wrapper system prompt. This is fundamentally different from the automatic API run, where each model is tested under both `vendor` and `none` conditions.

## IDE / coding assistants (built-in system prompt)

| ID | Vendor | Interface | Underlying model (typical) |
|---|---|---|---|
| gitlab-duo-chat | GitLab | GitLab.com UI or IDE extension | Claude Sonnet 4.x via Vertex |
| github-copilot-chat | GitHub / Microsoft | VS Code / JetBrains Copilot Chat | GPT-4.x / Claude (user choice) |
| cursor | Cursor | Cursor IDE | Various (Claude / GPT-4) |
| windsurf | Codeium | Windsurf IDE | Various |
| zed-ai | Zed | Zed editor | Claude / GPT-4 |
| tabnine-chat | Tabnine | IDE extension | Proprietary |

## Web chat UIs (no API system-prompt override in native mode)

| ID | Vendor | URL |
|---|---|---|
| claude.ai-web | Anthropic | claude.ai |
| chatgpt-web | OpenAI | chat.openai.com |
| gemini-web | Google | gemini.google.com |
| copilot-web | Microsoft | copilot.microsoft.com |
| perplexity | Perplexity | perplexity.ai |
| grok-web | xAI | grok.com |
| mistral-le-chat | Mistral | chat.mistral.ai |
| deepseek-chat | DeepSeek | chat.deepseek.com |
| qwen-chat | Alibaba | chat.qwen.ai |

**Important distinction:** testing `claude-sonnet-4-x` via API with no system prompt is the `none` condition (model with no directing context). Testing `claude.ai-web` is *the native claude.ai system prompt* condition. These are two different experiments. The delta between them measures the contribution of the web-interface wrapper prompt to compliance robustness.

## Procedure for manual testing

1. Create a folder `manual_results/<agent_id>/`
2. Open the agent's chat interface
3. For each of the 50 prompts in `prompts/caid_v1.json` (5 combos × 5 pressure levels × 2 conditions):
   - Open a **fresh chat** (no shared history)
   - Paste the prompt verbatim
   - Save the full response to `manual_results/<agent_id>/<combo_id>_p<pressure>_<condition>.txt`
4. N=1 is the practical default for manual runs (full N=10 is unrealistic by hand)
5. Optional: feed responses through `src/judge.py` for LLM-based classification, or classify manually

## Importing manual results into the analysis pipeline

A future addition will provide a manual-results importer that converts `manual_results/` text files into the same JSONL format as automatic runs, so the same `analyze.py` can process them.
