"""
LLM API Integration Module
==========================
Unified interface for Large Language Model (LLM) APIs and open-source models:
  1. OpenAI (e.g. gpt-4o, gpt-4o-mini)
  2. Anthropic (e.g. claude-3-5-sonnet, claude-3-5-haiku)
  3. Open models via Hugging Face Hub / Pipeline
  4. Local open models via Ollama (e.g. llama3, mistral, phi3)
  5. Deterministic Mock Engine (enables 100% offline testing without API keys)

Part of Week 17-18: Deep Learning with PyTorch, CNNs, Sequence Models,
Transfer Learning, Transformers & LLM APIs.
Made by Dhanish Ladwani (https://github.com/dhanish0711/)
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import requests


@dataclass
class LLMResponse:
    provider: str
    model: str
    task: str
    output_text: str
    latency_ms: float
    token_usage: Dict[str, int]
    is_mock: bool
    status: str


class LLMClientManager:
    """
    Unified manager across commercial and open-source LLM providers.
    Gracefully falls back to high-fidelity mock simulations when API keys
    or local Ollama endpoints are unavailable.
    """

    def __init__(self):
        self.openai_key = os.environ.get("OPENAI_API_KEY", "")
        self.anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "") or os.environ.get("HF_TOKEN", "")
        self.ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def is_provider_available(self, provider: str) -> bool:
        p = provider.lower()
        if p == "openai":
            return bool(self.openai_key and self.openai_key.startswith("sk-"))
        elif p == "anthropic":
            return bool(self.anthropic_key and self.anthropic_key.startswith("sk-ant-"))
        elif p in ("huggingface", "hf"):
            return bool(self.hf_token)
        elif p == "ollama":
            try:
                res = requests.get(f"{self.ollama_host}/api/tags", timeout=1.0)
                return res.status_code == 200
            except Exception:
                return False
        elif p == "mock":
            return True
        return False

    def query(
        self,
        task: str,
        prompt: str,
        provider: str = "mock",
        model: Optional[str] = None,
        context: Optional[str] = None,
        system_instruction: Optional[str] = None
    ) -> LLMResponse:
        p = provider.lower()
        t0 = time.perf_counter()

        if not self.is_provider_available(p) or p == "mock":
            mock_res = self._mock_dispatch(task, prompt, context, p, model)
            latency = (time.perf_counter() - t0) * 1000.0
            mock_res.latency_ms = round(latency, 2)
            return mock_res

        try:
            if p == "openai":
                return self._call_openai(task, prompt, model or "gpt-4o-mini", context, system_instruction, t0)
            elif p == "anthropic":
                return self._call_anthropic(task, prompt, model or "claude-3-5-haiku-20241022", context, system_instruction, t0)
            elif p == "ollama":
                return self._call_ollama(task, prompt, model or "llama3", context, system_instruction, t0)
            elif p in ("huggingface", "hf"):
                return self._call_huggingface(task, prompt, model or "mistralai/Mistral-7B-Instruct-v0.2", context, system_instruction, t0)
            else:
                return self._mock_dispatch(task, prompt, context, p, model)
        except Exception as e:
            res = self._mock_dispatch(task, prompt, context, f"{p} (Fallback: {str(e)[:30]})", model)
            res.latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return res

    def _call_openai(self, task: str, prompt: str, model: str, context: Optional[str], system_instruction: Optional[str], t0: float) -> LLMResponse:
        import openai
        client = openai.OpenAI(api_key=self.openai_key)
        sys_msg = system_instruction or f"You are an expert deep learning assistant. Task: {task}."
        user_content = f"Context: {context}\n\nTask: {prompt}" if context else prompt

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_content}],
            temperature=0.2,
            max_tokens=300
        )
        latency = (time.perf_counter() - t0) * 1000.0
        text = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0
        }
        return LLMResponse("OpenAI", model, task, text, round(latency, 2), usage, False, "Success")

    def _call_anthropic(self, task: str, prompt: str, model: str, context: Optional[str], system_instruction: Optional[str], t0: float) -> LLMResponse:
        import anthropic
        client = anthropic.Anthropic(api_key=self.anthropic_key)
        sys_msg = system_instruction or f"You are an expert deep learning assistant. Task: {task}."
        user_content = f"Context:\n{context}\n\nInstruction:\n{prompt}" if context else prompt

        response = client.messages.create(
            model=model,
            system=sys_msg,
            max_tokens=300,
            messages=[{"role": "user", "content": user_content}]
        )
        latency = (time.perf_counter() - t0) * 1000.0
        text = response.content[0].text if response.content else ""
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens
        }
        return LLMResponse("Anthropic", model, task, text, round(latency, 2), usage, False, "Success")

    def _call_ollama(self, task: str, prompt: str, model: str, context: Optional[str], system_instruction: Optional[str], t0: float) -> LLMResponse:
        full_prompt = f"System: {system_instruction or 'You are an AI assistant.'}\n"
        if context:
            full_prompt += f"Context: {context}\n"
        full_prompt += f"Prompt: {prompt}\nResponse:"

        payload = {"model": model, "prompt": full_prompt, "stream": False, "options": {"temperature": 0.2, "num_predict": 250}}
        resp = requests.post(f"{self.ollama_host}/api/generate", json=payload, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        latency = (time.perf_counter() - t0) * 1000.0
        text = data.get("response", "")
        usage = {"total_tokens": data.get("eval_count", 0)}
        return LLMResponse("Ollama (Local)", model, task, text, round(latency, 2), usage, False, "Success")

    def _call_huggingface(self, task: str, prompt: str, model: str, context: Optional[str], system_instruction: Optional[str], t0: float) -> LLMResponse:
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        payload = {"inputs": f"{context}\n\n{prompt}" if context else prompt, "parameters": {"max_new_tokens": 150}}
        resp = requests.post(api_url, headers=headers, json=payload, timeout=30.0)
        resp.raise_for_status()
        res_json = resp.json()
        latency = (time.perf_counter() - t0) * 1000.0
        text = res_json[0].get("generated_text", "") if isinstance(res_json, list) else str(res_json)
        return LLMResponse("HuggingFace Hub", model, task, text, round(latency, 2), {"total_tokens": len(text.split())}, False, "Success")

    def _mock_dispatch(self, task: str, prompt: str, context: Optional[str], provider: str, model: Optional[str]) -> LLMResponse:
        p_name = provider.capitalize()
        m_name = model or ("gpt-4o-mini" if "openai" in provider.lower() else "claude-3-5-haiku" if "anthropic" in provider.lower() else "llama3-8b")
        t = task.lower()
        if "summar" in t:
            output = (
                "Executive Summary:\n"
                "• Convolutional Neural Networks (CNNs) extract spatial features via shift-invariant filters.\n"
                "• Recurrent architectures (LSTM/GRU) maintain hidden state vectors using gating mechanisms.\n"
                "• Transfer learning dramatically accelerates convergence by leveraging pretrained representations."
            )
        elif "question" in t or "qa" in t:
            output = (
                "Answer: Vanishing gradients occur because repeated chain-rule matrix multiplications with "
                "derivatives < 1 decay the gradient exponentially. LSTMs solve this via an additive cell state line."
            )
        else:
            output = (
                "Modern deep learning workflows combine modular autograd computation with high-level pretrained models. "
                "By leveraging transfer learning and Transformer architectures, state-of-the-art results are achievable efficiently."
            )

        tokens = len(output.split()) + len(prompt.split())
        return LLMResponse(
            provider=f"{p_name} (Simulation Engine)",
            model=m_name,
            task=task,
            output_text=output,
            latency_ms=10.0,
            token_usage={"total_tokens": tokens},
            is_mock=True,
            status="Simulated"
        )


if __name__ == "__main__":
    mgr = LLMClientManager()
    print("OpenAI available:", mgr.is_provider_available("openai"))
    res = mgr.query("Text Generation", "Explain Dropout.", provider="mock")
    print(res.output_text)
