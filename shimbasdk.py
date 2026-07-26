"""
shimbasdk.py — Python SDK for Shimba LLM API (and OpenAI-compatible APIs).

Connect to a local Shimba server, a remote one, or any OpenAI-compatible API.

Usage:
    from shimbasdk import ShimbaClient

    # Connect to local Shimba server
    client = ShimbaClient()

    # Or to a remote Shimba server
    client = ShimbaClient(base_url="http://192.168.1.100:8000")

    # Or to any OpenAI-compatible API
    client = ShimbaClient(base_url="https://api.openai.com/v1", api_key="sk-...")

    # Text completion
    result = client.complete(prompt="Hello", max_tokens=100)
    print(result["choices"][0]["text"])

    # Chat completion
    result = client.chat(messages=[{"role": "user", "content": "Hello"}])
    print(result["choices"][0]["message"]["content"])

    # Streaming
    for chunk in client.stream_complete(prompt="Tell me a story", max_tokens=200):
        print(chunk, end="", flush=True)

    # List models
    print(client.list_models())
"""

import json, os, sys, time
from typing import Optional, List, Dict, Any, Generator


class ShimbaClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: Optional[str] = None,
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("SHIMBA_API_KEY", "")
        self.timeout = timeout
        self._session = None

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _request(self, method: str, path: str, **kwargs) -> dict:
        import urllib.request, urllib.error
        url = f"{self.base_url}{path}"
        data = json.dumps(kwargs).encode() if kwargs else None
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
            return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            err = e.read().decode()
            try:
                return json.loads(err)
            except:
                return {"error": err, "status": e.code}
        except Exception as e:
            return {"error": str(e)}

    def _stream_request(self, path: str, **kwargs) -> Generator[str, None, None]:
        import urllib.request
        kwargs["stream"] = True
        data = json.dumps(kwargs).encode()
        req = urllib.request.Request(f"{self.base_url}{path}", data=data, headers=self._headers())
        resp = urllib.request.urlopen(req, timeout=self.timeout)
        buffer = ""
        while True:
            chunk = resp.read(1).decode()
            if not chunk:
                break
            buffer += chunk
            if buffer.endswith("\n\n"):
                for line in buffer.strip().split("\n"):
                    if line.startswith("data: "):
                        d = line[6:]
                        if d.strip() == "[DONE]":
                            return
                        yield d
                buffer = ""

    # ── Endpoints ─────────────────────────────────────────────

    def list_models(self) -> dict:
        return self._request("GET", "/v1/models")

    def health(self) -> dict:
        return self._request("GET", "/")

    def complete(self, prompt: str, max_tokens: int = 200, temperature: float = 0.8,
                 top_k: int = 40, top_p: float = 0.95, **kwargs) -> dict:
        return self._request("POST", "/v1/completions",
            prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, **kwargs)

    def chat(self, messages: List[Dict], max_tokens: int = 200, temperature: float = 0.8,
             top_k: int = 40, top_p: float = 0.95, **kwargs) -> dict:
        return self._request("POST", "/v1/chat/completions",
            messages=messages, max_tokens=max_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, **kwargs)

    def stream_complete(self, prompt: str, max_tokens: int = 200, temperature: float = 0.8,
                        top_k: int = 40, top_p: float = 0.95, **kwargs) -> Generator[str, None, None]:
        for chunk in self._stream_request("/v1/completions",
            prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, **kwargs):
            yield chunk

    def stream_chat(self, messages: List[Dict], max_tokens: int = 200, temperature: float = 0.8,
                    top_k: int = 40, top_p: float = 0.95, **kwargs) -> Generator[str, None, None]:
        for chunk in self._stream_request("/v1/chat/completions",
            messages=messages, max_tokens=max_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, **kwargs):
            yield chunk

    # ── Convenience ──────────────────────────────────────────

    def ask(self, prompt: str, **kwargs) -> str:
        """Quick text completion, returns just the text."""
        r = self.complete(prompt=prompt, **kwargs)
        return r.get("choices", [{}])[0].get("text", "")

    def ask_chat(self, message: str, system: str = "", **kwargs) -> str:
        """Quick chat, returns just the response text."""
        messages = []
        if system: messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})
        r = self.chat(messages=messages, **kwargs)
        return r.get("choices", [{}])[0].get("message", {}).get("content", "")


# ── CLI ─────────────────────────────────────────────────────
# python shimbasdk.py --url http://127.0.0.1:8000 ask "Hello"

def main():
    import argparse
    p = argparse.ArgumentParser(description="Shimba SDK CLI")
    p.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    p.add_argument("--key", default="", help="API key")
    sub = p.add_subparsers(dest="cmd")

    pa = sub.add_parser("ask", help="Text completion")
    pa.add_argument("prompt")
    pa.add_argument("--max-tokens", type=int, default=200)
    pa.add_argument("--temp", type=float, default=0.8)
    pa.add_argument("--stream", action="store_true")

    pc = sub.add_parser("chat", help="Chat completion")
    pc.add_argument("message")
    pc.add_argument("--system", default="")
    pc.add_argument("--max-tokens", type=int, default=200)
    pc.add_argument("--temp", type=float, default=0.8)
    pc.add_argument("--stream", action="store_true")

    sub.add_parser("models", help="List models")
    sub.add_parser("health", help="Health check")

    args = p.parse_args()
    client = ShimbaClient(base_url=args.url, api_key=args.key)

    if args.cmd == "ask":
        if args.stream:
            for chunk in client.stream_complete(args.prompt, max_tokens=args.max_tokens, temperature=args.temp):
                d = json.loads(chunk)
                print(d.get("choices", [{}])[0].get("text", ""), end="", flush=True)
            print()
        else:
            print(client.ask(args.prompt, max_tokens=args.max_tokens, temperature=args.temp))

    elif args.cmd == "chat":
        if args.stream:
            messages = []
            if args.system: messages.append({"role": "system", "content": args.system})
            messages.append({"role": "user", "content": args.message})
            for chunk in client.stream_chat(messages, max_tokens=args.max_tokens, temperature=args.temp):
                d = json.loads(chunk)
                print(d.get("choices", [{}])[0].get("delta", {}).get("content", ""), end="", flush=True)
            print()
        else:
            print(client.ask_chat(args.message, system=args.system, max_tokens=args.max_tokens, temperature=args.temp))

    elif args.cmd == "models":
        print(json.dumps(client.list_models(), indent=2))

    elif args.cmd == "health":
        print(json.dumps(client.health(), indent=2))

    else:
        p.print_help()


if __name__ == "__main__":
    main()
