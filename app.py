#!/usr/bin/env python3
"""Stroncius - anonymous AI chat on Tor.

Python stdlib only. No accounts, no server-side history.
Proxies to local llama.cpp servers (OpenAI-compatible API).

Conversations live in the browser's localStorage only.
"""

import html
import json
import re
import threading
import time
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 18098

MODELS = {
    "llama": {"name": "Llama 3.2 1B", "url": "http://127.0.0.1:18090/v1/chat/completions"},
    "smol": {"name": "SmolLM2 1.7B", "url": "http://127.0.0.1:18091/v1/chat/completions"},
}

SYSTEM_PROMPT = ("You are Stroncius, a helpful AI assistant running on a small "
                 "local model inside the Tor network. Be concise and direct. "
                 "Answer in the same language the user writes in.")

MAX_MSGS = 20          # max messages per request
MAX_MSG_LEN = 2000     # max chars per message
MAX_TOKENS = 512
RATE_LIMIT = 30        # requests per hour per IP
RATE_WINDOW = 3600

_rate = {}
_rate_lock = threading.Lock()


def check_rate(ip):
    with _rate_lock:
        t = time.time()
        lst = [x for x in _rate.get(ip, []) if t - x < RATE_WINDOW]
        if len(lst) >= RATE_LIMIT:
            return False
        lst.append(t)
        _rate[ip] = lst
        return True


def esc(s):
    return html.escape(s or "", quote=True)


CSS = """
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0d0d0e;--panel:#151517;--panel2:#1c1c1f;--border:#232326;--border2:#323237;
--text:#f4f4f5;--muted:#9c9ca6;--faint:#63636c;--acc:#8ab4ff}
html{-webkit-text-size-adjust:100%;height:100%}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,
"Segoe UI",Roboto,Helvetica,Arial,sans-serif;line-height:1.55;font-size:15px;
-webkit-font-smoothing:antialiased;height:100%;display:flex;flex-direction:column}
a{color:inherit}
.topbar{background:rgba(13,13,14,.94);backdrop-filter:blur(12px);
-webkit-backdrop-filter:blur(12px);border-bottom:1px solid var(--border);z-index:20}
.topbar-in{max-width:720px;margin:0 auto;display:flex;align-items:center;gap:10px;
padding:10px 16px;min-height:58px}
.wordmark{font-size:19px;font-weight:800;letter-spacing:-.03em;text-decoration:none}
.wordmark .dot{color:var(--faint)}
.badge{font-size:11px;font-weight:700;color:var(--faint);background:var(--panel);
border:1px solid var(--border);border-radius:6px;padding:3px 8px}
.spacer{flex:1}
select{background:var(--panel);color:var(--text);border:1px solid var(--border2);
border-radius:10px;padding:9px 12px;font-size:13.5px;font-family:inherit;min-height:44px}
.newbtn{background:var(--panel2);border:1px solid var(--border2);color:var(--text);
border-radius:10px;padding:9px 16px;font-size:13.5px;font-weight:700;cursor:pointer;
font-family:inherit;min-height:44px}
#chat{flex:1;overflow-y:auto;max-width:720px;width:100%;margin:0 auto;padding:20px 16px}
.msg{margin-bottom:18px;max-width:88%}
.msg .bubble{padding:12px 16px;border-radius:16px;font-size:14.5px;white-space:pre-wrap;
word-break:break-word}
.msg.user{margin-left:auto}
.msg.user .bubble{background:var(--panel2);border:1px solid var(--border2);
border-bottom-right-radius:5px}
.msg.ai .bubble{background:var(--panel);border:1px solid var(--border);
border-bottom-left-radius:5px}
.msg .meta{font-size:11px;color:var(--faint);margin-top:5px}
.msg.user .meta{text-align:right}
.typing{display:inline-block}
.typing span{width:7px;height:7px;background:var(--muted);border-radius:50%;
display:inline-block;margin-right:4px;animation:blink 1.2s infinite}
.typing span:nth-child(2){animation-delay:.2s}
.typing span:nth-child(3){animation-delay:.4s}
@keyframes blink{0%,100%{opacity:.25}50%{opacity:1}}
.empty{text-align:center;color:var(--faint);padding:60px 20px}
.empty h1{font-size:24px;font-weight:800;color:var(--text);margin-bottom:10px;
letter-spacing:-.02em}
.empty p{font-size:14px;max-width:380px;margin:0 auto 8px}
.suggestions{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:20px}
.suggestions button{background:var(--panel);border:1px solid var(--border2);
color:var(--text);border-radius:999px;padding:10px 18px;font-size:13.5px;cursor:pointer;
font-family:inherit;min-height:44px}
.suggestions button:hover{border-color:var(--muted)}
.composer{border-top:1px solid var(--border);background:rgba(13,13,14,.96);
padding:12px 16px calc(12px + env(safe-area-inset-bottom))}
.composer-in{max-width:720px;margin:0 auto;display:flex;gap:10px;align-items:flex-end}
textarea{flex:1;background:var(--panel);border:1px solid var(--border2);
border-radius:14px;padding:13px 16px;color:var(--text);font-size:15px;
font-family:inherit;resize:none;max-height:140px;min-height:52px;line-height:1.4}
textarea:focus{outline:none;border-color:var(--muted)}
.sendbtn{flex:none;background:var(--text);color:#0d0d0e;border:none;border-radius:14px;
width:52px;height:52px;font-size:18px;cursor:pointer;font-weight:700}
.sendbtn:disabled{opacity:.4;cursor:default}
.sendbtn:not(:disabled):active{transform:scale(.94)}
.err{background:rgba(244,63,94,.08);border:1px solid rgba(244,63,94,.35);color:#fda4b4;
padding:10px 16px;border-radius:12px;margin-bottom:14px;font-size:13.5px;text-align:center}
.hintline{text-align:center;color:var(--faint);font-size:11.5px;padding:8px 0 2px}
@media(max-width:660px){
.topbar-in{padding:8px 12px;min-height:54px}
#chat{padding:14px 12px}
.msg{max-width:94%}
.composer{padding:10px 12px calc(10px + env(safe-area-inset-bottom))}
}
"""

JS = """
var chatEl=document.getElementById('chat'),ta=document.getElementById('ta'),
sendBtn=document.getElementById('send'),modelSel=document.getElementById('model'),
busy=false;
var history=JSON.parse(localStorage.getItem('stroncius_history')||'[]');
function save(){try{localStorage.setItem('stroncius_history',JSON.stringify(history.slice(-40)));}catch(e){}}
function render(){
chatEl.innerHTML='';
if(!history.length){
chatEl.innerHTML='<div class="empty"><h1>Ask anything.</h1>'
+'<p>Anonymous AI chat on Tor. No accounts, no logs. '
+'Conversations stay in your browser only.</p>'
+'<div class="suggestions">'
+'<button onclick="ask(this)">Explain Tor in simple terms</button>'
+'<button onclick="ask(this)">Write a haiku about privacy</button>'
+'<button onclick="ask(this)">Python quicksort example</button>'
+'</div></div>';return;}
history.forEach(function(m){
var d=document.createElement('div');d.className='msg '+(m.role==='user'?'user':'ai');
var b=document.createElement('div');b.className='bubble';b.textContent=m.content;
var meta=document.createElement('div');meta.className='meta';
meta.textContent=(m.role==='user'?'You':modelSel.options[modelSel.selectedIndex].text);
d.appendChild(b);d.appendChild(meta);chatEl.appendChild(d);});
chatEl.scrollTop=chatEl.scrollHeight;}
function ask(btn){ta.value=btn.textContent;send();}
function showErr(t){var e=document.createElement('div');e.className='err';e.textContent=t;
chatEl.appendChild(e);chatEl.scrollTop=chatEl.scrollHeight;
setTimeout(function(){e.remove();},5000);}
function typing(){
var d=document.createElement('div');d.className='msg ai';d.id='typing';
d.innerHTML='<div class="bubble"><span class="typing"><span></span><span></span><span></span></span></div>';
chatEl.appendChild(d);chatEl.scrollTop=chatEl.scrollHeight;}
function send(){
var v=ta.value.trim();if(!v||busy)return;
busy=true;sendBtn.disabled=true;
history.push({role:'user',content:v});save();ta.value='';ta.style.height='auto';render();typing();
var msgs=history.slice(-20);
fetch('/api/chat',{method:'POST',credentials:'same-origin',
headers:{'Content-Type':'application/json'},
body:JSON.stringify({model:modelSel.value,messages:msgs})})
.then(function(r){return r.json().then(function(j){return {s:r.status,j:j};});})
.then(function(r){
document.getElementById('typing').remove();
if(r.s===200&&r.j.reply){history.push({role:'assistant',content:r.j.reply});}
else{showErr(r.j.error||'Something went wrong. Try again.');}
save();render();})
.catch(function(){document.getElementById('typing');var t=document.getElementById('typing');if(t)t.remove();
showErr('Connection failed. Try again.');})
.finally(function(){busy=false;sendBtn.disabled=false;ta.focus();});}
ta.addEventListener('input',function(){ta.style.height='auto';
ta.style.height=Math.min(ta.scrollHeight,140)+'px';});
ta.addEventListener('keydown',function(e){
if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}});
sendBtn.onclick=send;
document.getElementById('newchat').onclick=function(){
if(!busy&&confirm('Start a new conversation?')){history=[];save();render();}};
modelSel.value=localStorage.getItem('stroncius_model')||'llama';
modelSel.onchange=function(){localStorage.setItem('stroncius_model',modelSel.value);};
render();ta.focus();
"""

PAGE = ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1,viewport-fit=cover'>"
        "<meta name='description' content='Anonymous AI chat on Tor. No accounts, no logs.'>"
        "<title>Stroncius - Anonymous AI chat</title><style>" + CSS + "</style></head>"
        "<body>"
        "<div class='topbar'><div class='topbar-in'>"
        "<a class='wordmark' href='/'>Stroncius<span class='dot'>.</span></a>"
        "<span class='badge'>anonymous AI</span><span class='spacer'></span>"
        "<select id='model'><option value='llama'>Llama 3.2 1B</option>"
        "<option value='smol'>SmolLM2 1.7B</option></select>"
        "<button class='newbtn' id='newchat'>New</button>"
        "</div></div>"
        "<div id='chat'></div>"
        "<div class='composer'><div class='composer-in'>"
        "<textarea id='ta' rows='1' placeholder='Ask anything...' maxlength='2000'></textarea>"
        "<button class='sendbtn' id='send'>&uarr;</button>"
        "</div><div class='hintline'>Runs on small local models &middot; nothing leaves Tor</div></div>"
        "<script>" + JS + "</script></body></html>")


class Handler(BaseHTTPRequestHandler):
    server_version = "Stroncius/1.0"
    protocol_version = "HTTP/1.1"

    def client_ip(self):
        return self.client_address[0]

    def send_json(self, obj, code=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            data = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "9")
            self.end_headers()
            self.wfile.write(b"not found")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path != "/api/chat":
            self.send_json({"error": "not found"}, 404)
            return
        if not check_rate(self.client_ip()):
            self.send_json({"error": "Rate limit reached. Slow down a little."}, 429)
            return
        try:
            ln = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self.send_json({"error": "bad request"}, 400)
            return
        if ln <= 0 or ln > 65536:
            self.send_json({"error": "bad request"}, 400)
            return
        try:
            body = json.loads(self.rfile.read(ln).decode("utf-8", "replace"))
        except Exception:
            self.send_json({"error": "bad request"}, 400)
            return
        model_key = body.get("model", "llama")
        model = MODELS.get(model_key, MODELS["llama"])
        msgs = body.get("messages", [])
        if not isinstance(msgs, list) or not msgs:
            self.send_json({"error": "no messages"}, 400)
            return
        # sanitize: only role/content, cap length and count
        clean = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in msgs[-MAX_MSGS:]:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            content = str(m.get("content", ""))[:MAX_MSG_LEN]
            if content.strip():
                clean.append({"role": role, "content": content})
        if len(clean) <= 1:
            self.send_json({"error": "no messages"}, 400)
            return
        payload = json.dumps({
            "messages": clean,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.7,
        }).encode("utf-8")
        try:
            req = urllib.request.Request(
                model["url"], data=payload,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.loads(r.read().decode("utf-8", "replace"))
            reply = (resp.get("choices", [{}])[0]
                     .get("message", {}).get("content", "").strip())
            if not reply:
                self.send_json({"error": "Empty response from model."}, 502)
                return
            self.send_json({"reply": reply})
        except urllib.error.URLError:
            self.send_json({"error": "Model is busy. Try again in a moment."}, 502)
        except Exception:
            self.send_json({"error": "Something went wrong."}, 500)

    def log_message(self, *args):
        pass


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print("Stroncius on 127.0.0.1:%d" % PORT, flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
