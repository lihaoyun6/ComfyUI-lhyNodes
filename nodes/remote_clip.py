import socket
import struct
import threading
import time
import json
import torch
import folder_paths
import comfy.sd
import comfy.utils
from comfy.model_management import processing_interrupted
import gc
import comfy.model_management

class RemoteSerializer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return {"__tensor__": True, "value": obj.detach().cpu().tolist(), "dtype": str(obj.dtype)}
        return super().default(obj)

def remote_hook(dct):
    if "__tensor__" in dct:
        dtype = getattr(torch, dct["dtype"].split(".")[-1])
        return torch.tensor(dct["value"], dtype=dtype)
    return dct

def log(msg):
    print(f"{msg}", flush=True)

def send_packet(sock, meta_dict, blob_data):
    meta_bytes = json.dumps(meta_dict, cls=RemoteSerializer).encode("utf-8")
    sock.sendall(struct.pack(">Q", len(meta_bytes)))
    sock.sendall(meta_bytes)
    sock.sendall(blob_data)

def recv_exact(sock, size):
    buf = b""
    while len(buf) < size:
        chunk = sock.recv(size - len(buf))
        if not chunk: raise ConnectionError("Socket closed")
        buf += chunk
    return buf

def recv_packet(sock):
    raw_size = recv_exact(sock, 8)
    meta_size = struct.unpack(">Q", raw_size)[0]
    meta = json.loads(recv_exact(sock, meta_size).decode("utf-8"), object_hook=remote_hook)
    blob = recv_exact(sock, meta.get("blob_size", 0))
    return meta, blob

def pack_tensors(tensors_dict):
    meta, blobs = {}, []
    for name, t in tensors_dict.items():
        if not isinstance(t, torch.Tensor): continue
        t = t.detach().contiguous().cpu()
        raw = t.numpy().tobytes()
        meta[name] = {"dtype": str(t.dtype).split('.')[-1], "shape": list(t.shape), "size": len(raw)}
        blobs.append(raw)
    return meta, b"".join(blobs)

def unpack_tensors(meta, blob):
    out, offset = {}, 0
    for name, info in meta.items():
        size, dtype = info["size"], getattr(torch, info["dtype"])
        raw = bytearray(blob[offset:offset + size])
        offset += size
        out[name] = torch.frombuffer(raw, dtype=dtype).clone().reshape(info["shape"])
    return out

class RemoteCLIPProxy:
    def __init__(self, ip, port, clip_id=0):
        self.ip = ip
        self.port = port
        self.clip_id = clip_id
        self.load_device = torch.device("cpu")
        self._lock = threading.Lock()
        self._sock = None
        
    def tokenize(self, text, **kwargs):
        return {"remote_text": text, "remote_kwargs": kwargs, "clip_id": self.clip_id}
    
    def _get_conn(self, force_reconnect=False):
        if force_reconnect and self._sock:
            try: self._sock.close()
            except: pass
            self._sock = None
            
        if self._sock is None:
            try:
                self._sock = socket.create_connection((self.ip, self.port), timeout=10)
                self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                log(f"Connected to CLIP_{self.clip_id}@{self.ip}:{self.port}")
            except Exception as e:
                self._sock = None
                raise RuntimeError(f"Connection failed: {e}")
        return self._sock
    
    def encode_from_tokens_scheduled(self, tokens):
        for attempt in range(2):
            try:
                with self._lock:
                    conn = self._get_conn(force_reconnect=(attempt > 0))
                    meta = {
                        "type": "clip",
                        "text": tokens["remote_text"], 
                        "kwargs": tokens["remote_kwargs"], 
                        "clip_id": tokens["clip_id"]
                    }
                    send_packet(conn, meta, b"")
                    
                    resp_meta, blob = recv_packet(conn)
                    res = unpack_tensors(resp_meta["tensors"], blob)
                    
                    cond = res.pop("cond")
                    if "pooled" in res:
                        res["pooled_output"] = res.pop("pooled")
                    return [[cond, res]]
                
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, socket.timeout):
                if attempt == 0:
                    time.sleep(1)
                    continue
                raise RuntimeError("Server has been disconnected!")
            except Exception as e:
                raise e
                
    def encode_from_tokens(self, tokens, return_pooled=True, return_dict=False):
        out = self.encode_from_tokens_scheduled(tokens)
        cond, extra = out[0]
        pooled = extra.get("pooled_output")
        if return_dict:
            return {"cond": cond, "pooled_output": pooled, **extra}
        return cond, pooled
    
    def clone(self): return self
    def add_patches(self, *_, **__): return self
    @property
    def patcher(self): return None

class RemoteVAEProxy:
    def __init__(self, ip, port, vae_id=0):
        self.ip = ip
        self.port = port
        self.vae_id = vae_id
        self._lock = threading.Lock()
        self._sock = None
        
    def _get_conn(self, force_reconnect=False):
        if force_reconnect and self._sock:
            try: self._sock.close()
            except: pass
            self._sock = None
            
        if self._sock is None:
            try:
                self._sock = socket.create_connection((self.ip, self.port), timeout=10)
                self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                log(f"Connected to VAE_{self.vae_id}@{self.ip}:{self.port}")
            except Exception as e:
                self._sock = None
                raise RuntimeError(f"Connection failed: {e}")
        return self._sock
    
    def _send_request(self, cmd, tensors_in):
        for attempt in range(2):
            try:
                with self._lock:
                    conn = self._get_conn(force_reconnect=(attempt > 0))
                    meta_t, blob = pack_tensors(tensors_in)
                    
                    send_packet(conn, {"type": "vae", "cmd": cmd, "vae_id": self.vae_id, "tensors": meta_t, "blob_size": len(blob)}, blob)
                    
                    resp_meta, resp_blob = recv_packet(conn)
                    return unpack_tensors(resp_meta["tensors"], resp_blob)
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, socket.timeout):
                if attempt == 0:
                    #log("Connection lost, retrying...")
                    time.sleep(1)
                    continue
                raise RuntimeError("Server has been disconnected!")
            except Exception as e:
                raise e
                
    def decode(self, samples_in):
        res = self._send_request("decode", {"samples": samples_in})
        return res["image"]
    
    def encode(self, pixel_values):
        res = self._send_request("encode", {"pixels": pixel_values})
        return res["samples"]
    
class RemoteClipServer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
    "required": {
        "bind_ip": ("STRING", {"default": "0.0.0.0"}),
        "listen_port": ("INT", {"default": 8181, "min": 1024, "max": 65535}),
    },
    "optional": {
        "clip_1": ("CLIP",), "clip_2": ("CLIP",), "clip_3": ("CLIP",), "clip_4": ("CLIP",),
        "vae_1": ("VAE",), "vae_2": ("VAE",), "vae_3": ("VAE",), "vae_4": ("VAE",),
    }
    }
    
    RETURN_TYPES = ()
    FUNCTION = "main"
    OUTPUT_NODE = True
    CATEGORY = "Remote Server"
    
    def main(self, bind_ip, listen_port, **kwargs):
        clips = {0: kwargs.get("clip_1"), 1: kwargs.get("clip_2"), 2: kwargs.get("clip_3"), 3: kwargs.get("clip_4")}
        vaes = {0: kwargs.get("vae_1"), 1: kwargs.get("vae_2"), 2: kwargs.get("vae_3"), 3: kwargs.get("vae_4")}
        
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.settimeout(0.5) 
        
        try:
            server.bind((bind_ip, listen_port))
            server.listen(20)
            log(f"RemoteCLIP Worker running on port {listen_port}")
        except Exception as e:
            log(f"RemoteCLIP Worker start failed: {e}")
            return ()
        
        def handle_client(conn, addr):
            with conn:
                while True:
                    try:
                        meta, blob = recv_packet(conn)
                        if not meta: break
                        
                        req_type = meta.get("type", "clip")
                        
                        with torch.inference_mode():
                            if req_type == "clip":
                                clip_id = meta.get("clip_id", 0)
                                target_clip = clips.get(clip_id)
                                if target_clip is None: break
                                
                                tokens = target_clip.tokenize(meta["text"], **meta["kwargs"])
                                out = target_clip.encode_from_tokens_scheduled(tokens)
                                cond, extra = out[0]
                                
                                to_send = {"cond": cond}
                                for k, v in extra.items():
                                    if isinstance(v, torch.Tensor):
                                        to_send["pooled" if k == "pooled_output" else k] = v
                                        
                                meta_t, res_blob = pack_tensors(to_send)
                                send_packet(conn, {"tensors": meta_t, "blob_size": len(res_blob)}, res_blob)
                                
                            elif req_type == "vae":
                                vae_id = meta.get("vae_id", 0)
                                cmd = meta.get("cmd")
                                target_vae = vaes.get(vae_id)
                                if target_vae is None: break
                                
                                tensors = unpack_tensors(meta["tensors"], blob)
                                if cmd == "decode":
                                    out_res = target_vae.decode(tensors["samples"])
                                    res_key = "image"
                                else:
                                    out_res = target_vae.encode(tensors["pixels"])
                                    res_key = "samples"
                                    
                                meta_t, res_blob = pack_tensors({res_key: out_res})
                                send_packet(conn, {"tensors": meta_t, "blob_size": len(res_blob)}, res_blob)
                                
                    except Exception as e:
                        log(f"Client {addr} error: {e}")
                        break
                    
        try:
            while True:
                if processing_interrupted():
                    break
                
                try:
                    conn, addr = server.accept()
                    t = threading.Thread(target=handle_client, args=(conn, addr))
                    t.daemon = True
                    t.start()
                except socket.timeout:
                    continue
        finally:
            server.close()
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        return ()

class RemoteClipClient:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "worker_ip": ("STRING", {"default": "127.0.0.1"}),
                "port": ("INT", {"default": 8181, "min": 8000, "max": 9999, "step": 1}),
                "refesh": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "You can toggle this switch to force a reacquisition when you change the remote CLIP."
                })
            }
        }

    RETURN_TYPES = ("CLIP", "CLIP", "CLIP", "CLIP", "VAE", "VAE", "VAE", "VAE")
    RETURN_NAMES = ("CLIP_1", "CLIP_2", "CLIP_3", "CLIP_4", "VAE_1", "VAE_2", "VAE_3", "VAE_4")
    FUNCTION = "main"
    CATEGORY = "RemoteCLIP"

    def main(self, worker_ip, port, refesh):
        clip1 = RemoteCLIPProxy(worker_ip, port, clip_id=0)
        clip2 = RemoteCLIPProxy(worker_ip, port, clip_id=1)
        clip3 = RemoteCLIPProxy(worker_ip, port, clip_id=3)
        clip4 = RemoteCLIPProxy(worker_ip, port, clip_id=4)
        vae1 = RemoteVAEProxy(worker_ip, port, vae_id=0)
        vae2 = RemoteVAEProxy(worker_ip, port, vae_id=1)
        vae3 = RemoteVAEProxy(worker_ip, port, vae_id=3)
        vae4 = RemoteVAEProxy(worker_ip, port, vae_id=4)
        return (clip1, clip2, clip3, clip4, vae1, vae2, vae3, vae4)

NODE_CLASS_MAPPINGS = {
    "RemoteClipServer": RemoteClipServer,
    "RemoteClipClient": RemoteClipClient
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteClipServer": "RemoteCLIP Server",
    "RemoteClipClient": "RemoteCLIP Client"
}