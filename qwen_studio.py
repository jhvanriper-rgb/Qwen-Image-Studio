#!/usr/bin/env python3
"""Qwen Image Studio — desktop front end for Qwen-Image-2.1-Turbo.

Opens a window, starts the model engine in the background, then gives you a
prompt box, the pipeline's real options, a live preview and a session gallery.
Every image is saved as a PNG (settings embedded) in the output folder.

Run it with the Python from the environment that has torch + diffusers:
    ~/qwen-image-env/bin/python qwen_studio.py

Options:
    --gpu auto|0|1        which card to use (default auto = the one with most VRAM)
    --offload model|sequential|none
                          model      one model part on the GPU at a time (default)
                          sequential layer by layer, very slow, needs little VRAM
                          none       everything on the GPU (needs the VRAM)
    --dtype bf16|fp16     weight precision (default bf16)
    --compat auto|on|off  workarounds for pre-Ampere cards such as the V100:
                          plain attention, VAE in 32-bit (default auto)
    --out DIR             where images are saved (default: "output" next to this
                          script; the log file studio.log lives next to it too)
    --model ID_OR_PATH    default Qwen/Qwen-Image-2.1-Turbo
    --no-autoload         open the window without loading the model
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import queue
import random
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

# The model cache lives on the models drive. A launcher started outside a shell
# won't have the shell's HF_HOME, so point at it when it exists. This has to
# happen before torch / diffusers are imported.
if "HF_HOME" not in os.environ and "HF_HUB_CACHE" not in os.environ:
    if Path("/mnt/models/.hf").is_dir():
        os.environ["HF_HOME"] = "/mnt/models/.hf"

_TK_ERROR = None
try:
    import tkinter as tk
    from tkinter import filedialog, ttk

    from PIL import ImageTk
except ImportError as _exc:  # pragma: no cover
    tk, _TK_ERROR = None, _exc

from PIL import Image, PngImagePlugin

LOG_PATH: Path | None = None


def log(msg: str) -> None:
    """Timestamped line to the terminal and, once known, the log file."""
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    print(line, file=sys.stderr, flush=True)
    if LOG_PATH is not None:
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass

DEFAULT_MODEL = "Qwen/Qwen-Image-2.1-Turbo"
APP_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = APP_DIR / "output"
STEP = 32  # the pipeline floors width/height to a multiple of 32

# (label, width, height). The large ones are the sizes on the model card; the
# smaller ones are drafts that are much quicker and lighter on an older GPU.
PRESETS = [
    ("1:1   1024 × 1024  (draft)", 1024, 1024),
    ("4:3   1152 × 864  (draft)", 1152, 864),
    ("3:4   864 × 1152  (draft)", 864, 1152),
    ("3:2   1216 × 832  (draft)", 1216, 832),
    ("2:3   832 × 1216  (draft)", 832, 1216),
    ("16:9  1344 × 768  (draft)", 1344, 768),
    ("9:16  768 × 1344  (draft)", 768, 1344),
    ("1:1   2048 × 2048", 2048, 2048),
    ("4:3   2400 × 1792", 2400, 1792),
    ("3:4   1792 × 2400", 1792, 2400),
    ("3:2   2528 × 1696", 2528, 1696),
    ("2:3   1696 × 2528", 1696, 2528),
    ("16:9  2752 × 1536", 2752, 1536),
    ("9:16  1536 × 2752", 1536, 2752),
]

BG = "#14161d"
CARD = "#1d212c"
TRACK = "#2a3040"
TEXT = "#e8ebf2"
SUB = "#8b93a7"
ACCENT = "#4da3ff"
GOOD = "#3ddc84"
WARN = "#ffb545"
BAD = "#ff5c5c"


# ---------------------------------------------------------------------------
# Engine: owns the pipeline. All of its methods run on the one worker thread.
# ---------------------------------------------------------------------------

class Cancelled(Exception):
    pass


def patch_vae_encode(pipe) -> None:
    """Make reference/edit images work when the VAE runs in float32.

    The pipeline converts input images to the model dtype (bf16) before
    VAE-encoding them, which fails against a float32 VAE ("Input type
    (BFloat16) and bias type (float) should be the same"). Encode in the
    VAE's own dtype instead and hand the latents back in the original dtype.
    """
    original = pipe._encode_vae_image

    def _encode_vae_image(image, generator):
        vae_dtype = getattr(pipe.vae, "dtype", None)
        if vae_dtype is None:
            vae_dtype = next(pipe.vae.parameters()).dtype
        out = original(image.to(vae_dtype), generator)
        return out.to(image.dtype)

    pipe._encode_vae_image = _encode_vae_image


class Engine:
    def __init__(self, args, say):
        self.args = args
        self.say = say  # say(text): progress text for the UI
        self.pipe = None
        self.torch = None
        self.gpu = 0
        self.summary = ""
        self.note = ""  # a warning worth showing after load, e.g. low free VRAM

    @property
    def ready(self) -> bool:
        return self.pipe is not None

    def load(self) -> str:
        a = self.args
        self.say("Importing PyTorch…")
        import torch

        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this Python environment.")
        count = torch.cuda.device_count()
        if str(a.gpu) == "auto":
            gpu = max(range(count),
                      key=lambda i: torch.cuda.get_device_properties(i).total_memory)
        else:
            gpu = int(a.gpu)
            if not 0 <= gpu < count:
                raise RuntimeError(f"GPU {gpu} not found (PyTorch sees {count}).")
        self.gpu = gpu
        torch.cuda.set_device(gpu)
        props = torch.cuda.get_device_properties(gpu)
        legacy = props.major < 8  # no flash attention / bf16 cuDNN convs
        compat = a.compat == "on" or (a.compat == "auto" and legacy)

        free, _total = torch.cuda.mem_get_info(gpu)
        free_gib = free / 2**30
        note = ""
        if a.offload == "model" and free_gib < 19:
            note = (f"  Only {free_gib:.1f} GiB free on that GPU; the model used about "
                    "19 GiB in testing. Stop other GPU jobs (llama-server, Ollama) first.")

        if compat:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)

        self.say("Importing diffusers…")
        from diffusers import QwenImage21Pipeline

        dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float16
        self.say(f"Loading {a.model} …" + note)
        pipe = QwenImage21Pipeline.from_pretrained(a.model, dtype=dtype)
        if compat:
            pipe.vae.to(torch.float32)
            patch_vae_encode(pipe)
        pipe.set_progress_bar_config(disable=True)

        self.say(f"Placing model on GPU {gpu} ({a.offload} offload) …")
        if a.offload == "model":
            pipe.enable_model_cpu_offload(gpu_id=gpu)
        elif a.offload == "sequential":
            pipe.enable_sequential_cpu_offload(gpu_id=gpu)
        else:
            pipe.to(f"cuda:{gpu}")
        self.pipe = pipe

        self.summary = (f"{props.name} (GPU {gpu}) · {a.offload} offload · {a.dtype}"
                        + (" · V100-class fixes on" if compat else ""))
        self.note = note.strip()
        return self.summary

    def unload(self) -> None:
        self.pipe = None
        gc.collect()
        if self.torch is not None:
            self.torch.cuda.empty_cache()

    def generate(self, p: dict, images: list, on_step, cancel: threading.Event):
        """One image. on_step(step, total, seconds_per_step) after each step."""
        torch = self.torch
        pipe = self.pipe
        kw = dict(
            prompt=p["prompt"],
            use_kv_cache=p["kv_cache"],
            output_resolution=p["output_resolution"],
            generator=torch.Generator("cpu").manual_seed(p["seed"]),
        )
        if not (images and p["match_input"]):
            kw["width"], kw["height"] = p["width"], p["height"]
        if images:
            kw["image"] = images if len(images) > 1 else images[0]
        if p["cfg"] > 1.0:
            kw["true_cfg_scale"] = p["cfg"]
            kw["negative_prompt"] = p["negative"] or " "

        last = [time.time()]

        def _on_step_end(*cb_args):
            # Called as (pipe, step, timestep, kwargs) or (step, timestep, kwargs).
            kwargs, step = cb_args[-1], cb_args[-3]
            now = time.time()
            per, last[0] = now - last[0], now
            try:
                total = len(pipe.scheduler.timesteps)
            except Exception:  # noqa: BLE001
                total = 8
            on_step(step + 1, total, per)
            if cancel.is_set():
                raise Cancelled()
            return kwargs

        kw["callback_on_step_end"] = _on_step_end
        try:
            return pipe(**kw).images[0]
        except Cancelled:
            raise
        except Exception:
            gc.collect()
            torch.cuda.empty_cache()
            raise


def save_png(img: Image.Image, out_dir: Path, p: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{time.strftime('%Y%m%d-%H%M%S')}_{p['seed']}"
    path = out_dir / f"{stem}.png"
    n = 1
    while path.exists():
        path = out_dir / f"{stem}_{n}.png"
        n += 1
    meta = PngImagePlugin.PngInfo()
    meta.add_text("prompt", p["prompt"])
    meta.add_text("parameters", json.dumps(p, default=str))
    img.save(path, pnginfo=meta)
    return path


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

class StudioApp:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.out_dir = Path(args.out).expanduser()
        self.jobs: queue.Queue = queue.Queue()
        self.ui_q: queue.Queue = queue.Queue()
        self.cancel = threading.Event()
        self.engine = Engine(args, lambda t: (log(t), self.ui_q.put(("log", t))))
        root.report_callback_exception = self._callback_error
        self.engine_state = "unloaded"  # unloaded | loading | ready | error
        self.busy = False
        self.edit_paths: list[str] = []
        self.items: list[dict] = []
        self.current: Image.Image | None = None
        self._photo = None
        self._thumbs: list = []
        self._build()
        threading.Thread(target=self._worker, daemon=True).start()
        self.root.after(100, self._drain)
        if not args.no_autoload:
            self._request_load()

    # -- layout -------------------------------------------------------------

    def _build(self):
        root = self.root
        root.title("Qwen Image Studio")
        root.configure(bg=BG)
        root.geometry("1220x820")
        root.minsize(1000, 660)

        # Window / taskbar icon: icon.png next to this script, if present.
        icon_file = APP_DIR / "icon.png"
        if icon_file.is_file():
            try:
                src = Image.open(icon_file).convert("RGBA")
                self._icons = [ImageTk.PhotoImage(src.resize((s, s), Image.LANCZOS))
                               for s in (256, 64, 32)]
                root.iconphoto(True, *self._icons)
            except Exception as exc:  # noqa: BLE001
                log(f"couldn't set the window icon: {exc}")
        else:
            log(f"no window icon: {icon_file} not found")

        style = ttk.Style()
        style.theme_use("clam")
        for name in ("TCombobox", "TSpinbox"):
            style.configure(name, fieldbackground=CARD, background=CARD,
                            foreground=TEXT, arrowcolor=TEXT, bordercolor=TRACK,
                            lightcolor=CARD, darkcolor=CARD, insertcolor=TEXT)
        style.map("TCombobox", fieldbackground=[("readonly", CARD)],
                  foreground=[("readonly", TEXT)])
        style.configure("Horizontal.TProgressbar", troughcolor=TRACK,
                        background=ACCENT, bordercolor=BG, lightcolor=ACCENT,
                        darkcolor=ACCENT)
        root.option_add("*TCombobox*Listbox.background", CARD)
        root.option_add("*TCombobox*Listbox.foreground", TEXT)
        root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)

        # Top bar: engine status ------------------------------------------
        top = tk.Frame(root, bg=BG)
        top.pack(fill="x", padx=14, pady=(10, 6))
        tk.Label(top, text="QWEN IMAGE STUDIO", bg=BG, fg=SUB,
                 font=("Helvetica", 8, "bold")).pack(side="left")
        self.engine_btn = tk.Button(top, text="Load engine", command=self._toggle_engine,
                                    bg=CARD, fg=TEXT, activebackground=TRACK,
                                    activeforeground=TEXT, relief="flat", bd=0,
                                    padx=10, pady=2, font=("Helvetica", 9),
                                    cursor="hand2", highlightthickness=0)
        self.engine_btn.pack(side="right")
        self.engine_lbl = tk.Label(top, text="Engine: not loaded", bg=BG, fg=SUB,
                                   font=("Helvetica", 9), anchor="e")
        self.engine_lbl.pack(side="right", padx=10)

        body = tk.Frame(root, bg=BG)
        body.pack(fill="both", expand=True, padx=14, pady=(0, 10))

        left = tk.Frame(body, bg=BG, width=380)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(14, 0))

        self._build_controls(left)
        self._build_output(right)

    def _label(self, parent, text, **kw):
        return tk.Label(parent, text=text, bg=kw.pop("bg", CARD), fg=kw.pop("fg", SUB),
                        font=kw.pop("font", ("Helvetica", 9)), anchor="w", **kw)

    def _build_controls(self, left):
        tk.Label(left, text="Prompt", bg=BG, fg=SUB, anchor="w",
                 font=("Helvetica", 8, "bold")).pack(fill="x")
        self.prompt = tk.Text(left, height=8, wrap="word", bg=CARD, fg=TEXT,
                              insertbackground=TEXT, relief="flat", padx=8, pady=6,
                              font=("Helvetica", 10), highlightthickness=1,
                              highlightbackground=TRACK, highlightcolor=ACCENT)
        self.prompt.pack(fill="x", pady=(2, 8))
        self.prompt.bind("<Control-Return>", lambda e: (self._generate(), "break")[1])
        self.prompt.focus_set()

        btns = tk.Frame(left, bg=BG)
        btns.pack(fill="x", pady=(0, 10))
        self.gen_btn = tk.Button(btns, text="Generate  (Ctrl+Enter)", command=self._generate,
                                 bg=ACCENT, fg="#0b0d12", activebackground=ACCENT,
                                 relief="flat", bd=0, padx=14, pady=6,
                                 font=("Helvetica", 10, "bold"), cursor="hand2",
                                 highlightthickness=0, disabledforeground="#5b6577")
        self.gen_btn.pack(side="left", fill="x", expand=True)
        self.cancel_btn = tk.Button(btns, text="Cancel", command=self._cancel,
                                    bg=CARD, fg=TEXT, activebackground=TRACK,
                                    relief="flat", bd=0, padx=14, pady=6,
                                    font=("Helvetica", 10), cursor="hand2",
                                    highlightthickness=0, disabledforeground="#5b6577")
        self.cancel_btn.pack(side="left", padx=(8, 0))

        card = tk.Frame(left, bg=CARD, highlightbackground="#262c3a",
                        highlightthickness=1)
        card.pack(fill="x")
        card.columnconfigure(1, weight=1)
        r = 0

        def add(label, widget_builder, hint=None):
            nonlocal r
            self._label(card, label).grid(row=r, column=0, sticky="w", padx=(10, 6), pady=3)
            w = widget_builder(card)
            w.grid(row=r, column=1, sticky="ew", padx=(0, 10), pady=3)
            if hint:
                r += 1
                self._label(card, hint, font=("Helvetica", 8), wraplength=250,
                            justify="left").grid(
                    row=r, column=1, sticky="w", padx=(0, 10), pady=(0, 2))
            r += 1

        # Size
        self.preset_var = tk.StringVar(value=PRESETS[0][0])
        self.w_var = tk.StringVar(value=str(PRESETS[0][1]))
        self.h_var = tk.StringVar(value=str(PRESETS[0][2]))

        def size_box(parent):
            f = tk.Frame(parent, bg=CARD)
            cb = ttk.Combobox(f, textvariable=self.preset_var, state="readonly",
                              values=[p[0] for p in PRESETS] + ["Custom"], width=26)
            cb.pack(fill="x")
            cb.bind("<<ComboboxSelected>>", self._preset_chosen)
            row = tk.Frame(f, bg=CARD)
            row.pack(fill="x", pady=(4, 0))
            ttk.Spinbox(row, from_=256, to=4096, increment=STEP, width=6,
                        textvariable=self.w_var, command=self._custom).pack(side="left")
            tk.Label(row, text=" × ", bg=CARD, fg=SUB).pack(side="left")
            ttk.Spinbox(row, from_=256, to=4096, increment=STEP, width=6,
                        textvariable=self.h_var, command=self._custom).pack(side="left")
            tk.Button(row, text="swap", command=self._swap, bg=TRACK, fg=TEXT, relief="flat",
                      bd=0, padx=8, cursor="hand2", highlightthickness=0,
                      activebackground=TRACK, activeforeground=TEXT).pack(side="left", padx=8)
            return f

        add("Size", size_box, "Rounded down to a multiple of 32. The big sizes are slow on an older GPU.")

        # Seed
        self.seed_var = tk.StringVar(value="")
        self.rand_var = tk.BooleanVar(value=True)

        def seed_box(parent):
            f = tk.Frame(parent, bg=CARD)
            tk.Entry(f, textvariable=self.seed_var, width=14, bg=TRACK, fg=TEXT,
                     insertbackground=TEXT, relief="flat").pack(side="left")
            tk.Checkbutton(f, text="random", variable=self.rand_var, bg=CARD, fg=TEXT,
                           selectcolor=TRACK, activebackground=CARD,
                           activeforeground=TEXT, highlightthickness=0).pack(side="left", padx=8)
            return f

        add("Seed", seed_box)

        # Batch
        self.batch_var = tk.StringVar(value="1")
        add("Images", lambda p: ttk.Spinbox(p, from_=1, to=16, width=5,
                                            textvariable=self.batch_var),
            "More than one uses consecutive seeds, one after another.")

        # KV cache
        self.kv_var = tk.BooleanVar(value=True)
        add("KV cache", lambda p: tk.Checkbutton(
            p, text="reuse text / reference context across steps", variable=self.kv_var,
            bg=CARD, fg=TEXT, selectcolor=TRACK, activebackground=CARD,
            activeforeground=TEXT, highlightthickness=0, anchor="w"))

        # CFG + negative
        self.cfg_var = tk.StringVar(value="1.0")
        self.neg_var = tk.StringVar(value="")
        add("Guidance", lambda p: ttk.Spinbox(p, from_=1.0, to=10.0, increment=0.5,
                                              width=5, textvariable=self.cfg_var),
            "Turbo is tuned for 1.0. Above 1.0 needs the negative prompt and takes about twice as long.")
        add("Negative", lambda p: tk.Entry(p, textvariable=self.neg_var, bg=TRACK, fg=TEXT,
                                          insertbackground=TEXT, relief="flat"))

        # Steps (read-only)
        add("Steps", lambda p: self._label(p, "fixed by the Turbo schedule (8)", fg=TEXT))

        # Edit images
        self.edit_lbl = None
        self.match_var = tk.BooleanVar(value=True)
        self.area_var = tk.StringVar(value="1024")

        def edit_box(parent):
            f = tk.Frame(parent, bg=CARD)
            row = tk.Frame(f, bg=CARD)
            row.pack(fill="x")
            tk.Button(row, text="Add images…", command=self._add_images, bg=TRACK, fg=TEXT,
                      relief="flat", bd=0, padx=8, cursor="hand2", highlightthickness=0,
                      activebackground=TRACK, activeforeground=TEXT).pack(side="left")
            tk.Button(row, text="Clear", command=self._clear_images, bg=TRACK, fg=TEXT,
                      relief="flat", bd=0, padx=8, cursor="hand2", highlightthickness=0,
                      activebackground=TRACK, activeforeground=TEXT).pack(side="left", padx=6)
            self.edit_lbl = tk.Label(f, text="none (text to image)", bg=CARD, fg=SUB,
                                     anchor="w", font=("Helvetica", 9))
            self.edit_lbl.pack(fill="x", pady=(3, 0))
            tk.Checkbutton(f, text="match first input's shape", variable=self.match_var,
                           bg=CARD, fg=TEXT, selectcolor=TRACK, activebackground=CARD,
                           activeforeground=TEXT, highlightthickness=0,
                           anchor="w").pack(fill="x")
            row2 = tk.Frame(f, bg=CARD)
            row2.pack(fill="x")
            tk.Label(row2, text="work area", bg=CARD, fg=SUB,
                     font=("Helvetica", 9)).pack(side="left")
            ttk.Spinbox(row2, from_=512, to=2048, increment=64, width=6,
                        textvariable=self.area_var).pack(side="left", padx=6)
            tk.Label(row2, text="px per side (editing only)", bg=CARD, fg=SUB,
                     font=("Helvetica", 8)).pack(side="left")
            return f

        add("Edit", edit_box)

    def _build_output(self, right):
        self.canvas = tk.Canvas(right, bg=CARD, highlightthickness=1,
                                highlightbackground="#262c3a")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._render_preview())
        self.canvas.create_text(10, 10, anchor="nw", fill=SUB, font=("Helvetica", 10),
                                text="Your image will appear here.", tags="hint")

        self.progress = ttk.Progressbar(right, mode="determinate", maximum=1, value=0)
        self.progress.pack(fill="x", pady=(8, 0))
        self.status = tk.Label(right, text="Starting…", bg=BG, fg=SUB, anchor="w",
                               font=("Helvetica", 9), justify="left", wraplength=760)
        self.status.pack(fill="x", pady=(4, 6))

        bar = tk.Frame(right, bg=BG)
        bar.pack(fill="x", pady=(0, 6))
        for text, cmd in (("Open folder", self._open_folder),
                          ("Save a copy…", self._save_copy),
                          ("Reuse these settings", self._reuse)):
            tk.Button(bar, text=text, command=cmd, bg=CARD, fg=TEXT, relief="flat", bd=0,
                      padx=10, pady=3, cursor="hand2", highlightthickness=0,
                      activebackground=TRACK, activeforeground=TEXT,
                      font=("Helvetica", 9)).pack(side="left", padx=(0, 6))
        self.path_lbl = tk.Label(bar, text="", bg=BG, fg=SUB, anchor="w",
                                 font=("Helvetica", 8))
        self.path_lbl.pack(side="left", fill="x", expand=True, padx=6)

        gal = tk.Frame(right, bg=BG, height=118)
        gal.pack(fill="x")
        gal.pack_propagate(False)
        self.gal_canvas = tk.Canvas(gal, bg=BG, highlightthickness=0, height=100)
        sb = ttk.Scrollbar(gal, orient="horizontal", command=self.gal_canvas.xview)
        self.gal_canvas.configure(xscrollcommand=sb.set)
        self.gal_canvas.pack(fill="x")
        sb.pack(fill="x")
        self.gal_inner = tk.Frame(self.gal_canvas, bg=BG)
        self.gal_canvas.create_window((0, 0), window=self.gal_inner, anchor="nw")
        self._sync_buttons()

    # -- small handlers -----------------------------------------------------

    def _preset_chosen(self, _e=None):
        for label, w, h in PRESETS:
            if label == self.preset_var.get():
                self.w_var.set(str(w))
                self.h_var.set(str(h))

    def _custom(self):
        self.preset_var.set("Custom")

    def _swap(self):
        w, h = self.w_var.get(), self.h_var.get()
        self.w_var.set(h)
        self.h_var.set(w)
        self._custom()

    def _add_images(self):
        paths = filedialog.askopenfilenames(
            title="Images to edit",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")])
        if paths:
            self.edit_paths.extend(paths)
            self._refresh_edit_label()

    def _clear_images(self):
        self.edit_paths = []
        self._refresh_edit_label()

    def _refresh_edit_label(self):
        n = len(self.edit_paths)
        self.edit_lbl.config(
            text="none (text to image)" if not n else
            f"{n} image{'s' if n != 1 else ''}: " + ", ".join(Path(p).name for p in self.edit_paths)[:60],
            fg=SUB if not n else ACCENT)

    def _set_status(self, text, color=SUB):
        self.status.config(text=text, fg=color)

    def _sync_buttons(self):
        ready = self.engine_state == "ready"
        self.gen_btn.config(state="normal" if ready and not self.busy else "disabled")
        self.cancel_btn.config(state="normal" if self.busy else "disabled")
        if self.engine_state == "loading":
            self.engine_btn.config(text="Loading…", state="disabled")
        elif ready:
            self.engine_btn.config(text="Unload engine",
                                   state="disabled" if self.busy else "normal")
        else:
            self.engine_btn.config(text="Load engine", state="normal")

    # -- engine control -----------------------------------------------------

    def _toggle_engine(self):
        if self.engine_state == "ready":
            self.jobs.put({"kind": "unload"})
            self.engine_state = "loading"
            self.engine_lbl.config(text="Engine: unloading…", fg=WARN)
            self._sync_buttons()
        elif self.engine_state in ("unloaded", "error"):
            self._request_load()

    def _request_load(self):
        self.engine_state = "loading"
        self.engine_lbl.config(text="Engine: loading…", fg=WARN)
        self._set_status("Starting the engine. The first load can take a minute.", WARN)
        self.progress.config(mode="indeterminate")
        self.progress.start(14)
        self._sync_buttons()
        self.jobs.put({"kind": "load"})

    # -- generating ---------------------------------------------------------

    def _generate(self):
        if self.engine_state != "ready" or self.busy:
            return
        prompt = self.prompt.get("1.0", "end").strip()
        if not prompt:
            self._set_status("Type a prompt first.", WARN)
            return
        try:
            w = max(256, int(float(self.w_var.get())) // STEP * STEP)
            h = max(256, int(float(self.h_var.get())) // STEP * STEP)
            n = max(1, min(16, int(float(self.batch_var.get()))))
            cfg = max(1.0, float(self.cfg_var.get()))
            area = max(512, int(float(self.area_var.get())))
        except ValueError:
            self._set_status("One of the numbers isn't valid.", BAD)
            return
        self.w_var.set(str(w))
        self.h_var.set(str(h))

        seed_txt = self.seed_var.get().strip()
        if self.rand_var.get() or not seed_txt:
            base = random.randrange(2**32 - 16)
        else:
            try:
                base = int(seed_txt)
            except ValueError:
                self._set_status("The seed must be a whole number.", BAD)
                return
        self.seed_var.set(str(base))

        images = []
        try:
            for path in self.edit_paths:
                images.append(Image.open(path).convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Couldn't open an input image: {exc}", BAD)
            return

        params = dict(prompt=prompt, width=w, height=h, cfg=cfg,
                      negative=self.neg_var.get().strip(), kv_cache=self.kv_var.get(),
                      output_resolution=area, match_input=self.match_var.get(),
                      edit_images=[Path(p).name for p in self.edit_paths])
        self.cancel.clear()
        self.busy = True
        self._sync_buttons()
        self.progress.config(mode="determinate", maximum=1, value=0)
        self._set_status("Encoding prompt…", ACCENT)
        self.jobs.put({"kind": "generate", "params": params, "images": images,
                       "seeds": [base + i for i in range(n)]})

    def _cancel(self):
        if self.busy:
            self.cancel.set()
            self._set_status("Cancelling after the current step…", WARN)

    # -- worker thread ------------------------------------------------------

    def _worker(self):
        while True:
            job = self.jobs.get()
            kind = job["kind"]
            try:
                if kind == "load":
                    summary = self.engine.load()
                    self.ui_q.put(("engine", "ready", summary))
                elif kind == "unload":
                    self.engine.unload()
                    self.ui_q.put(("engine", "unloaded", "not loaded (GPU memory released)"))
                elif kind == "generate":
                    self._run_generate(job)
            except Exception as exc:  # noqa: BLE001
                log(f"{kind} failed:\n{traceback.format_exc()}")
                if kind == "load":
                    self.ui_q.put(("engine", "error", f"{type(exc).__name__}: {exc}"))
                else:
                    self.ui_q.put(("error", self._explain(exc)))
                    self.ui_q.put(("done",))

    def _callback_error(self, exc_type, exc, tb):
        log("UI error:\n" + "".join(traceback.format_exception(exc_type, exc, tb)))
        self._set_status(f"Internal error: {exc_type.__name__}: {exc} (see the log)", BAD)

    @staticmethod
    def _explain(exc: Exception) -> str:
        text = f"{type(exc).__name__}: {exc}"
        if type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower():
            return ("Out of GPU memory. Try a smaller size, close other GPU jobs, or "
                    "restart with --offload sequential. (" + text.splitlines()[0][:120] + ")")
        return text

    def _run_generate(self, job):
        n = len(job["seeds"])
        for i, seed in enumerate(job["seeds"]):
            if self.cancel.is_set():
                break
            p = dict(job["params"], seed=seed)
            t0 = time.time()
            self.ui_q.put(("progress", i, n, 0, 0, None))

            def on_step(step, total, per, i=i):
                self.ui_q.put(("progress", i, n, step, total, per))

            try:
                img = self.engine.generate(p, job["images"], on_step, self.cancel)
            except Cancelled:
                break
            path = save_png(img, self.out_dir, p)
            self.ui_q.put(("image", img, path, p, time.time() - t0))
        self.ui_q.put(("done",))

    # -- UI queue -----------------------------------------------------------

    def _drain(self):
        try:
            while True:
                self._handle(self.ui_q.get_nowait())
        except queue.Empty:
            pass
        self.root.after(100, self._drain)

    def _handle(self, msg):
        kind = msg[0]
        if kind == "log":
            if self.engine_state == "loading":
                self._set_status(msg[1], WARN)
        elif kind == "engine":
            _, state, text = msg
            self.engine_state = state
            self.progress.stop()
            self.progress.config(mode="determinate", maximum=1, value=0)
            colors = {"ready": GOOD, "error": BAD, "unloaded": SUB}
            self.engine_lbl.config(text=f"Engine: {text}", fg=colors.get(state, SUB))
            log(f"engine {state}: {text}")
            if state == "ready":
                note = self.engine.note
                self._set_status("Engine ready. Write a prompt and press Generate."
                                 + (f"  {note}" if note else ""), WARN if note else GOOD)
            elif state == "error":
                self._set_status("The engine failed to start. See the line at the top "
                                 "and the terminal for details.", BAD)
            self._sync_buttons()
        elif kind == "progress":
            _, i, n, step, total, per = msg
            head = f"Image {i + 1}/{n}  ·  " if n > 1 else ""
            if total:
                self.progress.config(maximum=total, value=step)
                if step >= total:
                    self._set_status(head + "decoding image…", ACCENT)
                else:
                    self._set_status(head + f"step {step}/{total}  ·  {per:.1f} s/step", ACCENT)
            else:
                self.progress.config(maximum=1, value=0)
                self._set_status(head + "encoding prompt…", ACCENT)
        elif kind == "image":
            _, img, path, p, secs = msg
            self.items.append({"img": img, "path": path, "params": p})
            self._add_thumb(len(self.items) - 1)
            self._show_item(len(self.items) - 1)
            self._set_status(f"Done in {secs:.0f} s  ·  seed {p['seed']}  ·  "
                             f"{img.size[0]}×{img.size[1]}", GOOD)
        elif kind == "error":
            self._set_status(msg[1], BAD)
        elif kind == "done":
            self.busy = False
            self.progress.config(value=0)
            if self.cancel.is_set():
                self._set_status("Cancelled.", WARN)
            self._sync_buttons()

    # -- preview / gallery --------------------------------------------------

    def _render_preview(self):
        if self.current is None:
            return
        self.canvas.delete("all")
        cw, ch = max(self.canvas.winfo_width(), 60), max(self.canvas.winfo_height(), 60)
        iw, ih = self.current.size
        s = min((cw - 8) / iw, (ch - 8) / ih, 1.0)
        disp = self.current.resize((max(1, int(iw * s)), max(1, int(ih * s))), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(disp)
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo)

    def _show_item(self, idx):
        self.sel = idx
        self.current = self.items[idx]["img"]
        self.path_lbl.config(text=str(self.items[idx]["path"]))
        self._render_preview()

    def _add_thumb(self, idx):
        thumb = self.items[idx]["img"].copy()
        thumb.thumbnail((96, 96))
        photo = ImageTk.PhotoImage(thumb)
        self._thumbs.append(photo)
        lbl = tk.Label(self.gal_inner, image=photo, bg=BG, cursor="hand2", bd=2,
                       relief="flat")
        lbl.pack(side="left", padx=3)
        lbl.bind("<Button-1>", lambda e, i=idx: self._show_item(i))
        self.gal_inner.update_idletasks()
        self.gal_canvas.configure(scrollregion=self.gal_canvas.bbox("all"))
        self.gal_canvas.xview_moveto(1.0)

    def _selected(self):
        return self.items[self.sel] if self.items and hasattr(self, "sel") else None

    def _open_folder(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.Popen(["xdg-open", str(self.out_dir)])
        except OSError as exc:
            self._set_status(f"Couldn't open the folder: {exc}", BAD)

    def _save_copy(self):
        item = self._selected()
        if not item:
            return
        dest = filedialog.asksaveasfilename(defaultextension=".png",
                                            initialfile=Path(item["path"]).name)
        if dest:
            item["img"].save(dest)
            self._set_status(f"Saved a copy to {dest}", GOOD)

    def _reuse(self):
        item = self._selected()
        if not item:
            return
        p = item["params"]
        self.prompt.delete("1.0", "end")
        self.prompt.insert("1.0", p["prompt"])
        self.w_var.set(str(p["width"]))
        self.h_var.set(str(p["height"]))
        self._custom()
        self.seed_var.set(str(p["seed"]))
        self.rand_var.set(False)
        self.cfg_var.set(str(p["cfg"]))
        self.neg_var.set(p["negative"])
        self.kv_var.set(p["kv_cache"])
        self.area_var.set(str(p["output_resolution"]))
        self.match_var.set(p["match_input"])
        self._set_status("Settings restored. The seed is locked so you can vary one thing.", SUB)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gpu", default="auto", help="auto, or a GPU index as PyTorch numbers them")
    ap.add_argument("--offload", choices=("model", "sequential", "none"), default="model")
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--compat", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--no-autoload", action="store_true")
    args = ap.parse_args(argv)

    global LOG_PATH
    out = Path(args.out).expanduser()
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    for folder in (APP_DIR, out):  # log beside the script, else beside the images
        try:
            (folder / "studio.log").touch()
            LOG_PATH = folder / "studio.log"
            break
        except OSError:
            continue
    log(f"Qwen Image Studio starting · python {sys.version.split()[0]} · {sys.executable}")
    log(f"log file: {LOG_PATH} · HF_HOME={os.environ.get('HF_HOME', '(unset)')}")
    if tk is None:
        log(f"Tkinter/Pillow-Tk import failed: {_TK_ERROR!r}  "
            "(Linux Mint: sudo apt install python3-tk)")
        return 2
    sys.excepthook = lambda t, v, tb: log(
        "uncaught:\n" + "".join(traceback.format_exception(t, v, tb)))
    log(f"Tk {tk.TkVersion} · opening window")
    root = tk.Tk()
    StudioApp(root, args)
    log("window ready, engine loading in the background"
        if not args.no_autoload else "window ready (autoload off)")
    root.mainloop()
    log("window closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
