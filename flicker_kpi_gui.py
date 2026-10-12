"""
Flicker KPI GUI
===============
使用方式:
    python flicker_kpi_gui.py
需求: numpy, opencv-python, pillow  (tkinter 為 Python 內建)

功能:
  - 選擇資料夾：自動把 <prefix>0001 ~ <prefix>0006 分組成多組序列，左上清單逐組檢視
  - 「批次分析+匯出全部」：每組輸出 <name>_risk_map.png，另輸出全部序列的 CSV 與 summary
  - 20 個 threshold (pixel-based 10 + block-based 10) 皆可即時調整
  - 即時重算超標比例，metric 表格以顏色標示 PASS / FAIL
  - 右側 heatmap: 風險分級總圖 / ratio (value/threshold) / raw / binary fail mask / 綜合超標張數
  - 風險分級總圖: 20 項 metric 取 max(value/threshold)，以 5 色疊在原圖上
  - 設定存取: metric 開關 / threshold / K / 風險分界 / 分格參數 / difference 定義 / 檔名 pattern
              可存成 JSON；開啟時自動載入程式同目錄的 flicker_kpi_settings.json，
              關閉視窗時自動存回同一個檔案
  - Y 定義可選: BT.601 / (R+2G+B)/4 / R+2G+B
  - 每項 metric 可各自開關；停用者不列入 PASS/FAIL、風險圖、分格與最佳化
  - FAIL 判定: 任一 metric 超標位置數 >= K 即 FAIL (K=1 即「任一 pixel 超標」)
  - Threshold 最佳化: 用「不閃 folder」+「混合 folder（逐組標記閃/不閃）」求一組 threshold 與 K，
                      使漏抓 = 0 且誤報最少，可一鍵套用回主視窗
  - 人工框 (ROI): 在閃序列上框出「這裡必須抓到」的位置，存成資料夾的 flicker_rois.csv；
                  最佳化時當次要目標（不影響誤報數，只在同分時盡量多抓到框）
  - 右側圖可手動縮放: 符合視窗 / 25%~800%；Ctrl+滾輪縮放、滾輪捲動、右鍵拖曳平移
  - 分格總圖: 以 tile x tile (預設 40x40) 彙整超標點，RED/ORANGE/YELLOW 外框標示，
              點格子可看該格明細（最大 ratio、位置、pixel/block 超標數、主要 metric）
  - 匯出 CSV 報表 + 單張風險分級總圖 (risk_map.png)

效能備註: metric map 只在載入序列或切換 difference mode 時重算一次；
         調 threshold 只做比較運算，不重算 metric，所以拖 slider 不會卡。
"""

import os
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import cv2
from PIL import Image, ImageTk

from flicker_kpi_core import (
    DEFAULT_THRESHOLDS, METRIC_NAMES,
    analyze_sequence, evaluate_kpi, combined_fail_map,
    render_heatmap, upscale,
    DEFAULT_RISK_BOUNDS, RISK_LEVELS, compute_risk, render_risk_map,
    parse_risk_bounds, risk_range_labels, imwrite_unicode,
    discover_sequences, summarize_result, max_pool_fit,
    DEFAULT_TILE, DEFAULT_TILE_RULES, compute_tiles, tile_detail, render_tile_map,
    parse_tile_size, parse_tile_rule, tile_rule_text,
    metric_is_fail, k_for_level,
    default_settings_path, save_settings, load_settings,
    Y_MODES, DEFAULT_Y_MODE, all_enabled, is_enabled, LEGEND_H,
    load_rois, save_rois, clip_box, ROI_FILE, box_metric_stats,
)
import flicker_kpi_opt as kopt

N_FRAMES = 6
CSV_HEADER = ["sequence", "level", "metric", "threshold", "fail_ratio_%",
              "fail_count", "total", "max", "p99.9", "p99", "mean",
              "max/threshold", "verdict"]
SUMMARY_HEADER = ["sequence", "verdict", "pixel_fail_metrics", "block_fail_metrics",
                  "FAIL+SEVERE_area_%", "worst_level", "max_ratio", "max_metric",
                  "max_x", "max_y", "tile_size", "tiles_red", "tiles_orange",
                  "tiles_yellow", "tiles_fail", "tile_fail_%"]
SEQ_COLS = ("sequence", "verdict", "pix_fail", "blk_fail", "danger%", "worst", "max_ratio",
            "tiles R/O/Y")


def metric_rows(result, min_fail=1, enabled=None):
    rows = []
    bs = result.get("block_size", 2)
    for level in ("pixel", "block"):
        rep = result[level]["report"]
        for n in METRIC_NAMES:
            s = rep[n]
            rows.append([result["name"], level, n, s["threshold"],
                         f"{s['fail_ratio']*100:.6f}", s["fail_count"], s["total"],
                         f"{s['max']:.3f}", f"{s['p999']:.3f}", f"{s['p99']:.3f}",
                         f"{s['mean']:.4f}", f"{s['margin']:.3f}",
                         "OFF" if not is_enabled(enabled, level, n) else
                         ("FAIL" if metric_is_fail(s, level, min_fail, bs) else "PASS")])
    return rows


def summary_row(sm):
    return [sm["name"], sm["verdict"], sm["pixel_fail_metrics"], sm["block_fail_metrics"],
            f"{sm['danger_pct']:.4f}", sm["worst_level"], f"{sm['max_ratio']:.3f}",
            sm["max_metric"], sm["max_xy"][0], sm["max_xy"][1],
            sm["tile_size"], sm["tiles_red"], sm["tiles_orange"], sm["tiles_yellow"],
            sm["tiles_fail"], f"{sm['tile_fail_pct']:.4f}"]


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(rows)


def enable_dpi_awareness():
    """Windows 高 DPI：讓程式拿到實際像素，避免系統把視窗模糊放大"""
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)     # Win 8.1+
        except AttributeError:
            ctypes.windll.user32.SetProcessDPIAware()          # Win 7
    except Exception:
        pass


def ui_scale(root):
    """回傳相對 96 dpi 的縮放倍率，並讓 Tk 字型跟著 dpi 走"""
    try:
        dpi = float(root.winfo_fpixels("1i"))
    except Exception:
        dpi = 96.0
    scale = max(0.75, min(dpi / 96.0, 3.0))
    try:
        root.tk.call("tk", "scaling", dpi / 72.0)
        ttk.Style().configure("Treeview", rowheight=int(20 * scale))
    except Exception:
        pass
    return scale


def fit_window(win, want_w, want_h, scale=1.0, center=True):
    """視窗不超出螢幕；小螢幕自動縮小"""
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    w = min(int(want_w * scale), sw - 60)
    h = min(int(want_h * scale), sh - 80)
    if center:
        win.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 2 - 20)}")
    else:
        win.geometry(f"{w}x{h}")
    win.minsize(min(int(820 * scale), sw - 40), min(int(520 * scale), sh - 40))
    return w, h


def make_wrapping(frame, gap=6, pady=3):
    """
    控制列自動換行：視窗變窄時，原本橫向排列的元件會往下折，不會被切掉或擠出畫面。
    """
    state = {"w": -1}

    def _pad():
        try:                                   # ttk.Frame 的 padding 會縮減可用寬度
            v = frame.cget("padding")
            v = v[0] if isinstance(v, (list, tuple)) and v else v
            return int(str(v) or 0)
        except (ValueError, TypeError, tk.TclError):
            return 0

    def layout(_=None):
        width = max(frame.winfo_width() - 2 * _pad() - 2, 50)
        if width == state["w"]:
            return
        state["w"] = width
        x = y = rowh = 0
        for w in frame.winfo_children():
            rw, rh = w.winfo_reqwidth(), w.winfo_reqheight()
            if x > 0 and x + rw > width:
                x = 0
                y += rowh + pady
                rowh = 0
            w.place(x=x, y=y, width=rw, height=rh)
            x += rw + gap
            rowh = max(rowh, rh)
        frame.configure(height=y + rowh + 2)

    for w in frame.winfo_children():
        if w.winfo_manager() == "pack":
            w.pack_forget()
    frame.pack_propagate(False)
    frame.bind("<Configure>", layout)
    frame.after(0, layout)
    return layout


def _sort_key(v):
    """表格排序用：數字（含 %、x、a/b/c 形式）依數值，其餘依字串；空白與 — 排最後"""
    t = str(v).strip()
    if t in ("", "-", "—"):
        return (2, 0.0, "")
    core = t.rstrip("%x").replace(",", "")
    try:
        return (0, float(core), "")
    except ValueError:
        pass
    if "/" in t:                                   # 例如 tiles R/O/Y 的 "12/3/0"
        parts = t.split("/")
        try:
            nums = [float(x) for x in parts]
            return (0, sum(n * 1000 ** (len(nums) - 1 - i) for i, n in enumerate(nums)), "")
        except ValueError:
            pass
    return (1, 0.0, t.lower())


def make_sortable(tree):
    """點欄位標題即依該欄排序，再點一次反向；回傳可重新套用排序的函式"""
    state = {"col": None, "rev": False}
    cols = tree["columns"]

    def apply_sort():
        col = state["col"]
        if col is None:
            return
        items = [(_sort_key(tree.set(i, col)), i) for i in tree.get_children("")]
        items.sort(key=lambda p: p[0], reverse=state["rev"])
        for pos, (_, iid) in enumerate(items):
            tree.move(iid, "", pos)

    def on_click(col):
        state["rev"] = not state["rev"] if state["col"] == col else False
        state["col"] = col
        for c in cols:
            arrow = ("  ▼" if state["rev"] else "  ▲") if c == col else ""
            tree.heading(c, text=f"{c}{arrow}")
        apply_sort()

    for c in cols:
        tree.heading(c, text=c, command=lambda c=c: on_click(c))
    return apply_sort


def _report_errors(title):
    """tkinter callback 的例外只會印到 console（pythonw 下看不到），這裡顯式跳窗"""
    def deco(fn):
        def wrapper(self, *a, **k):
            try:
                return fn(self, *a, **k)
            except Exception:
                import traceback
                tb = traceback.format_exc()
                print(tb)
                messagebox.showerror(title, tb)
                self.status.config(text=title)
        return wrapper
    return deco


class FlickerKPIApp:
    def __init__(self, root):
        self.root = root
        root.title("Flicker KPI Analyzer")
        self.S = ui_scale(root)                      # 依螢幕 dpi 調整的縮放倍率
        fit_window(root, 1500, 950, self.S)

        self.result = None
        self.folder = None
        self.seqs = []          # discover_sequences 結果
        self.cur_idx = None
        self.thresholds = {
            "pixel": dict(DEFAULT_THRESHOLDS["pixel"]),
            "block": dict(DEFAULT_THRESHOLDS["block"]),
        }
        self.vars = {"pixel": {}, "block": {}}
        self.enabled = all_enabled()                 # 每項 metric 的啟用開關
        self.enable_vars = {"pixel": {}, "block": {}}
        self.tree_items = {"pixel": {}, "block": {}}

        self.level = tk.StringVar(value="pixel")
        self.metric = tk.StringVar(value="R_difference")
        self.view_mode = tk.StringVar(value="tile")
        self.tile_text = tk.StringVar(value=str(DEFAULT_TILE))
        self.rule_text = {k: tk.StringVar(value=",".join(f"{x:g}" for x in v))
                          for k, v in DEFAULT_TILE_RULES.items()}
        self.tile_size = DEFAULT_TILE
        self.tile_rules = dict(DEFAULT_TILE_RULES)
        self.sel_tile = None        # (row, col)
        self._tile_view = None      # (tiles, render scale)
        self._geom = None           # 畫面上圖片的 (ox, oy, display scale)
        self._base = None           # (影像, 倍率, ox, oy, 顯示寬, 顯示高)
        self._img_scale = 1.0       # 原圖 pixel -> 繪製影像 pixel 的倍率
        self.rois = {}              # {sequence name: [(x0,y0,x1,y1)]}，對應目前資料夾
        self.sel_box = None
        self._drag = None
        self.draw_roi = tk.BooleanVar(value=False)
        self.risk_bounds_text = tk.StringVar(
            value=",".join(f"{b:g}" for b in DEFAULT_RISK_BOUNDS))
        self.risk_bounds = DEFAULT_RISK_BOUNDS
        self.diff_mode = tk.StringVar(value="peak_to_peak")
        self.y_mode = tk.StringVar(value=DEFAULT_Y_MODE)
        self.pattern = tk.StringVar(value="*.bmp")
        self.vmax_ratio = tk.DoubleVar(value=2.0)
        self.zoom_text = tk.StringVar(value=ZOOM_FIT)
        self.min_fail_text = tk.StringVar(value="1")
        self.min_fail = 1
        self.opt_win = None

        self._build_ui()
        self._autoload_settings()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=6)
        top.pack(fill="x")

        ttk.Button(top, text="載入資料夾", command=self.load_folder).pack(side="left")
        self.path_label = ttk.Label(top, text="(未載入)", width=40)
        self.path_label.pack(side="left", padx=8)

        ttk.Label(top, text="檔名 pattern:").pack(side="left")
        ttk.Entry(top, textvariable=self.pattern, width=10).pack(side="left", padx=4)

        ttk.Label(top, text="difference 定義:").pack(side="left", padx=(12, 2))
        diff_cb = ttk.Combobox(top, textvariable=self.diff_mode, width=14, state="readonly",
                               values=["peak_to_peak", "max_adjacent"])
        diff_cb.pack(side="left")
        diff_cb.bind("<<ComboboxSelected>>", lambda e: self.recompute_all())

        ttk.Label(top, text="Y 定義:").pack(side="left", padx=(12, 2))
        y_cb = ttk.Combobox(top, textvariable=self.y_mode, width=12, state="readonly",
                            values=list(Y_MODES))
        y_cb.pack(side="left")
        y_cb.bind("<<ComboboxSelected>>", lambda e: self._on_y_mode_change())

        ttk.Button(top, text="還原預設 threshold",
                   command=self.reset_thresholds).pack(side="left", padx=12)
        ttk.Button(top, text="匯出目前序列",
                   command=self.export).pack(side="left")
        ttk.Button(top, text="批次分析+匯出全部",
                   command=self.batch_export).pack(side="left", padx=6)

        top2 = ttk.Frame(self.root, padding=(6, 0))
        top2.pack(fill="x")
        ttk.Label(top2, text="FAIL 判定: 任一 metric 超標位置數 >= K，K =").pack(side="left")
        ke = ttk.Entry(top2, textvariable=self.min_fail_text, width=6)
        ke.pack(side="left", padx=4)
        ke.bind("<Return>", lambda e: self.apply_min_fail())
        ttk.Button(top2, text="套用", command=self.apply_min_fail).pack(side="left")
        ttk.Label(top2, text="(pixel 計；block 以 ceil(K/4) 個 block 計)").pack(side="left", padx=6)
        ttk.Button(top2, text="Threshold 最佳化...",
                   command=self.open_optimizer).pack(side="left", padx=20)
        ttk.Button(top2, text="儲存設定", command=self.save_settings_default).pack(side="left")
        ttk.Button(top2, text="另存...", command=self.save_settings_as).pack(side="left", padx=4)
        ttk.Button(top2, text="載入設定...", command=self.load_settings_dialog).pack(side="left")

        body = ttk.PanedWindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=6, pady=6)

        left = ttk.Frame(body)
        body.add(left, weight=3)
        right = ttk.Frame(body)
        body.add(right, weight=4)

        # ---- 左上：序列清單 ----
        sf = ttk.LabelFrame(left, text="序列清單 (<prefix>0001~0006)", padding=4)
        sf.pack(fill="x")
        self.seq_tree = ttk.Treeview(sf, columns=SEQ_COLS, show="headings", height=7,
                                     selectmode="browse")
        for c, w in zip(SEQ_COLS, (170, 55, 55, 55, 65, 60, 65, 85)):
            self.seq_tree.heading(c, text=c)
            self.seq_tree.column(c, width=int(w * self.S), anchor="e")
        self.seq_tree.column("sequence", anchor="w")
        vsb = ttk.Scrollbar(sf, orient="vertical", command=self.seq_tree.yview)
        self.seq_tree.configure(yscrollcommand=vsb.set)
        self.seq_tree.pack(side="left", fill="x", expand=True)
        vsb.pack(side="right", fill="y")
        self.seq_tree.tag_configure("fail", background="#ffd6d6")
        self.seq_tree.tag_configure("pass", background="#e3f7e3")
        self.seq_tree.tag_configure("stale", foreground="#999999")
        self.seq_tree.tag_configure("error", background="#ffb0b0")
        self._resort_seqs = make_sortable(self.seq_tree)
        self.seq_tree.bind("<<TreeviewSelect>>", lambda e: self._on_seq_select())

        # ---- 左：threshold 控制 + 結果表 ----
        nb = ttk.Notebook(left)
        nb.pack(fill="both", expand=True)
        for level, title in (("pixel", "Pixel-based"), ("block", "Block-based (2x2)")):
            tab = ttk.Frame(nb, padding=4)
            nb.add(tab, text=title)
            self._build_level_tab(tab, level)
        nb.bind("<<NotebookTabChanged>>",
                lambda e: self._on_level_change(nb.index("current")))

        # ---- 右：heatmap ----
        ctrl = ttk.Frame(right, padding=4)
        ctrl.pack(fill="x")
        ttk.Label(ctrl, text="顯示 metric:").pack(side="left")
        mcb = ttk.Combobox(ctrl, textvariable=self.metric, width=16,
                           state="readonly", values=METRIC_NAMES)
        mcb.pack(side="left", padx=4)
        mcb.bind("<<ComboboxSelected>>", lambda e: self.refresh_heatmap())

        ttk.Label(ctrl, text="模式:").pack(side="left", padx=(12, 2))
        for text, val in (("分格總圖", "tile"), ("風險分級總圖", "risk"),
                          ("ratio(值/門檻)", "ratio"), ("raw", "raw"),
                          ("fail mask", "binary"), ("綜合超標數", "combined")):
            ttk.Radiobutton(ctrl, text=text, value=val, variable=self.view_mode,
                            command=self.refresh_heatmap).pack(side="left")

        ttk.Label(ctrl, text="色階上限(×門檻):").pack(side="left", padx=(12, 2))
        sc = ttk.Scale(ctrl, from_=1.0, to=5.0, variable=self.vmax_ratio,
                       orient="horizontal", length=100,
                       command=lambda e: self.refresh_heatmap())
        sc.pack(side="left")

        ctrl2 = ttk.Frame(right, padding=(4, 0))
        ctrl2.pack(fill="x")
        ttk.Label(ctrl2, text="風險分界 (×門檻, 4 個遞增值 → SAFE/WATCH/NEAR/FAIL/SEVERE):"
                  ).pack(side="left")
        be = ttk.Entry(ctrl2, textvariable=self.risk_bounds_text, width=18)
        be.pack(side="left", padx=4)
        be.bind("<Return>", lambda e: self.apply_risk_bounds())
        ttk.Button(ctrl2, text="套用", command=self.apply_risk_bounds).pack(side="left")

        ctrl3 = ttk.Frame(right, padding=(4, 2))
        ctrl3.pack(fill="x")
        ttk.Label(ctrl3, text="分格:").pack(side="left")
        tcb = ttk.Combobox(ctrl3, textvariable=self.tile_text, width=5,
                           values=["20", "40", "60", "120"])
        tcb.pack(side="left", padx=(2, 10))
        tcb.bind("<<ComboboxSelected>>", lambda e: self.apply_tile_params())
        tcb.bind("<Return>", lambda e: self.apply_tile_params())
        for key in ("RED", "ORANGE"):
            ttk.Label(ctrl3, text=f"{key} (ratio,px,blk 任一達到):").pack(side="left")
            e = ttk.Entry(ctrl3, textvariable=self.rule_text[key], width=10)
            e.pack(side="left", padx=(2, 10))
            e.bind("<Return>", lambda ev: self.apply_tile_params())
        ttk.Button(ctrl3, text="套用", command=self.apply_tile_params).pack(side="left")

        ttk.Checkbutton(ctrl3, text="畫框", variable=self.draw_roi,
                        command=self._refresh_roi_hint).pack(side="left", padx=(10, 2))
        ttk.Button(ctrl3, text="刪除選取框", command=self._delete_box).pack(side="left")
        ttk.Button(ctrl3, text="清除本序列", command=self._clear_boxes).pack(side="left", padx=4)
        ttk.Button(ctrl3, text="儲存框", command=self._save_rois).pack(side="left")

        ttk.Label(ctrl3, text="縮放:").pack(side="left", padx=(10, 2))
        zcb = ttk.Combobox(ctrl3, textvariable=self.zoom_text, width=8, state="readonly",
                           values=ZOOM_LEVELS)
        zcb.pack(side="left")
        zcb.bind("<<ComboboxSelected>>", lambda e: self.refresh_heatmap())
        ttk.Button(ctrl3, text="-", width=3,
                   command=lambda: self._zoom_step(-1)).pack(side="left", padx=(4, 0))
        ttk.Button(ctrl3, text="+", width=3,
                   command=lambda: self._zoom_step(1)).pack(side="left")

        cf = ttk.Frame(right)
        cf.pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas = tk.Canvas(cf, bg="#202020")
        hsb = ttk.Scrollbar(cf, orient="horizontal",
                            command=lambda *a: self._scroll(self.canvas.xview, *a))
        vsb = ttk.Scrollbar(cf, orient="vertical",
                            command=lambda *a: self._scroll(self.canvas.yview, *a))
        self.canvas.configure(xscrollcommand=hsb.set, yscrollcommand=vsb.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="we")
        cf.rowconfigure(0, weight=1)
        cf.columnconfigure(0, weight=1)
        self.canvas.bind("<Button-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Button-3>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B3-Motion>", self._on_pan)
        self.canvas.bind("<Configure>", lambda e: self._redraw_viewport())
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):   # Windows / X11
            self.canvas.bind(seq, self._on_wheel)
        self.canvas.bind("<Control-MouseWheel>", self._on_wheel_zoom)
        self.canvas.bind("<Control-Button-4>", self._on_wheel_zoom)
        self.canvas.bind("<Control-Button-5>", self._on_wheel_zoom)

        inf = ttk.Frame(right)
        inf.pack(fill="x", padx=4, pady=(0, 4))
        self.info = tk.Text(inf, height=11, font=("Consolas", 9), wrap="none")
        isb = ttk.Scrollbar(inf, orient="vertical", command=self.info.yview)
        self.info.configure(yscrollcommand=isb.set)
        self.info.pack(side="left", fill="x", expand=True)
        isb.pack(side="right", fill="y")

        self.status = ttk.Label(self.root, text="就緒", relief="sunken", anchor="w")
        self.status.pack(fill="x", side="bottom")

        for row in (top, top2, ctrl, ctrl2, ctrl3):   # 視窗變窄時控制列自動換行
            make_wrapping(row)

    def _build_level_tab(self, parent, level):
        thr_frame = ttk.LabelFrame(parent, text="Threshold (可直接輸入或拖曳)", padding=4)
        thr_frame.pack(fill="x")

        for i, name in enumerate(METRIC_NAMES):
            row = i % 5
            col = i // 5
            ev = tk.BooleanVar(value=self.enabled[level][name])
            self.enable_vars[level][name] = ev
            cb = ttk.Checkbutton(thr_frame, text=name, width=15, variable=ev,
                                 command=lambda l=level: self.apply_thresholds(l))
            cb.grid(row=row, column=col * 3, sticky="w", pady=1)
            var = tk.DoubleVar(value=self.thresholds[level][name])
            self.vars[level][name] = var
            ent = ttk.Entry(thr_frame, textvariable=var, width=9)
            ent.grid(row=row, column=col * 3 + 1, padx=(2, 8))
            ent.bind("<Return>", lambda e, l=level: self.apply_thresholds(l))
            ent.bind("<FocusOut>", lambda e, l=level: self.apply_thresholds(l))

        btns = ttk.Frame(thr_frame)
        btns.grid(row=5, column=0, columnspan=6, pady=4, sticky="we")
        ttk.Button(btns, text="套用",
                   command=lambda l=level: self.apply_thresholds(l)).pack(side="left")
        ttk.Button(btns, text="本頁全開",
                   command=lambda l=level: self._set_level_enabled(l, True)
                   ).pack(side="left", padx=4)
        ttk.Button(btns, text="本頁全關",
                   command=lambda l=level: self._set_level_enabled(l, False)).pack(side="left")
        ttk.Label(btns, text="(取消勾選 = 該 metric 不列入判定)").pack(side="left", padx=6)

        cols = ("metric", "thr", "fail%", "fail_cnt", "max", "p99.9", "mean", "max/thr")
        tree = ttk.Treeview(parent, columns=cols, show="headings", height=12)
        for c, w in zip(cols, (120, 70, 80, 80, 80, 80, 75, 70)):
            tree.heading(c, text=c)
            tree.column(c, width=int(w * self.S), anchor="e")
        tree.column("metric", anchor="w")
        make_sortable(tree)
        tree.pack(fill="both", expand=True, pady=4)
        tree.tag_configure("fail", background="#ffd6d6")
        tree.tag_configure("pass", background="#e3f7e3")
        tree.tag_configure("off", background="#ebebeb", foreground="#999999")
        tree.bind("<<TreeviewSelect>>", lambda e, t=tree: self._on_tree_select(t))
        setattr(self, f"tree_{level}", tree)

    # ------------------------------------------------------------------
    # 行為
    # ------------------------------------------------------------------
    def _on_level_change(self, idx):
        self.level.set("pixel" if idx == 0 else "block")
        self.refresh_heatmap()

    def _on_tree_select(self, tree):
        sel = tree.selection()
        if sel:
            self.metric.set(tree.item(sel[0], "values")[0])
            self.refresh_heatmap()

    @_report_errors("載入失敗")
    def load_folder(self):
        folder = filedialog.askdirectory(title="選擇資料夾（含一組或多組 xxxx0001~0006）")
        if not folder:
            return
        seqs, skipped = discover_sequences(folder, self.pattern.get(), N_FRAMES)
        if not seqs:
            msg = f"{folder} 內找不到完整的 <prefix>0001~{N_FRAMES:04d} ({self.pattern.get()})"
            if skipped:
                msg += "\n\n不完整:\n" + "\n".join(f"  {p}: {r}" for p, r in skipped[:20])
            messagebox.showerror("沒有可分析的序列", msg)
            return
        self.folder, self.seqs, self.cur_idx, self.result = folder, seqs, None, None
        self.rois = load_rois(folder)
        self.sel_box = None
        self.seq_tree.delete(*self.seq_tree.get_children())
        for i, sq in enumerate(seqs):
            self.seq_tree.insert("", "end", iid=str(i),
                                 values=(sq["name"], "—", "", "", "", "", "", ""))
        self.path_label.config(text=folder)
        if self.rois:
            n = sum(len(v) for v in self.rois.values())
            self.status.config(text=f"已讀入 {ROI_FILE}：{n} 個框（{len(self.rois)} 組序列）")
        if skipped:
            messagebox.showwarning(
                "部分 prefix 未納入",
                f"找到 {len(seqs)} 組完整序列；以下 {len(skipped)} 組缺幀，已略過:\n"
                + "\n".join(f"  {p}: {r}" for p, r in skipped[:20]))
        self.seq_tree.selection_set("0")     # 觸發 _on_seq_select 分析第一組

    def _on_seq_select(self):
        sel = self.seq_tree.selection()
        if sel and int(sel[0]) != self.cur_idx:
            self._analyze_current(int(sel[0]))

    @_report_errors("分析失敗")
    def _analyze_current(self, idx):
        sq = self.seqs[idx]
        self.status.config(text=f"計算中: {sq['name']} ...")
        self.root.update_idletasks()
        self.result = analyze_sequence(
            self.folder, thresholds=self.thresholds, diff_mode=self.diff_mode.get(),
            paths=sq["paths"], name=sq["name"], y_mode=self.y_mode.get())
        self.cur_idx = idx
        self.sel_tile = None
        self.sel_box = None
        self.update_tables()
        self.refresh_heatmap()
        self._update_current_row()
        h, w = self.result["shape"]
        extra = f"   (忽略多出的 frame: {sq['extra']})" if sq["extra"] else ""
        self.status.config(text=f"[{idx+1}/{len(self.seqs)}] {sq['name']}: "
                                f"{self.result['n_frames']} frames, {w}x{h}{extra}")

    def _set_seq_row(self, idx, sm, tag):
        self.seq_tree.item(str(idx), tags=(tag,), values=(
            sm["name"], sm["verdict"], sm["pixel_fail_metrics"], sm["block_fail_metrics"],
            f"{sm['danger_pct']:.3f}", sm["worst_level"], f"{sm['max_ratio']:.2f}",
            f"{sm['tiles_red']}/{sm['tiles_orange']}/{sm['tiles_yellow']}"))

    def _update_current_row(self):
        if self.result is None or self.cur_idx is None:
            return
        risk = compute_risk(self.result, self.risk_bounds, enabled=self.enabled)
        sm = summarize_result(self.result, risk, self._tiles_for(self.result, risk),
                              self.min_fail, self.enabled)
        self._set_seq_row(self.cur_idx, sm, "fail" if sm["verdict"] == "FAIL" else "pass")
        self._resort_seqs()

    def _mark_others_stale(self):
        """threshold / 定義改了之後，其他序列的摘要已過期（灰字），需重選或重跑批次"""
        for iid in self.seq_tree.get_children():
            if int(iid) != self.cur_idx and self.seq_tree.item(iid, "values")[1] != "—":
                self.seq_tree.item(iid, tags=("stale",))

    def _on_y_mode_change(self):
        """Y 定義改變 -> metric map 要重算；R+2G+B 與其餘兩者量級差 4 倍，提醒 threshold"""
        self.recompute_all()
        note = ""
        if self.y_mode.get() == "R+2G+B":
            note = "   (量級為 (R+2G+B)/4 的 4 倍：Y_difference threshold 需 x4、Y_variance x16)"
        self.status.config(text=f"Y 定義: {self.y_mode.get()}{note}")

    def recompute_all(self):
        if self.cur_idx is not None:
            self._analyze_current(self.cur_idx)
            self._mark_others_stale()

    def apply_thresholds(self, level=None):
        levels = [level] if level else ["pixel", "block"]
        for l in levels:
            for name, var in self.vars[l].items():
                try:
                    self.thresholds[l][name] = float(var.get())
                except (tk.TclError, ValueError):
                    pass
                try:
                    self.enabled[l][name] = bool(self.enable_vars[l][name].get())
                except (tk.TclError, ValueError):
                    pass
        if self.result:
            # 只重算判定，不重算 metric map
            for l in ["pixel", "block"]:
                self.result[l]["report"] = evaluate_kpi(
                    self.result[l]["maps"], self.thresholds[l])
            self.update_tables()
            self.refresh_heatmap()
            self._refresh_box_info()
            self._update_current_row()
            self._mark_others_stale()

    def apply_risk_bounds(self):
        try:
            self.risk_bounds = parse_risk_bounds(self.risk_bounds_text.get())
        except ValueError as e:
            messagebox.showerror("風險分界錯誤", str(e))
            return
        self.view_mode.set("risk")
        self.refresh_heatmap()
        self._update_current_row()
        self._mark_others_stale()

    # ------------------------------------------------------------------
    # 設定存取
    # ------------------------------------------------------------------
    def collect_settings(self):
        self.apply_thresholds()                     # 先把輸入框的值收進 self.thresholds
        return {
            "thresholds": {lv: dict(self.thresholds[lv]) for lv in ("pixel", "block")},
            "enabled": {lv: dict(self.enabled[lv]) for lv in ("pixel", "block")},
            "min_fail": int(self.min_fail),
            "risk_bounds": list(self.risk_bounds),
            "tile_size": int(self.tile_size),
            "tile_rules": {k: list(v) for k, v in self.tile_rules.items()},
            "diff_mode": self.diff_mode.get(),
            "y_mode": self.y_mode.get(),
            "pattern": self.pattern.get(),
        }

    def apply_settings(self, st):
        """套用載入的設定；未出現在設定檔中的項目維持現狀"""
        thr = st.get("thresholds", {})
        for lv in ("pixel", "block"):
            for m, v in thr.get(lv, {}).items():
                self.vars[lv][m].set(float(v))
        for lv in ("pixel", "block"):
            for m, v in st.get("enabled", {}).get(lv, {}).items():
                self.enable_vars[lv][m].set(bool(v))
        if "min_fail" in st:
            self.min_fail = int(st["min_fail"])
            self.min_fail_text.set(str(self.min_fail))
        if "risk_bounds" in st:
            self.risk_bounds = tuple(st["risk_bounds"])
            self.risk_bounds_text.set(",".join(f"{b:g}" for b in self.risk_bounds))
        if "tile_size" in st:
            self.tile_size = int(st["tile_size"])
            self.tile_text.set(str(self.tile_size))
        if "tile_rules" in st:
            self.tile_rules = {k: tuple(v) for k, v in st["tile_rules"].items()}
            for k, v in self.tile_rules.items():
                self.rule_text[k].set(",".join(f"{x:g}" for x in v))
        if "pattern" in st:
            self.pattern.set(st["pattern"])
        need_recompute = any(st.get(k) not in (None, v.get()) for k, v in
                             (("diff_mode", self.diff_mode), ("y_mode", self.y_mode)))
        if "diff_mode" in st:
            self.diff_mode.set(st["diff_mode"])
        if "y_mode" in st:
            self.y_mode.set(st["y_mode"])
        if self.result is not None and need_recompute:
            self.recompute_all()                    # diff 定義改了要重算 metric map
        else:
            self.apply_thresholds()                 # 只重算判定

    def _autoload_settings(self):
        path = default_settings_path()
        if not os.path.isfile(path):
            self.status.config(text=f"就緒（尚無設定檔，關閉視窗時會存到 {path}）")
            return
        try:
            st, warn = load_settings(path)
        except (OSError, ValueError) as e:
            self.status.config(text=f"設定檔載入失敗: {e}")
            return
        self.apply_settings(st)
        self.status.config(text=f"已載入設定: {path}"
                                + (f"   ({len(warn)} 項已略過)" if warn else ""))
        if warn:
            print("設定檔警告:\n  " + "\n  ".join(warn))

    def save_settings_default(self):
        self._save_to(default_settings_path())

    def _save_to(self, path):
        try:
            save_settings(path, self.collect_settings())
        except OSError as e:
            messagebox.showerror("儲存設定失敗", f"{path}\n\n{e}")
            self.status.config(text="儲存設定失敗")
            return
        self.status.config(text=f"設定已儲存: {path}")

    def save_settings_as(self):
        path = filedialog.asksaveasfilename(
            title="另存設定", defaultextension=".json",
            initialfile=os.path.basename(default_settings_path()),
            filetypes=[("JSON", "*.json"), ("所有檔案", "*.*")])
        if not path:
            return
        self._save_to(path)

    def load_settings_dialog(self):
        path = filedialog.askopenfilename(title="載入設定",
                                          filetypes=[("JSON", "*.json"), ("所有檔案", "*.*")])
        if not path:
            return
        try:
            st, warn = load_settings(path)
        except (OSError, ValueError) as e:
            messagebox.showerror("載入設定失敗", f"{path}\n\n{e}")
            self.status.config(text="載入設定失敗")
            return
        self.apply_settings(st)
        self.status.config(text=f"已載入設定: {path}")
        if warn:
            messagebox.showwarning("部分項目未套用", "\n".join(warn))

    def _on_close(self):
        """關閉視窗時自動存回預設設定檔，下次開啟就是這組參數"""
        try:
            save_settings(default_settings_path(), self.collect_settings())
        except (OSError, ValueError, tk.TclError) as e:
            print(f"設定自動儲存失敗: {e}")
        self.root.destroy()

    def _refresh_box_info(self):
        if self.sel_box is not None and self.result:
            self._show_box_info(self.sel_box)

    def apply_min_fail(self):
        try:
            k = int(self.min_fail_text.get())
            if k < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("K 錯誤", "K 需為 >= 1 的整數")
            return
        self.min_fail = k
        if self.result:
            self.update_tables()
            self.refresh_heatmap()
            self._refresh_box_info()
            self._update_current_row()
            self._mark_others_stale()

    def open_optimizer(self):
        if self.opt_win is not None and self.opt_win.win.winfo_exists():
            self.opt_win.win.lift()
            return
        self.opt_win = ThresholdOptWindow(self)

    def set_thresholds_and_k(self, thr, K):
        """由最佳化視窗套用：thr = {"pixel": {...}, "block": {...}}"""
        for lv in ("pixel", "block"):
            for m, v in thr[lv].items():
                self.vars[lv][m].set(float(v))    # 最佳化端已進位到 4 位並驗證，不可再捨入
        self.min_fail_text.set(str(int(K)))
        self.min_fail = int(K)
        self.apply_thresholds()
        if self.result is None:
            return
        self._update_current_row()
        self._mark_others_stale()

    def apply_tile_params(self):
        try:
            bs = self.result.get("block_size", 2) if self.result else 2
            tile = parse_tile_size(self.tile_text.get(), block_size=bs)
            rules = {k: parse_tile_rule(v.get()) for k, v in self.rule_text.items()}
        except ValueError as e:
            messagebox.showerror("分格參數錯誤", str(e))
            return
        if tile != self.tile_size:
            self.sel_tile = None
        self.tile_size, self.tile_rules = tile, rules
        self.view_mode.set("tile")
        self.refresh_heatmap()
        self._update_current_row()
        self._mark_others_stale()

    def _tiles_for(self, result, risk):
        return compute_tiles(result, risk, self.tile_size, self.tile_rules)

    def _set_level_enabled(self, level, on):
        for v in self.enable_vars[level].values():
            v.set(on)
        self.apply_thresholds(level)

    def reset_thresholds(self):
        for l in ("pixel", "block"):
            for name, v in DEFAULT_THRESHOLDS[l].items():
                self.vars[l][name].set(v)
        self.apply_thresholds()

    def update_tables(self):
        for level in ("pixel", "block"):
            tree = getattr(self, f"tree_{level}")
            tree.delete(*tree.get_children())
            rep = self.result[level]["report"]
            bs = self.result.get("block_size", 2)
            for name in METRIC_NAMES:
                s = rep[name]
                if not self.enabled[level][name]:
                    tag = "off"
                else:
                    tag = "fail" if metric_is_fail(s, level, self.min_fail, bs) else "pass"
                tree.insert("", "end", values=(
                    name, f"{s['threshold']:.1f}", f"{s['fail_ratio']*100:.4f}%",
                    s["fail_count"], f"{s['max']:.1f}", f"{s['p999']:.1f}",
                    f"{s['mean']:.2f}", f"{s['margin']:.2f}"), tags=(tag,))

    # ------------------------------------------------------------------
    # heatmap
    # ------------------------------------------------------------------
    def refresh_heatmap(self):
        if not self.result:
            return
        level = self.level.get()
        rep = self.result[level]["report"]
        maps = self.result[level]["maps"]
        name = self.metric.get()
        mode = self.view_mode.get()

        H, W = self.result["shape"]
        z = self._zoom_factor()
        # 繪圖目標尺寸：符合視窗時用畫布大小，手動縮放時用 原圖 x 倍率
        if z is None:
            target = self._canvas_size()
        else:
            target = (max(1, int(round(W * z))), max(1, int(round(H * z))) + LEGEND_H)

        if mode != "tile":
            self._tile_view = None
        if mode == "tile":
            risk = compute_risk(self.result, self.risk_bounds, enabled=self.enabled)
            tiles = self._tiles_for(self.result, risk)
            img, sc = render_tile_map(self.result, tiles, fit=target)
            self._tile_view = (tiles, sc)
            self._img_scale = sc
            self._show(img, target[0] / img.shape[1] if z is not None else None)
            self._draw_tile_selection()
            self._draw_rois()
            self._update_tile_info(tiles)
            self._refresh_box_info()
            return

        if mode == "risk":
            risk = compute_risk(self.result, self.risk_bounds, enabled=self.enabled)
            img, sc = render_risk_map(self.result, risk, fit=target)
            self._img_scale = sc
            self._show(img, target[0] / img.shape[1] if z is not None else None)
            self._update_risk_info(risk)
            self._draw_rois()
            self._refresh_box_info()
            return

        # 縮圖一律用 max-pool（先換算回原圖尺度），1x1 超標點才不會在畫面上消失
        f = max(1, H // maps[name].shape[0])          # block-based 的放大倍率
        fit = (max(1, target[0] // f), max(1, (target[1] - (LEGEND_H if z else 0)) // f))

        if mode == "combined":
            acc, pool = max_pool_fit(combined_fail_map(rep), *fit)
            mx = max(int(acc.max()), 1)
            vis = cv2.applyColorMap(
                (acc / mx * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            desc = f"綜合超標數 (0~{mx} 項 metric 同時 fail)"
        else:
            m, pool = max_pool_fit(maps[name], *fit)
            thr = rep[name]["threshold"]
            vis = render_heatmap(m, thr, mode=mode,
                                 vmax_ratio=self.vmax_ratio.get())
            desc = f"{name}  mode={mode}  threshold={thr:.1f}"

        # block-based 放大回與 pixel-based 相同的顯示尺度
        vis = upscale(vis, f)
        self._img_scale = 1.0 / pool
        self._show(vis, target[0] / vis.shape[1] if z is not None else None)
        self._update_info(level, name, mode, desc)
        self._draw_rois()
        self._refresh_box_info()

    # ------------------------------------------------------------------
    # 縮放 / 捲動
    # ------------------------------------------------------------------
    def _zoom_factor(self):
        """None = 符合視窗；否則為顯示倍率"""
        t = self.zoom_text.get()
        if t == ZOOM_FIT:
            return None
        try:
            return float(t.rstrip("%")) / 100
        except ValueError:
            return None

    def _zoom_step(self, d):
        try:
            i = ZOOM_LEVELS.index(self.zoom_text.get())
        except ValueError:
            i = 0
        self.zoom_text.set(ZOOM_LEVELS[min(max(i + d, 0), len(ZOOM_LEVELS) - 1)])
        self.refresh_heatmap()

    @staticmethod
    def _wheel_dir(ev):
        return -1 if (getattr(ev, "num", None) == 5 or getattr(ev, "delta", 0) < 0) else 1

    def _scroll(self, view, *args):
        view(*args)
        self._redraw_viewport()
        self._draw_tile_selection()
        self._draw_rois()

    def _on_pan(self, ev):
        self.canvas.scan_dragto(ev.x, ev.y, gain=1)
        self._redraw_viewport()

    def _on_wheel(self, ev):
        self.canvas.yview_scroll(-self._wheel_dir(ev), "units")
        self._redraw_viewport()

    def _on_wheel_zoom(self, ev):
        self._zoom_step(self._wheel_dir(ev))
        return "break"

    def _canvas_size(self):
        self.canvas.update_idletasks()
        return max(self.canvas.winfo_width(), 100), max(self.canvas.winfo_height(), 100)

    def _show(self, bgr, scale=None):
        """
        scale=None -> 縮到符合視窗；否則以指定倍率顯示（可捲動）。
        只把畫布看得到的那一塊放大貼上，放大倍率再高也不會做出整張巨圖。
        """
        cw, chh = self._canvas_size()
        h, w = bgr.shape[:2]
        if scale is None:
            scale = min(cw / w, chh / h)
        dw, dh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        ox, oy = max(0, (cw - dw) // 2), max(0, (chh - dh) // 2)   # 比畫布小就置中
        self._base = (bgr, scale, ox, oy, dw, dh)
        self._geom = (ox, oy, scale)
        self.canvas.configure(scrollregion=(0, 0, max(cw, ox + dw), max(chh, oy + dh)))
        self._redraw_viewport()

    def _redraw_viewport(self):
        """只繪製目前視野內的影像區塊"""
        if self._base is None:
            return
        bgr, scale, ox, oy, dw, dh = self._base
        cw, chh = self._canvas_size()
        vx0, vy0 = self.canvas.canvasx(0), self.canvas.canvasy(0)
        # 視野與影像的交集（畫布座標）
        x0, y0 = max(vx0, ox), max(vy0, oy)
        x1, y1 = min(vx0 + cw, ox + dw), min(vy0 + chh, oy + dh)
        self.canvas.delete("img")
        if x1 <= x0 or y1 <= y0:
            return
        h, w = bgr.shape[:2]
        sx0, sy0 = int((x0 - ox) / scale), int((y0 - oy) / scale)
        sx1 = min(w, int(np.ceil((x1 - ox) / scale)))
        sy1 = min(h, int(np.ceil((y1 - oy) / scale)))
        crop = bgr[sy0:sy1, sx0:sx1]
        if crop.size == 0:
            return
        tw = max(1, int(round(crop.shape[1] * scale)))
        th = max(1, int(round(crop.shape[0] * scale)))
        disp = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_NEAREST)
        self._photo = ImageTk.PhotoImage(Image.fromarray(disp[:, :, ::-1]))
        self.canvas.create_image(ox + sx0 * scale, oy + sy0 * scale,
                                 image=self._photo, anchor="nw", tags="img")
        self.canvas.tag_lower("img")

    # ------------------------------------------------------------------
    # 人工框 (ROI)
    # ------------------------------------------------------------------
    def _cur_name(self):
        return self.result["name"] if self.result else None

    def _cur_boxes(self):
        return self.rois.get(self._cur_name(), [])

    def _canvas_to_img(self, ev):
        """畫布事件座標 -> 原圖 pixel 座標"""
        if self._geom is None:
            return None
        ox, oy, ds = self._geom
        sc = self._img_scale or 1.0
        x = (self.canvas.canvasx(ev.x) - ox) / ds / sc
        y = (self.canvas.canvasy(ev.y) - oy) / ds / sc
        return x, y

    def _img_to_canvas(self, x, y):
        ox, oy, ds = self._geom
        sc = self._img_scale or 1.0
        return ox + x * sc * ds, oy + y * sc * ds

    def _refresh_roi_hint(self):
        self.status.config(
            text="畫框模式：左鍵拖曳新增框；分格總圖下單擊可加入該格；"
                 "點框內可選取後刪除（右鍵拖曳仍可平移）"
            if self.draw_roi.get() else "畫框模式已關閉")
        self._draw_rois()

    def _draw_rois(self):
        self.canvas.delete("roi")
        if self._geom is None or not self.result:
            return
        for i, b in enumerate(self._cur_boxes()):
            x0, y0 = self._img_to_canvas(b[0], b[1])
            x1, y1 = self._img_to_canvas(b[2], b[3])
            sel = (i == self.sel_box)
            self.canvas.create_rectangle(x0, y0, x1, y1, tags="roi",
                                         outline="#00ff00" if not sel else "#ffffff",
                                         width=3 if sel else 2,
                                         dash=() if not sel else (5, 3))

    def _add_box(self, box):
        box = clip_box(box, self.result["shape"])
        if box is None:
            return
        self.rois.setdefault(self._cur_name(), []).append(box)
        self.sel_box = len(self.rois[self._cur_name()]) - 1
        self._draw_rois()
        self._show_box_info(self.sel_box)
        self.status.config(text=f"已新增框 x {box[0]}~{box[2]-1}, y {box[1]}~{box[3]-1}"
                                f"（本序列共 {len(self._cur_boxes())} 個，記得按「儲存框」）")

    def _hit_box(self, x, y):
        for i, b in enumerate(self._cur_boxes()):
            if b[0] <= x < b[2] and b[1] <= y < b[3]:
                return i
        return None

    def _delete_box(self):
        name = self._cur_name()
        if name is None or self.sel_box is None or self.sel_box >= len(self._cur_boxes()):
            messagebox.showinfo("未選取", "請先在畫框模式下點選一個框")
            return
        self.rois[name].pop(self.sel_box)
        self.sel_box = None
        self._draw_rois()
        self.status.config(text=f"已刪除，本序列剩 {len(self._cur_boxes())} 個框")

    def _clear_boxes(self):
        name = self._cur_name()
        if name is None or not self._cur_boxes():
            return
        if messagebox.askyesno("清除", f"刪除 {name} 的全部 {len(self._cur_boxes())} 個框？"):
            self.rois.pop(name, None)
            self.sel_box = None
            self._draw_rois()
            self.status.config(text=f"已清除 {name} 的框")

    @_report_errors("儲存框失敗")
    def _save_rois(self):
        if not self.folder:
            messagebox.showwarning("尚未載入", "請先載入資料夾")
            return
        path = save_rois(self.folder, self.rois)
        n = sum(len(v) for v in self.rois.values())
        self.status.config(text=f"已儲存 {n} 個框（{len(self.rois)} 組序列）-> {path}")

    # ------------------------------------------------------------------
    def _on_canvas_press(self, ev):
        if self.draw_roi.get() and self.result:
            p = self._canvas_to_img(ev)
            if p is None:
                return
            self._drag = (self.canvas.canvasx(ev.x), self.canvas.canvasy(ev.y), p)
            return
        self._on_canvas_click(ev)

    def _on_canvas_drag(self, ev):
        if not self._drag:
            return
        x0, y0, _ = self._drag
        self.canvas.delete("rubber")
        self.canvas.create_rectangle(x0, y0, self.canvas.canvasx(ev.x),
                                     self.canvas.canvasy(ev.y),
                                     outline="#00ff00", dash=(3, 2), tags="rubber")

    def _on_canvas_release(self, ev):
        if not self._drag:
            return
        cx0, cy0, (ix0, iy0) = self._drag
        self._drag = None
        self.canvas.delete("rubber")
        p = self._canvas_to_img(ev)
        if p is None or not self.result:
            return
        ix1, iy1 = p
        moved = max(abs(self.canvas.canvasx(ev.x) - cx0),
                    abs(self.canvas.canvasy(ev.y) - cy0))
        if moved < 4:                                   # 視為單擊
            hit = self._hit_box(ix0, iy0)
            if hit is not None:                         # 點到既有框 -> 選取並列出明細
                b = self._cur_boxes()[hit]
                self._select_box(hit)
                self.status.config(text=f"選取框 x {b[0]}~{b[2]-1}, y {b[1]}~{b[3]-1}"
                                        f"（明細見下方；按「刪除選取框」可移除）")
                return
            if self.view_mode.get() == "tile" and self._tile_view:   # 分格總圖 -> 整格加入
                t = self._tile_view[0]["tile"]
                c, r = int(ix0 // t), int(iy0 // t)
                self._add_box((c * t, r * t, (c + 1) * t, (r + 1) * t))
            return
        self._add_box((ix0, iy0, ix1, iy1))

    def _select_box(self, i):
        self.sel_box = i
        self.sel_tile = None
        self._draw_tile_selection()
        self._draw_rois()
        self._show_box_info(i)

    def _metric_table(self, box, label="範圍"):
        """某一範圍內 20 項 metric 的最大值 / 門檻 / 比值 / 超標數，依比值排序"""
        rows = box_metric_stats(self.result, box, self.enabled, self.min_fail)
        rows.sort(key=lambda r: -r["ratio"])
        hit = [r for r in rows if r["enabled"] and r["fail"]]
        hit_txt = ("閃  (" + ", ".join(f"{r['level']}:{r['metric']}" for r in hit[:3])
                   + (" ...)" if len(hit) > 3 else ")")) if hit else "未達門檻"
        lines = [f"判定: {hit_txt}", "",
                 f"{'metric':<22}{label + '最大':>10}{'門檻':>10}{'最大/門檻':>10}"
                 f"{'超標數':>9}{label + '內數':>8}  狀態"]
        for r in rows:
            state = "停用" if not r["enabled"] else ("FAIL" if r["fail"] else "")
            lines.append(f"{r['level']+':'+r['metric']:<22}{r['max']:>10.2f}{r['thr']:>10.2f}"
                         f"{r['ratio']:>10.2f}{r['count']:>9d}{r['total']:>8d}  {state}")
        lines.append("")
        lines.append(f"超標數 = {label}內超過門檻的位置數（block 層級以 2x2 block 計）；"
                     f"FAIL = 超標數 >= min(K, {label}內位置數)")
        return lines

    def _show_box_info(self, i):
        boxes = self._cur_boxes()
        if not self.result or i is None or i >= len(boxes):
            return
        b = boxes[i]
        w, h = b[2] - b[0], b[3] - b[1]
        txt = [f"框 #{i+1}/{len(boxes)}   x {b[0]}~{b[2]-1}, y {b[1]}~{b[3]-1}"
               f"   ({w}x{h} = {w*h} px)   K={self.min_fail}"]
        txt += self._metric_table(b, "框")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", "\n".join(txt))

    def _on_canvas_click(self, ev):
        p = self._canvas_to_img(ev)                 # 先判斷有沒有點到框
        if p is not None and self.result:
            hit = self._hit_box(*p)
            if hit is not None:
                self._select_box(hit)
                return
        if self._tile_view is None or self._geom is None:
            return
        tiles, sc = self._tile_view
        ox, oy, ds = self._geom
        x = (self.canvas.canvasx(ev.x) - ox) / ds / sc        # -> 原圖 pixel 座標
        y = (self.canvas.canvasy(ev.y) - oy) / ds / sc
        H, W = tiles["shape"]
        if not (0 <= x < W and 0 <= y < H):
            return
        t = tiles["tile"]
        self.sel_tile = (int(y // t), int(x // t))
        self.sel_box = None                     # 改看格子 -> 取消框的選取
        self._draw_rois()
        self._draw_tile_selection()
        self._update_tile_info(tiles)

    def _draw_tile_selection(self):
        self.canvas.delete("sel")
        if self._tile_view is None or self._geom is None or self.sel_tile is None:
            return
        tiles, sc = self._tile_view
        ox, oy, ds = self._geom
        H, W = tiles["shape"]
        t = tiles["tile"]
        r, c = self.sel_tile
        k = sc * ds
        self.canvas.create_rectangle(ox + c * t * k, oy + r * t * k,
                                     ox + min((c + 1) * t, W) * k,
                                     oy + min((r + 1) * t, H) * k,
                                     outline="#00ffff", width=2, dash=(4, 2), tags="sel")

    def _update_tile_info(self, tiles):
        c = tiles["counts"]
        txt = [f"分格總圖  tile {tiles['tile']}x{tiles['tile']}   "
               f"超標格 {tiles['n_fail']}/{tiles['n_tiles']} "
               f"({tiles['n_fail']/tiles['n_tiles']*100:.2f}%)",
               f"  RED {c['RED']}   ({tile_rule_text(tiles['rules']['RED'])})",
               f"  ORANGE {c['ORANGE']}   ({tile_rule_text(tiles['rules']['ORANGE'])})",
               f"  YELLOW {c['YELLOW']}   (其餘有超標的格子)", ""]
        if self.sel_tile is None:
            txt.append("點選圖上的格子查看該格 20 項 metric 的明細")
        else:
            d = tile_detail(tiles, *self.sel_tile)
            txt.append(f"選取格 (col {d['col']}, row {d['row']})  x {d['x0']}~{d['x1']-1}, "
                       f"y {d['y0']}~{d['y1']-1}   等級: {d['level']}   "
                       f"最大 ratio {d['max_ratio']:.2f}x @ x={d['max_x']}, y={d['max_y']}")
            txt += self._metric_table((d["x0"], d["y0"], d["x1"], d["y1"]), "格")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", "\n".join(txt))

    def _update_risk_info(self, risk):
        txt = ["風險分級總圖  risk = max(value/threshold)，pixel 10 項 + block 10 項", ""]
        for (label, _), rng, fr in zip(RISK_LEVELS, risk_range_labels(risk["bounds"]),
                                       risk["frac"]):
            txt.append(f"  {label:<7}{rng:<12}{fr*100:9.4f}%")
        danger = risk["level"] >= 3
        if danger.any():
            cnt = np.bincount(risk["best"][danger].ravel(), minlength=len(risk["names"]))
            top = np.argsort(-cnt)[:4]
            txt.append("")
            txt.append("FAIL/SEVERE 區主要成因 (最大 ratio 來源 metric, 佔危險區 %):")
            txt.append("  " + ",  ".join(
                f"{risk['names'][i]} {cnt[i]/danger.sum()*100:.1f}%"
                for i in top if cnt[i] > 0))
        self.info.delete("1.0", "end")
        self.info.insert("1.0", "\n".join(txt))

    def _update_info(self, level, name, mode, desc):
        rep = self.result[level]["report"]
        txt = [desc, ""]
        if mode != "combined":
            s = rep[name]
            txt.append(f"超標比例 : {s['fail_ratio']*100:.4f}%  "
                       f"({s['fail_count']} / {s['total']})")
            txt.append(f"最大值   : {s['max']:.2f}   (= {s['margin']:.2f} x threshold)")
            txt.append(f"p99.9    : {s['p999']:.2f}    p99: {s['p99']:.2f}    "
                       f"mean: {s['mean']:.3f}")
            txt.append("")
        bs = self.result.get("block_size", 2)
        fails = [(n, rep[n]["fail_ratio"]) for n in METRIC_NAMES
                 if self.enabled[level][n] and metric_is_fail(rep[n], level, self.min_fail, bs)]
        if fails:
            fails.sort(key=lambda x: -x[1])
            txt.append("此層級 FAIL 的 metric (依超標比例排序):")
            txt.append("  " + ",  ".join(f"{n} {r*100:.4f}%" for n, r in fails))
        else:
            txt.append("此層級全部 metric PASS")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", "\n".join(txt))

    # ------------------------------------------------------------------
    @_report_errors("匯出失敗")
    def export(self):
        """只匯出目前選取的序列"""
        if not self.result:
            messagebox.showwarning("尚未載入", "請先載入資料夾")
            return
        out = filedialog.askdirectory(title="選擇輸出資料夾")
        if not out:
            return
        os.makedirs(out, exist_ok=True)
        name = self.result["name"]
        write_csv(os.path.join(out, f"{name}_flicker_kpi_report.csv"),
                  CSV_HEADER, metric_rows(self.result, self.min_fail, self.enabled))
        risk = compute_risk(self.result, self.risk_bounds, enabled=self.enabled)
        imwrite_unicode(os.path.join(out, f"{name}_risk_map.png"),
                        render_risk_map(self.result, risk)[0])
        imwrite_unicode(os.path.join(out, f"{name}_tile_map.png"),
                        render_tile_map(self.result, self._tiles_for(self.result, risk))[0])
        messagebox.showinfo("完成", f"{name} 已輸出到 {out}")
        self.status.config(text=f"已匯出 {name} 至 {out}")

    @_report_errors("批次匯出失敗")
    def batch_export(self):
        """每組序列：分析 -> <name>_risk_map.png + <name>_tile_map.png；全部序列合併 CSV + summary"""
        if not self.seqs:
            messagebox.showwarning("尚未載入", "請先載入資料夾")
            return
        out = filedialog.askdirectory(title="選擇批次輸出資料夾")
        if not out:
            return
        os.makedirs(out, exist_ok=True)

        all_rows, summ_rows, errors = [], [], []
        n = len(self.seqs)
        for i, sq in enumerate(self.seqs):
            self.status.config(text=f"批次 {i+1}/{n}: {sq['name']} ...")
            self.seq_tree.see(str(i))
            self.root.update()
            try:
                r = analyze_sequence(self.folder, thresholds=self.thresholds,
                                     diff_mode=self.diff_mode.get(),
                                     paths=sq["paths"], name=sq["name"],
                                     y_mode=self.y_mode.get())
                risk = compute_risk(r, self.risk_bounds, enabled=self.enabled)
                tiles = self._tiles_for(r, risk)
                imwrite_unicode(os.path.join(out, f"{sq['name']}_risk_map.png"),
                                render_risk_map(r, risk)[0])
                imwrite_unicode(os.path.join(out, f"{sq['name']}_tile_map.png"),
                                render_tile_map(r, tiles)[0])
                sm = summarize_result(r, risk, tiles, self.min_fail, self.enabled)
                all_rows += metric_rows(r, self.min_fail, self.enabled)
                summ_rows.append(summary_row(sm))
                self._set_seq_row(i, sm, "fail" if sm["verdict"] == "FAIL" else "pass")
                self._resort_seqs()
                if i == self.cur_idx:
                    self.result = r
                del r, risk, tiles                      # 逐組釋放，避免大圖序列吃光記憶體
            except Exception as e:
                errors.append(f"{sq['name']}: {e}")
                summ_rows.append([sq["name"], "ERROR"] + [""] * (len(SUMMARY_HEADER) - 2))
                self.seq_tree.item(str(i), tags=("error",),
                                   values=(sq["name"], "ERROR", "", "", "", "", "", ""))

        write_csv(os.path.join(out, "flicker_kpi_report_all.csv"), CSV_HEADER, all_rows)
        write_csv(os.path.join(out, "flicker_kpi_summary.csv"), SUMMARY_HEADER, summ_rows)

        n_fail = sum(1 for r in summ_rows if r[1] == "FAIL")
        msg = (f"{n} 組序列完成: FAIL {n_fail} / PASS {n - n_fail - len(errors)}"
               f" / ERROR {len(errors)}\n輸出: {out}")
        if errors:
            msg += "\n\n錯誤:\n" + "\n".join(errors[:20])
        self.status.config(text=msg.splitlines()[0])
        messagebox.showinfo("批次完成", msg)


# ======================================================================
# Threshold 最佳化視窗
# ======================================================================
ZOOM_FIT = "符合視窗"
ZOOM_LEVELS = [ZOOM_FIT, "25%", "50%", "75%", "100%", "150%", "200%", "400%", "800%"]
LABEL_FILE = "flicker_labels.csv"
LABEL_TXT = {1: "閃", 0: "不閃", None: "未標記"}


class ThresholdOptWindow:
    """
    可加入多個資料夾；每個資料夾加入時給一個預設標記（不閃 / 未標記），
    之後任何一組序列都能手動改成 閃 / 不閃。
    標記存於各資料夾自己的 flicker_labels.csv，加入資料夾時會自動讀回。
    執行：對每個候選 K 求「漏抓 = 0、誤報最少」的 20 個 threshold，結果可套用回主視窗
    """
    SEQ_COLS = ("資料夾", "sequence", "label", "判定", "max S/t", "主要 metric")

    def __init__(self, app):
        self.app = app
        self.win = tk.Toplevel(app.root)
        self.win.title("Threshold 最佳化（漏抓 = 0，誤報最少）")
        self.S = getattr(app, "S", 1.0)
        fit_window(self.win, 1300, 860, self.S, center=False)
        self.folders = []       # [{"path", "default"}]
        self.rows = []          # [{"folder","source","name","paths","label"}]
        self.topk_cache = {}    # (paths, diff_mode, y_mode, block, kmax) -> topk dict
        self.results = []
        self.k_text = tk.StringVar(value=",".join(str(k) for k in kopt.DEFAULT_K_LIST))
        self.tol_text = tk.StringVar(value="0")
        self.fa_per_box_text = tk.StringVar(value="0")
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        w = self.win
        f1 = ttk.LabelFrame(w, text="資料夾（可加入多個；每組序列都能手動改標記）", padding=4)
        f1.pack(fill="x", padx=6, pady=4)
        bar0 = ttk.Frame(f1)
        bar0.pack(fill="x")
        ttk.Button(bar0, text="加入資料夾（預設標為不閃）",
                   command=lambda: self._add_folder(0)).pack(side="left")
        ttk.Button(bar0, text="加入資料夾（預設未標記）",
                   command=lambda: self._add_folder(None)).pack(side="left", padx=4)
        ttk.Button(bar0, text="移除選取的資料夾",
                   command=self._remove_folder).pack(side="left")
        ttk.Button(bar0, text="重新讀取標記檔",
                   command=self._reload_labels).pack(side="left", padx=4)

        lf = ttk.Frame(f1)
        lf.pack(fill="x", pady=2)
        self.flist = tk.Listbox(lf, height=4, selectmode="extended",
                                font=("Consolas", 9))
        fsb = ttk.Scrollbar(lf, orient="vertical", command=self.flist.yview)
        self.flist.configure(yscrollcommand=fsb.set)
        self.flist.pack(side="left", fill="x", expand=True)
        fsb.pack(side="right", fill="y")

        pw = ttk.PanedWindow(w, orient="horizontal")
        pw.pack(fill="both", expand=True, padx=6, pady=4)
        left = ttk.Frame(pw)
        right = ttk.Frame(pw)
        pw.add(left, weight=1)
        pw.add(right, weight=1)

        # ---- 左：序列 + 標記 ----
        bar = ttk.Frame(left)
        bar.pack(fill="x")
        ttk.Button(bar, text="選取的標為「閃」",
                   command=lambda: self._set_label(1)).pack(side="left")
        ttk.Button(bar, text="選取的標為「不閃」",
                   command=lambda: self._set_label(0)).pack(side="left", padx=4)
        ttk.Button(bar, text="取消標記",
                   command=lambda: self._set_label(None)).pack(side="left")
        ttk.Button(bar, text="儲存標記", command=self._save_labels).pack(side="left", padx=4)
        ttk.Label(bar, text="(可多選；雙擊切換閃/不閃)").pack(side="left", padx=6)

        tf = ttk.Frame(left)
        tf.pack(fill="both", expand=True, pady=4)
        self.tree = ttk.Treeview(tf, columns=self.SEQ_COLS, show="headings",
                                 selectmode="extended")
        for c, wd in zip(self.SEQ_COLS, (110, 130, 50, 50, 60, 140)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=int(wd * self.S), anchor="w")
        vsb = ttk.Scrollbar(tf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self._resort_rows = make_sortable(self.tree)
        self.tree.bind("<Double-1>", self._toggle_label)
        self.tree.tag_configure("flick", background="#ffe0e0")
        self.tree.tag_configure("clean", background="#e3f7e3")
        self.tree.tag_configure("unl", background="#f0f0f0")
        self.tree.tag_configure("fa", background="#ff9f9f")      # 誤報
        self.tree.tag_configure("miss", background="#ffcf70")    # 漏抓

        # ---- 右：參數 + 結果 ----
        pf = ttk.LabelFrame(right, text="參數", padding=4)
        pf.pack(fill="x")
        ttk.Label(pf, text="候選 K (pixel 數):").grid(row=0, column=0, sticky="w")
        ttk.Entry(pf, textvariable=self.k_text, width=22).grid(row=0, column=1, sticky="w")
        ttk.Label(pf, text="嚴格模式容忍度 (%):").grid(row=1, column=0, sticky="w")
        ttk.Entry(pf, textvariable=self.tol_text, width=6).grid(row=1, column=1, sticky="w")
        ttk.Label(pf, text="每多抓 1 個框可接受的誤報數:").grid(row=4, column=0, sticky="w")
        ttk.Entry(pf, textvariable=self.fa_per_box_text, width=6).grid(row=4, column=1,
                                                                      sticky="w")
        ttk.Label(pf, text="嚴格: 門檻 = 不閃樣本最大值 x (1+容忍度)，不漏抓優先\n"
                           "平衡: 閃 / 不閃兩側邊界相等\n兩種模式都會算，結果表可直接比較",
                  foreground="#555").grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Button(pf, text="執行最佳化", command=self.run).grid(
            row=5, column=0, columnspan=2, sticky="we", pady=4)

        cols = ("模式", "K", "誤報", "誤報率", "漏抓", "框抓到", "閃最小S/t", "不閃最大S/t",
                "解法")
        self.rtree = ttk.Treeview(right, columns=cols, show="headings", height=11,
                                  selectmode="browse")
        for c, wd in zip(cols, (45, 40, 40, 55, 40, 60, 70, 75, 45)):
            self.rtree.heading(c, text=c)
            self.rtree.column(c, width=int(wd * self.S), anchor="e")
        self.rtree.pack(fill="x", pady=4)
        make_sortable(self.rtree)
        self.rtree.bind("<<TreeviewSelect>>", lambda e: self._show_result())

        ttk.Button(right, text="套用選取結果到主視窗 (threshold + K)",
                   command=self._apply).pack(fill="x")
        self.detail = tk.Text(right, font=("Consolas", 9), height=20)
        self.detail.pack(fill="both", expand=True, pady=4)

        self.status = ttk.Label(w, text="加入資料夾並完成標記後執行；至少需要一組閃與一組不閃",
                                relief="sunken", anchor="w")
        self.status.pack(fill="x", side="bottom")
        for row in (bar0, bar):
            make_wrapping(row)

    # ------------------------------------------------------------------
    # 資料夾
    # ------------------------------------------------------------------
    @_report_errors("加入資料夾失敗")
    def _add_folder(self, default_label):
        d = filedialog.askdirectory(
            parent=self.win,
            title="加入資料夾（預設標為不閃）" if default_label == 0
            else "加入資料夾（預設未標記）")
        if not d:
            return
        d = os.path.normpath(d)
        if any(f["path"] == d for f in self.folders):
            messagebox.showinfo("已加入過", f"{d} 已在清單中", parent=self.win)
            return
        seqs, skipped = discover_sequences(d, self.app.pattern.get(), N_FRAMES)
        if not seqs:
            messagebox.showerror("沒有序列", f"{d} 找不到完整的 <prefix>0001~{N_FRAMES:04d}",
                                 parent=self.win)
            return
        if skipped:
            messagebox.showwarning("部分 prefix 缺幀已略過",
                                   "\n".join(f"{p}: {r}" for p, r in skipped[:20]),
                                   parent=self.win)
        saved = self._load_labels(d)
        rois = load_rois(d)
        self.folders.append({"path": d, "default": default_label})
        name = os.path.basename(d) or d
        self.rows += [{"folder": d, "source": name, "name": s["name"], "paths": s["paths"],
                       "label": saved.get(s["name"], default_label),
                       "boxes": rois.get(s["name"], [])} for s in seqs]
        self._invalidate_results()
        n_saved = sum(1 for s in seqs if s["name"] in saved)
        self.status.config(
            text=f"已加入 {d}（{len(seqs)} 組"
                 + (f"，其中 {n_saved} 組沿用 {LABEL_FILE} 的標記）" if n_saved else "）"))

    def _remove_folder(self):
        sel = [self.folders[i]["path"] for i in self.flist.curselection()]
        if not sel:
            messagebox.showwarning("未選擇", "請先在清單中選取資料夾", parent=self.win)
            return
        self.folders = [f for f in self.folders if f["path"] not in sel]
        self.rows = [r for r in self.rows if r["folder"] not in sel]
        self._invalidate_results()
        self.status.config(text=f"已移除 {len(sel)} 個資料夾")

    @_report_errors("重新讀取標記失敗")
    def _reload_labels(self):
        n = 0
        for f in self.folders:
            saved = self._load_labels(f["path"])
            for r in self.rows:
                if r["folder"] == f["path"] and r["name"] in saved:
                    r["label"] = saved[r["name"]]
                    n += 1
        self._invalidate_results()
        self.status.config(text=f"已從各資料夾的 {LABEL_FILE} 讀回 {n} 筆標記")

    def _invalidate_results(self):
        """標記或資料夾變動 -> 既有最佳化結果失效"""
        self.results = []
        self.rtree.delete(*self.rtree.get_children())
        self.detail.delete("1.0", "end")
        for r in self.rows:
            r["_idx"] = None
        self._refresh_folders()
        self._fill_tree()

    def _refresh_folders(self):
        self.flist.delete(0, "end")
        for f in self.folders:
            rows = [r for r in self.rows if r["folder"] == f["path"]]
            n1 = sum(1 for r in rows if r["label"] == 1)
            n0 = sum(1 for r in rows if r["label"] == 0)
            nu = sum(1 for r in rows if r["label"] is None)
            self.flist.insert("end", f"{len(rows):>3} 組  閃 {n1:>3} / 不閃 {n0:>3} / "
                                     f"未標記 {nu:>3}   {f['path']}")

    # ------------------------------------------------------------------
    # 標記
    # ------------------------------------------------------------------
    def _fill_tree(self, res=None):
        self.tree.delete(*self.tree.get_children())
        for i, r in enumerate(self.rows):
            pred = ratio = metric = ""
            tag = {1: "flick", 0: "clean", None: "unl"}[r["label"]]
            if res is not None and r.get("_idx") is not None:
                k = r["_idx"]
                ratio = f"{res['seq_ratio'][k]:.2f}"
                metric = "{}:{}".format(*kopt.KEYS[int(res["seq_metric"][k])])
                pred = "閃" if res["seq_ratio"][k] > 1 else "不閃"
                if k in res["fa_idx"]:
                    tag = "fa"
                elif k in res["miss_idx"]:
                    tag = "miss"
            self.tree.insert("", "end", iid=str(i), tags=(tag,), values=(
                r["source"], r["name"], LABEL_TXT[r["label"]], pred, ratio, metric))
        self._resort_rows()

    def _set_label(self, lab, iids=None):
        iids = iids if iids is not None else self.tree.selection()
        if not iids:
            return
        for iid in iids:
            r = self.rows[int(iid)]
            r["label"] = lab
            self.tree.set(iid, "label", LABEL_TXT[lab])
            self.tree.item(iid, tags=({1: "flick", 0: "clean", None: "unl"}[lab],))
        self._refresh_folders()

    def _toggle_label(self, ev):
        iid = self.tree.identify_row(ev.y)
        if iid:
            cur = self.rows[int(iid)]["label"]
            self._set_label(0 if cur == 1 else 1, [iid])

    def _load_labels(self, d):
        path = os.path.join(d, LABEL_FILE)
        out = {}
        if os.path.isfile(path):
            with open(path, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    try:
                        lab = int(row["label"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if lab in (0, 1):
                        out[row["sequence"]] = lab
        return out

    @_report_errors("儲存標記失敗")
    def _save_labels(self):
        """每個資料夾各存一份 flicker_labels.csv（只存已標記的序列）"""
        if not self.folders:
            messagebox.showwarning("未加入資料夾", "請先加入資料夾", parent=self.win)
            return
        n_f = n_r = 0
        for f in self.folders:
            rows = [[r["name"], r["label"]] for r in self.rows
                    if r["folder"] == f["path"] and r["label"] is not None]
            if not rows:
                continue
            write_csv(os.path.join(f["path"], LABEL_FILE), ["sequence", "label"], rows)
            n_f += 1
            n_r += len(rows)
        self.status.config(text=f"已儲存 {n_r} 筆標記到 {n_f} 個資料夾的 {LABEL_FILE}"
                                f"  (label: 1=閃, 0=不閃)")

    # ------------------------------------------------------------------
    @_report_errors("最佳化失敗")
    def run(self):
        try:
            k_list = sorted({int(x) for x in self.k_text.get().replace(" ", "").split(",")})
            tol = float(self.tol_text.get()) / 100
            fa_per_box = float(self.fa_per_box_text.get())
            if not k_list or k_list[0] < 1 or tol < 0 or fa_per_box < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("參數錯誤", "K 需為逗號分隔的正整數；容忍度為 >= 0 的數字 (%)",
                                 parent=self.win)
            return
        use = [r for r in self.rows if r["label"] is not None]
        n_unl = len(self.rows) - len(use)
        if not any(r["label"] == 0 for r in use) or not any(r["label"] == 1 for r in use):
            messagebox.showerror("標記不足", "需要至少一組「不閃」與一組「閃」", parent=self.win)
            return
        if n_unl and not messagebox.askyesno(
                "有未標記序列", f"{n_unl} 組未標記，將不納入計算。繼續？", parent=self.win):
            return

        diff_mode = self.app.diff_mode.get()
        y_mode = self.app.y_mode.get()
        bs = 2
        kmax = max(k_list)
        topks, box_topks, box_rows = [], [], []
        for i, r in enumerate(use):
            boxes = tuple(r["boxes"]) if r["label"] == 1 else ()     # 框只用在閃序列
            key = (tuple(r["paths"]), diff_mode, y_mode, bs, kmax, boxes)
            if key not in self.topk_cache:
                self.status.config(text=f"計算 metric {i+1}/{len(use)}: {r['source']} {r['name']}")
                self.win.update()
                self.topk_cache[key] = kopt.sequence_topk(r["paths"], kmax, diff_mode, bs,
                                                          y_mode, boxes)
            tk_ = self.topk_cache[key]
            topks.append(tk_)
            if boxes:
                box_topks.append(tk_)
                box_rows += [(r, b) for b in boxes]
        self.box_rows = box_rows            # [(序列 row, box)]，順序與 B 的列一致
        for r in self.rows:
            r["_idx"] = None
        for k, r in enumerate(use):
            r["_idx"] = k
        labels = [r["label"] == 1 for r in use]

        self.status.config(text="最佳化中...")
        self.win.update()
        active = kopt.active_mask(self.app.enabled)
        if not active.any():
            messagebox.showerror("沒有啟用的 metric", "請先在主視窗啟用至少一項 metric",
                                 parent=self.win)
            return
        self.results = kopt.sweep_k(topks, labels, k_list, bs, tol=tol, active=active,
                                    box_topks=box_topks or None, fa_per_box=fa_per_box)
        self.n_active = int(active.sum())
        self.used = use
        n_clean = sum(1 for l in labels if not l)
        self.rtree.delete(*self.rtree.get_children())
        for i, res in enumerate(self.results):
            self.rtree.insert("", "end", iid=str(i), values=(
                "嚴格" if res["mode"] == "strict" else "平衡",
                res["K"], len(res["fa_idx"]), f"{len(res['fa_idx'])/n_clean*100:.1f}%",
                len(res["miss_idx"]),
                f"{res['n_box_caught']}/{res['n_box']}" if res["n_box"] else "-",
                f"{res['flick_min']:.3f}", f"{res['clean_max']:.3f}",
                "精確" if res["exact"] else "近似"))
        self.rtree.selection_set("0")                  # 嚴格模式的最佳結果
        best = self.results[0]
        self.status.config(
            text=f"完成：{len(use)} 組 (閃 {sum(labels)} / 不閃 {n_clean})；"
                 f"啟用 {self.n_active}/20 項 metric；"
                 + (f"框 {best['n_box_caught']}/{best['n_box']}；" if best["n_box"] else "")
                 + f"嚴格模式最佳 K={best['K']}，誤報 {len(best['fa_idx'])}，"
                   f"漏抓 {len(best['miss_idx'])}")

    def _show_result(self):
        sel = self.rtree.selection()
        if not sel or not self.results:
            return
        res = self.results[int(sel[0])]
        self._fill_tree(res)
        use = self.used
        cur = self.app.thresholds
        if res["mode"] == "strict":
            mode_txt = f"嚴格 (容忍度 {res['tol']*100:g}%"
            if res["tol_applied"] is not None and res["tol_applied"] < res["tol"] - 1e-12:
                mode_txt += f"，為避免漏抓自動降為 {res['tol_applied']*100:.2f}%"
            mode_txt += ")"
        else:
            mode_txt = f"平衡 (分離度 {res['separation']:.3f})"
        lines = [f"模式: {mode_txt}",
                 f"K = {res['K']}  (block: {k_for_level(res['K'], 'block')} 個 block)   "
                 f"解法: {'精確 (MILP)' if res['exact'] else '近似 (greedy)'}",
                 f"誤報 {len(res['fa_idx'])}   漏抓 {len(res['miss_idx'])}   "
                 f"閃序列最小 S/t {res['flick_min']:.3f}   "
                 f"不閃序列最大 S/t {res['clean_max']:.3f}",
                 (f"人工框: 抓到 {res['n_box_caught']}/{res['n_box']}"
                  f"   (每多抓 1 框可接受誤報 {res['fa_per_box']:g})"
                  if res["n_box"] else "人工框: 無"), "",
                 f"{'metric':<22}{'目前':>10}{'最佳化':>12}"]
        for j, (lv, m) in enumerate(kopt.KEYS):
            if not res["active"][j]:
                lines.append(f"{lv+':'+m:<22}{cur[lv][m]:>10.2f}{'(停用)':>12}")
                continue
            mark = " *" if j in res["lowered"] else ""
            lines.append(f"{lv+':'+m:<22}{cur[lv][m]:>10.2f}{res['thresholds'][j]:>12.2f}{mark}")
        lines.append("  (* = 造成誤報的 metric；(停用) = 主視窗關閉、未參與最佳化)")
        lines.append("")
        if res["fa_idx"]:
            lines.append("誤報的不閃序列:")
            lines += [f"  {use[k]['source']} {use[k]['name']}  S/t={res['seq_ratio'][k]:.2f}"
                      for k in res["fa_idx"]]
        ci = [k for k, r in enumerate(use) if r["label"] == 0 and k not in res["fa_idx"]]
        fi = [k for k, r in enumerate(use) if r["label"] == 1]
        if ci:
            k = max(ci, key=lambda k: res["seq_ratio"][k])
            lines.append(f"最接近門檻的不閃序列: {use[k]['name']}  S/t={res['seq_ratio'][k]:.3f}")
        if fi:
            k = min(fi, key=lambda k: res["seq_ratio"][k])
            lines.append(f"最接近門檻的閃序列  : {use[k]['name']}  S/t={res['seq_ratio'][k]:.3f}")
        if res["n_box"] and res["n_box_caught"] < res["n_box"]:
            lines.append("")
            lines.append("漏掉的框 (位置 / 最接近的 metric / S/t):")
            for b in np.flatnonzero(~res["box_caught"]):
                r0, bx = self.box_rows[b]
                pos = f"x {bx[0]}~{bx[2]-1}, y {bx[1]}~{bx[3]-1}"
                lines.append(f"  {r0['source']} {r0['name']}  {pos}  "
                             f"{'{}:{}'.format(*kopt.KEYS[int(res['box_metric'][b])])}  "
                             f"S/t={res['box_ratio'][b]:.3f}")
        lines.append("")
        lines.append("S/t > 1 判閃。閃最小 S/t 越大 -> 閃的抓得越穩；不閃最大 S/t 越小 -> 越不易誤報")
        lines.append("S = 該 metric 第 K 大的值 (即超標位置數 >= K 的判定量)")
        self.detail.delete("1.0", "end")
        self.detail.insert("1.0", "\n".join(lines))

    def _apply(self):
        sel = self.rtree.selection()
        if not sel or not self.results:
            messagebox.showwarning("無結果", "請先執行最佳化並選一個結果", parent=self.win)
            return
        res = self.results[int(sel[0])]
        self.app.set_thresholds_and_k(kopt.thresholds_to_dict(res["thresholds"]), res["K"])
        self.status.config(text=f"已套用 {'嚴格' if res['mode'] == 'strict' else '平衡'}模式 "
                                f"K={res['K']} 與 20 個 threshold 到主視窗")
        self._show_result()


if __name__ == "__main__":
    enable_dpi_awareness()
    root = tk.Tk()
    FlickerKPIApp(root)
    root.mainloop()
