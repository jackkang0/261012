"""
Flicker KPI threshold 最佳化
============================
輸入兩類已標記的序列：
  - 不閃 (label 0)：不閃 folder 的全部序列 + 混合 folder 中標為不閃者
  - 閃   (label 1)：混合 folder 中標為閃者

判定規則（序列層級）:
  某 metric m 有 >= K 個位置 value > t_m  ->  該序列判為「閃」；20 項 metric 任一成立即閃 (OR)
  等價於: s_m(K) = 該 metric map 第 K 大的值,  閃 <=> 存在 m 使 s_m(K) > t_m
  （超標幅度越大、超標位置越多，s_m(K) 越大 -> 越容易判閃）
  K 以 pixel 計；block-based 換算為 ceil(K / block_size^2) 個 block（面積一致）。

人工框 (ROI): 標在閃序列上的「這裡必須抓到」位置。框只當次要目標，不影響硬條件；
  框內統計量 s_box,m = 框內第 K 大的值，K 取 min(K, 框內位置數)，所以小框也可能被抓到。
  目標順序: (1) 每組閃序列都抓到  (2) 誤報最少  (3) 抓到的框最多  (4) 邊界最大
  fa_per_box > 0 時，每多抓一個框可接受 fa_per_box 組誤報（預設 0 = 框不影響誤報數）。

最佳化目標: 漏抓 = 0 的前提下，誤報（不閃序列被判閃）最少。
  1. 每個 metric 先把門檻放在所有不閃序列之上 (t_m > max_clean_m) -> 誤報 0
  2. 仍未被抓到的閃序列，需把某些 metric 的門檻降到其 s 值以下；每個「降門檻」選項
     會連帶誤報一些不閃序列。求「覆蓋全部未抓閃序列、且誤報序列聯集最小」:
       - scipy.optimize.milp 可用時解 0/1 整數規劃（精確解）
       - 否則用 greedy set cover + 反向刪除（近似解）
  3. 誤報集合確定後，以 max-min margin 決定門檻位置（log 域）:
       max δ  s.t.  每個未誤報的不閃序列  s_im <= t_m e^-δ  (所有 m)
                    每個閃序列            存在 m 使 s_jm >= t_m e^+δ
     令 t_m = max_clean_m · e^δ，可行性對 δ 單調 -> 二分搜尋。
     分離度 = e^δ*：所有閃序列 S/t >= e^δ*，所有未誤報不閃序列 S/t <= e^-δ*。
  3'. 嚴格模式（不漏抓優先）: t_m = max_clean_m · (1 + tol)
     每個 metric 貼著未誤報不閃序列的上緣，任何一項稍微超出不閃樣本就觸發。
     tol 若會讓某個已知閃序列漏抓，自動降到仍可全部抓到的最大值（回報 tol_applied）。
對每個候選 K 各解一次，取誤報最少者；
  同誤報數時：平衡模式取分離度最大者，嚴格模式取「閃序列最小 S/t」最大者。
"""

import math

import numpy as np

from flicker_kpi_core import (
    METRIC_NAMES, load_frames, compute_metric_maps, k_for_level, DEFAULT_Y_MODE,
)

LEVELS = ("pixel", "block")
KEYS = [(lv, m) for lv in LEVELS for m in METRIC_NAMES]      # 20 項，固定順序
DEFAULT_K_LIST = (1, 4, 16, 64, 256)


# --------------------------------------------------------------------------
# 1. 每組序列只需保留每個 metric 的前 Kmax 大值
# --------------------------------------------------------------------------
def _topk(flat, k):
    kk = min(int(k), flat.size)
    if kk <= 0:
        return np.zeros(0, np.float32)
    top = np.partition(flat, flat.size - kk)[flat.size - kk:]
    return np.sort(top)[::-1].astype(np.float32)


def sequence_topk(paths, kmax, diff_mode="peak_to_peak", block_size=2,
                  y_mode=DEFAULT_Y_MODE, boxes=()):
    """
    讀一組序列，回傳 {(level, metric): 由大到小的前 k 個值 (float32)}。
    只存 top-k，不保留整張 map（批次掃大量 1080p 序列時省記憶體）。
    boxes: [(x0,y0,x1,y1)]；有給時另外回傳 out["_boxes"] = [{"box", "n", "vals"}]
      n: {level: 該框在該層級涵蓋的位置數}
    """
    stack = load_frames(paths)
    out = {}
    box_acc = [{"box": b, "n": {}, "vals": {}} for b in boxes]
    for lv, blk in (("pixel", 1), ("block", block_size)):
        k = k_for_level(kmax, lv, block_size)
        for m, arr in compute_metric_maps(stack, block=blk, diff_mode=diff_mode,
                                          y_mode=y_mode).items():
            out[(lv, m)] = _topk(arr.ravel(), k)
            for acc in box_acc:
                x0, y0, x1, y1 = acc["box"]
                if lv == "block":
                    x0, y0 = x0 // blk, y0 // blk
                    x1, y1 = -(-x1 // blk), -(-y1 // blk)       # 覆蓋到的 block
                sub = arr[y0:min(y1, arr.shape[0]), x0:min(x1, arr.shape[1])]
                acc["n"][lv] = int(sub.size)
                acc["vals"][(lv, m)] = _topk(sub.ravel(), k) if sub.size else np.zeros(0, np.float32)
    if boxes:
        out["_boxes"] = box_acc
    return out


def box_stat_matrix(topks, K, block_size=2):
    """
    回傳 (B, owner)：B (n_box, 20) 每個框在每項 metric 的第 K 大值
    （K 取 min(K, 框內位置數)），owner[i] = 該框屬於 topks 的哪一個 index
    """
    rows, owner = [], []
    for si, tk in enumerate(topks):
        for acc in tk.get("_boxes", []):
            v = np.zeros(len(KEYS))
            for j, key in enumerate(KEYS):
                k = min(k_for_level(K, key[0], block_size), max(acc["n"][key[0]], 1))
                vals = acc["vals"][key]
                v[j] = vals[k - 1] if len(vals) >= k else 0.0
            rows.append(v)
            owner.append(si)
    return (np.array(rows) if rows else np.zeros((0, len(KEYS)))), np.array(owner, int)


def stat_matrix(topks, K, block_size=2):
    """topks: [dict] -> S (n_seq, 20)，S[i, m] = 第 K 大值（不足 K 個位置時為 0）"""
    S = np.zeros((len(topks), len(KEYS)), np.float64)
    for i, tk in enumerate(topks):
        for j, key in enumerate(KEYS):
            k = k_for_level(K, key[0], block_size)
            v = tk[key]
            S[i, j] = v[k - 1] if len(v) >= k else 0.0
    return S


# --------------------------------------------------------------------------
# 2. 最佳化
# --------------------------------------------------------------------------
def _gap_threshold(lo, hi):
    """lo < t < hi，取幾何中點（metric 皆非負；lo = 0 時取 hi/2）"""
    if hi <= 0:
        return -1e-6                    # 所有值皆 0 時仍需 > 判定成立
    if lo <= 0:
        return hi / 2.0
    return math.sqrt(lo * hi)


def _solve_cover(options, U, n_clean, n_box=0, free_box=None, fa_per_box=0.0):
    """
    options: [(metric_idx, cut, cover_set, fa_set, box_set)]
    求選一組 options：覆蓋 U（硬條件）、|∪fa| 最小（主要）、抓到的框最多（次要）。
    free_box: 不必降門檻就會被抓到的框 index 集合
    fa_per_box: 每多抓一個框可接受幾組誤報（0 = 框不影響誤報數，只當同分時的取捨）
    回傳 (選中 index list, exact: bool)
    """
    free_box = free_box or set()
    if not U and not n_box:
        return [], True
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import lil_matrix
        nO = len(options)
        nV = nO + n_clean + n_box
        # 框的權重：給 0 時取夠小的值，使抓到全部的框也抵不過一組誤報（字典序）
        lam = fa_per_box if fa_per_box > 0 else 0.5 / (n_box + 1)
        eps = lam / (10 * (nO + 1))                      # 再次要: 少降門檻
        c = np.concatenate([np.full(nO, eps), np.ones(n_clean), np.full(n_box, -lam)])
        U_list = sorted(U)
        n_rows = (len(U_list) + sum(len(o[3]) for o in options) + n_box)
        A = lil_matrix((n_rows, nV))
        lb, ub = [], []
        r = 0
        for j in U_list:                                  # 覆蓋: Σ x_o >= 1
            for oi, o in enumerate(options):
                if j in o[2]:
                    A[r, oi] = 1
            lb.append(1); ub.append(np.inf); r += 1
        for oi, o in enumerate(options):                  # 誤報: y_i - x_o >= 0
            for i in o[3]:
                A[r, oi] = -1
                A[r, nO + i] = 1
                lb.append(0); ub.append(np.inf); r += 1
        for b in range(n_box):                            # 框: z_b <= Σ x_o + free_b
            A[r, nO + n_clean + b] = -1
            if b not in free_box:
                for oi, o in enumerate(options):
                    if b in o[4]:
                        A[r, oi] = 1
                lb.append(0)
            else:
                lb.append(-1)                             # 免費抓到 -> z_b 不受限
            ub.append(np.inf); r += 1
        res = milp(c, constraints=LinearConstraint(A.tocsr(), lb, ub),
                   integrality=np.ones(nV), bounds=Bounds(0, 1))
        if res.success:
            return [oi for oi in range(nO) if res.x[oi] > 0.5], True
    except ImportError:
        pass

    # greedy set cover（以新增誤報數為成本）+ 反向刪除
    chosen, covered, fa = [], set(), set()
    while covered != U:
        best, best_key = None, None
        for oi, (_, _, cov, f, _bx) in enumerate(options):
            new_cov = len((cov & U) - covered)
            if new_cov == 0:
                continue
            new_fa = len(f - fa)
            key = (new_fa / new_cov, -new_cov)
            if best_key is None or key < best_key:
                best, best_key = oi, key
        if best is None:
            break
        chosen.append(best)
        covered |= options[best][2] & U
        fa |= options[best][3]
    for oi in list(reversed(chosen)):
        rest = [o for o in chosen if o != oi]
        cov = set().union(*(options[o][2] for o in rest)) if rest else set()
        if U <= cov:
            chosen = rest
    # 次要目標：再撿「不增加誤報」就能多抓到的框
    got = set(free_box) | set().union(*(options[o][4] for o in chosen)) if chosen else set(free_box)
    for oi, o in enumerate(options):
        if oi in chosen or not (o[4] - got):
            continue
        if not (o[3] - fa):                      # 不產生新的誤報
            chosen.append(oi)
            fa |= o[3]
            got |= o[4]
    return chosen, False


def active_mask(enabled):
    """{level: {metric: bool}} -> (20,) bool，對應 KEYS 的順序"""
    if enabled is None:
        return np.ones(len(KEYS), bool)
    return np.array([bool(enabled.get(lv, {}).get(m, True)) for lv, m in KEYS])


def optimize_thresholds(S, labels, mode="strict", tol=0.0, active=None,
                        B=None, fa_per_box=0.0):
    """
    S: (n, 20) 序列統計量；labels: (n,) bool，True = 閃
    active: (20,) bool，False 的 metric 不參與（門檻回傳 inf，表示停用）
    B: (n_box, 20) 人工框的統計量；框為次要目標，不影響硬條件
    fa_per_box: 每多抓一個框可接受幾組誤報（0 = 僅在誤報數相同時用來取捨）
    回傳 dict:
      thresholds (20,), fa_idx, miss_idx, exact, separation,
      seq_ratio (n,)  = max_m S/t（>1 判閃）, seq_metric (n,) = 造成該最大值的 metric index
    """
    labels = np.asarray(labels, bool)
    if active is not None:
        act = np.asarray(active, bool)
        if not act.any():
            raise ValueError("至少需要啟用一項 metric")
        sub = optimize_thresholds(S[:, act], labels, mode, tol,
                                  B=None if B is None else B[:, act],
                                  fa_per_box=fa_per_box)
        full = np.full(S.shape[1], np.inf)
        full[act] = sub["thresholds"]
        idx = np.flatnonzero(act)
        sub["thresholds"] = full
        sub["seq_metric"] = idx[sub["seq_metric"]]
        sub["lowered"] = [int(idx[m]) for m in sub["lowered"]]
        if len(sub["box_metric"]):
            sub["box_metric"] = idx[sub["box_metric"]]
        sub["active"] = act
        return sub
    ci, fi = np.flatnonzero(~labels), np.flatnonzero(labels)
    if len(ci) == 0:
        raise ValueError("需要至少一組不閃序列")
    C, F = S[ci], S[fi]
    maxc = C.max(axis=0)

    covered0 = (F > maxc).any(axis=1) if len(fi) else np.zeros(0, bool)
    U = set(np.flatnonzero(~covered0).tolist())               # index into fi

    Bx = np.zeros((0, S.shape[1])) if B is None else np.asarray(B, float)
    free_box = {b for b in range(len(Bx)) if (Bx[b] > maxc).any()}
    pending_box = [b for b in range(len(Bx)) if b not in free_box]

    options = []
    for m in range(S.shape[1]):
        cuts = {float(F[j, m]) for j in U} | {float(Bx[b, m]) for b in pending_box}
        for cut in sorted(cuts, reverse=True):
            if cut <= 0:
                continue
            cov = {j for j in U if F[j, m] >= cut}
            bx = {b for b in pending_box if Bx[b, m] >= cut}
            if not cov and not bx:
                continue
            fa = set(np.flatnonzero(C[:, m] >= cut).tolist())  # index into ci
            options.append((m, cut, cov, fa, bx))
    chosen, exact = _solve_cover(options, U, len(ci), len(Bx), free_box, fa_per_box)

    fa_set = set()
    for oi in chosen:
        fa_set |= options[oi][3]
    keep = [k for k in range(len(ci)) if k not in fa_set]      # 未誤報的不閃序列
    maxc2 = C[keep].max(axis=0) if keep else np.zeros(S.shape[1])
    base = np.maximum(maxc2, 1e-6)                             # 避免 log(0)

    # 門檻放在 base 時會抓到的框 -> 之後拉高邊界時不能把它們弄丟
    target_box = np.flatnonzero((Bx > base).any(axis=1)) if len(Bx) else np.zeros(0, int)

    def keeps_boxes(th):
        return True if not len(target_box) else bool((Bx[target_box] > th).any(axis=1).all())

    tol_applied = None
    if mode == "strict":
        # 閃序列需 F > t；tol 過大會漏抓時，二分搜尋仍可全抓的最大 tol
        def covers(d):
            th = base * math.exp(d)
            ok = bool((F > th).any(axis=1).all()) if len(fi) else True
            return ok and keeps_boxes(th)
        d_want = math.log1p(max(tol, 0.0))
        if covers(d_want):
            d_lo = d_want
        else:
            d_lo, d_hi = 0.0, d_want
            for _ in range(60):
                mid = (d_lo + d_hi) / 2
                if covers(mid):
                    d_lo = mid
                else:
                    d_hi = mid
        tol_applied = math.expm1(d_lo)
        t = base * math.exp(d_lo)
    else:
        def feasible(d):
            th = base * math.exp(2 * d)
            ok = bool((F >= th).any(axis=1).all())
            return ok and keeps_boxes(base * math.exp(d))

        with np.errstate(divide="ignore"):
            d_hi = float(np.max(np.log(np.maximum(F, 1e-12) / base)) / 2) if len(fi) else 0.0
        d_lo = 0.0
        if d_hi > 0 and feasible(d_hi):
            d_lo = d_hi
        else:
            for _ in range(60):
                mid = (d_lo + d_hi) / 2
                if feasible(mid):
                    d_lo = mid
                else:
                    d_hi = mid
        t = base * math.exp(d_lo)

    # 門檻無條件進位到小數 4 位（GUI 顯示/輸入用），並以主視窗相同的 float32 比較驗證
    # 判定不變；metric map 為 float32，判定式為 value > threshold
    S32 = S.astype(np.float32)

    def predict(th):
        return (S32 > np.asarray(th, np.float64).astype(np.float32)).any(axis=1)

    tq = np.ceil(t * 1e4) / 1e4
    if np.array_equal(predict(tq), predict(t)):
        t = tq

    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(t > 0, S / t, np.where(S > t, np.inf, 0.0))
    seq_ratio = R.max(axis=1)
    seq_metric = R.argmax(axis=1)
    pred = predict(t)
    fa_idx = ci[pred[ci]].tolist()
    miss_idx = fi[~pred[fi]].tolist()

    ok_clean = [i for i in ci if not pred[i]]
    sep_c = 1 / seq_ratio[ok_clean].max() if ok_clean and seq_ratio[ok_clean].max() > 0 else np.inf
    sep_f = seq_ratio[fi].min() if len(fi) else np.inf
    if len(Bx):
        with np.errstate(divide="ignore", invalid="ignore"):
            RB = np.where(t > 0, Bx / t, np.where(Bx > t, np.inf, 0.0))
        box_ratio = RB.max(axis=1)
        box_metric = RB.argmax(axis=1)
        box_caught = (Bx.astype(np.float32) >
                      np.asarray(t, np.float64).astype(np.float32)).any(axis=1)
    else:
        box_ratio = np.zeros(0)
        box_metric = np.zeros(0, int)
        box_caught = np.zeros(0, bool)

    # 造成誤報的 metric（誤報序列在該 metric 超過門檻）
    fa_metrics = sorted({m for i in fa_idx for m in np.flatnonzero(S[i] > t)})
    return {"thresholds": t, "fa_idx": fa_idx, "miss_idx": miss_idx, "exact": exact,
            "separation": float(min(sep_c, sep_f)), "seq_ratio": seq_ratio,
            "seq_metric": seq_metric, "lowered": fa_metrics,
            "flick_min": float(sep_f),                                  # 閃序列最小 S/t
            "clean_max": float(1 / sep_c) if np.isfinite(sep_c) else 0.0,  # 未誤報不閃最大 S/t
            "mode": mode, "tol": float(tol), "tol_applied": tol_applied,
            "active": np.ones(S.shape[1], bool),
            "box_caught": box_caught, "box_ratio": box_ratio, "box_metric": box_metric,
            "n_box": int(len(Bx)), "n_box_caught": int(box_caught.sum()),
            "fa_per_box": float(fa_per_box)}


def sweep_k(topks, labels, k_list=DEFAULT_K_LIST, block_size=2,
            modes=("strict", "balanced"), tol=0.0, active=None,
            box_topks=None, fa_per_box=0.0):
    """
    每個 (模式, K) 各最佳化一次。回傳依模式分組、組內依
    (誤報數, 漏抓數, -閃序列最小 S/t [嚴格] / -分離度 [平衡]) 排序的結果 list
    """
    results = []
    for mode in modes:
        group = []
        for K in k_list:
            S = stat_matrix(topks, K, block_size)
            B, owner = (box_stat_matrix(box_topks, K, block_size)
                        if box_topks else (None, np.zeros(0, int)))
            r = optimize_thresholds(S, labels, mode, tol, active, B, fa_per_box)
            r["box_owner"] = owner
            r["K"] = int(K)
            r["S"] = S
            group.append(r)
        score = (lambda r: r["flick_min"]) if mode == "strict" else (lambda r: r["separation"])
        group.sort(key=lambda r: (len(r["fa_idx"]), len(r["miss_idx"]),
                                  -r["n_box_caught"], -score(r)))
        results += group
    return results


def thresholds_to_dict(t):
    """inf（停用的 metric）不列入，呼叫端沿用原值"""
    out = {"pixel": {}, "block": {}}
    for (lv, m), v in zip(KEYS, t):
        if np.isfinite(v):
            out[lv][m] = float(v)
    return out


# ==========================================================================
# 框層級最佳化（獨立於序列層級的目標）
# ==========================================================================
# 目標：框內盡量被判成閃、框外盡量不要被判成閃。
#   正樣本：人工框（框內第 K_box 大的值 > 門檻 -> 這個框被抓到）
#   負樣本：閃序列的框外區域 + 不閃序列整張圖，以 tile x tile 的格子為單位，
#           同一格用同樣的規則判定（格內第 K_box 大的值 > 門檻 -> 這一格被誤判）
#   取捨：對每個「容許誤判格數」budget，求框抓到數最大 -> 得到一條取捨曲線
DEFAULT_ROI_BUDGETS = (0, 1, 5, 10, 50)
DEFAULT_K_BOX = 1


def _tile_kth(arr, tile, k):
    """(H,W) -> (格數,) 每格第 k 大的值；不足一格的邊緣捨去"""
    H, W = arr.shape
    th, tw = H // tile, W // tile
    if th == 0 or tw == 0:
        return np.zeros(0, np.float32), (0, 0)
    sub = arr[:th * tile, :tw * tile].reshape(th, tile, tw, tile)
    flat = sub.transpose(0, 2, 1, 3).reshape(th * tw, tile * tile)
    kk = min(int(k), flat.shape[1])
    part = np.partition(flat, flat.shape[1] - kk, axis=1)[:, flat.shape[1] - kk:]
    return part.min(axis=1).astype(np.float32), (th, tw)


def sequence_roi_stats(paths, boxes, tile=120, k_box=DEFAULT_K_BOX,
                       diff_mode="peak_to_peak", block_size=2, y_mode=DEFAULT_Y_MODE):
    """
    回傳 (box_stats, neg_stats)
      box_stats : (n_box, 20)  每個框第 K_box 大的值（K 取 min(K_box, 框內位置數)）
      neg_stats : (n_tile, 20) 與任何框都不重疊的格子，每格第 K_box 大的值
    """
    stack = load_frames(paths)
    H, W = stack.shape[1:3]
    th, tw = H // tile, W // tile
    keep = np.ones(th * tw, bool)                     # 與框重疊的格子不當負樣本
    for (x0, y0, x1, y1) in boxes:
        for r in range(max(0, y0 // tile), min(th, -(-y1 // tile))):
            for c in range(max(0, x0 // tile), min(tw, -(-x1 // tile))):
                keep[r * tw + c] = False

    box_rows = np.zeros((len(boxes), len(KEYS)))
    neg_rows = np.zeros((int(keep.sum()), len(KEYS)))
    for li, (lv, blk) in enumerate((("pixel", 1), ("block", block_size))):
        kk = k_for_level(k_box, lv, block_size)
        maps = compute_metric_maps(stack, block=blk, diff_mode=diff_mode, y_mode=y_mode)
        for mi, m in enumerate(METRIC_NAMES):
            col = li * len(METRIC_NAMES) + mi
            arr = maps[m]
            for bi, (x0, y0, x1, y1) in enumerate(boxes):
                if blk > 1:
                    sx0, sy0 = x0 // blk, y0 // blk
                    sx1, sy1 = -(-x1 // blk), -(-y1 // blk)
                else:
                    sx0, sy0, sx1, sy1 = x0, y0, x1, y1
                sub = arr[sy0:min(sy1, arr.shape[0]), sx0:min(sx1, arr.shape[1])]
                v = _topk(sub.ravel(), min(kk, sub.size)) if sub.size else None
                box_rows[bi, col] = v[-1] if v is not None and len(v) else 0.0
            vals, _ = _tile_kth(arr, max(1, tile // blk), kk)
            n = min(len(vals), len(keep))
            neg_rows[:, col] = vals[:n][keep[:n]]
    return box_rows, neg_rows


def _roi_counts(B, N, t):
    """以 float32 比較（與主視窗一致）回傳 (框抓到 bool, 負樣本格被判閃 bool)"""
    t32 = np.asarray(t, np.float64).astype(np.float32)
    caught = (B.astype(np.float32) > t32).any(axis=1) if len(B) else np.zeros(0, bool)
    flagged = (N.astype(np.float32) > t32).any(axis=1) if len(N) else np.zeros(0, bool)
    return caught, flagged


def optimize_roi(B, N, budgets=DEFAULT_ROI_BUDGETS, active=None):
    """
    對每個容許誤判格數 budget，求「框抓到數最大」的 20 個門檻。
    回傳 list，每個 budget 一筆結果。
    """
    B = np.asarray(B, float)
    N = np.asarray(N, float)
    nM = B.shape[1] if len(B) else (N.shape[1] if len(N) else len(KEYS))
    act = np.ones(nM, bool) if active is None else np.asarray(active, bool)
    if not act.any():
        raise ValueError("至少需要啟用一項 metric")

    # 候選門檻：只需放在某個框的值之下（更低只會多誤判、不會多抓框）
    cand = {}
    for m in np.flatnonzero(act):
        vals = sorted({float(v) for v in B[:, m] if v > 0}, reverse=True)
        cand[m] = vals

    # 沒被選中的 metric 放在「負樣本最大值」上：不會增加誤判，
    # 若有框高於它則免費抓到；停用的 metric 維持 inf（不參與）
    base = np.full(nM, np.inf)
    for m in np.flatnonzero(act):
        mx = float(N[:, m].max()) if len(N) else 0.0
        base[m] = mx if mx > 0 else (float(B[:, m].max()) / 2 if len(B) else np.inf)
    results = []
    for budget in budgets:
        chosen = _roi_solve(B, N, cand, act, int(budget), nM)
        exact = bool(chosen.pop("_exact", True))
        t = np.array(base)
        for m, cut in chosen.items():
            below = np.concatenate([B[:, m], N[:, m]]) if len(N) else B[:, m]
            below = below[below < cut]
            t[m] = min(t[m], _gap_threshold(float(below.max()) if below.size else 0.0, cut))
        caught, flagged = _roi_counts(B, N, t)
        results.append({
            "mode": "roi", "budget": int(budget), "thresholds": t,
            "box_caught": caught, "n_box": int(len(B)),
            "n_box_caught": int(caught.sum()),
            "n_neg": int(len(N)), "n_fp_tile": int(flagged.sum()),
            "fp_mask": flagged, "active": act, "exact": exact,
            "lowered": sorted(chosen),
        })
    return results


def _roi_solve(B, N, cand, act, budget, nM):
    """回傳 {metric: cut}；門檻放在 cut 之下即可抓到所有 >= cut 的框"""
    metrics = [m for m in cand if cand[m]]
    if not metrics:
        return {}
    # 每個 (metric, cut) 的效果：抓到哪些框、誤判哪些格
    opts = []
    for m in metrics:
        for cut in cand[m]:
            bx = frozenset(np.flatnonzero(B[:, m] >= cut).tolist())
            fp = frozenset(np.flatnonzero(N[:, m] >= cut).tolist()) if len(N) else frozenset()
            opts.append((m, cut, bx, fp))
    exact = True
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import lil_matrix
        nO, nB, nF = len(opts), B.shape[0], (N.shape[0] if len(N) else 0)
        if nO * (nB + nF) > 4_000_000:
            raise MemoryError
        nV = nO + nB + nF
        c = np.concatenate([np.full(nO, 1e-6), np.full(nB, -1.0), np.zeros(nF)])
        rows = sum(len(o[2]) for o in opts) + sum(len(o[3]) for o in opts)
        A = lil_matrix((nB + rows + 1, nV))
        lb, ub = [], []
        r = 0
        for b in range(nB):                       # z_b <= Σ x_o（o 能抓到 b）
            A[r, nO + b] = -1
            for oi, o in enumerate(opts):
                if b in o[2]:
                    A[r, oi] = 1
            lb.append(0); ub.append(np.inf); r += 1
        for oi, o in enumerate(opts):             # w_g >= x_o（o 會誤判 g）
            for gidx in o[3]:
                A[r, oi] = -1
                A[r, nO + nB + gidx] = 1
                lb.append(0); ub.append(np.inf); r += 1
        for g in range(nF):                       # Σ w_g <= budget
            A[r, nO + nB + g] = 1
        lb.append(0); ub.append(float(budget)); r += 1
        res = milp(c, constraints=LinearConstraint(A.tocsr()[:r], lb, ub),
                   integrality=np.ones(nV), bounds=Bounds(0, 1))
        if res.success:
            out = {}
            for oi, (m, cut, _, _) in enumerate(opts):
                if res.x[oi] > 0.5:
                    out[m] = min(cut, out.get(m, np.inf))
            out["_exact"] = True
            return out
        exact = False
    except (ImportError, MemoryError):
        exact = False

    # greedy：每次挑「新抓到的框 / 新誤判的格」最高者，直到用完 budget
    got_b, got_f, chosen = set(), set(), {}
    while True:
        best, best_key = None, None
        for oi, (m, cut, bx, fp) in enumerate(opts):
            nb = len(bx - got_b)
            if nb == 0:
                continue
            nf = len(fp - got_f)
            if len(got_f | fp) > budget:
                continue
            key = (-(nb / (nf + 0.5)), nf)
            if best_key is None or key < best_key:
                best, best_key = oi, key
        if best is None:
            break
        m, cut, bx, fp = opts[best]
        chosen[m] = min(cut, chosen.get(m, np.inf))
        got_b |= bx
        got_f |= fp
    chosen["_exact"] = exact
    return chosen
