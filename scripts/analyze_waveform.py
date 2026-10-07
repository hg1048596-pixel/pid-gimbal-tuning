#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VOFA+ 波形判读工具 —— 把 pid-gimbal-tuning skill 的两张查表变成可执行的诊断。

用法:
    python analyze_waveform.py <csv文件> [--target-col 目标列] [--actual-col 实际列]
                              [--integral-col 积分列] [--fs 控制频率Hz]

CSV 需带表头（VOFA+ 的 DataLoader 导出即可），至少包含"目标"和"实际/当前"两列。
只有一列（误差）也可以，会自动当作误差序列分析。

输出:
    - 自动判定的主导现象（高频颤动 / 低频漂移 / 超调振荡 / 响应迟缓 / 静差）
    - 对应的调参建议（对应 SKILL.md 的表 1、表 2）
    - 定量指标:超调量、振荡频率、上升时间、稳态误差、积分项是否发散
"""

import argparse
import csv
import math
import re
import sys

# ---- 现象判定阈值 ----------------------------------------------------------
# KI 过大的典型表现是"来回以 1-2 秒周期来回晃"，即 0.5~1 Hz。
# 所以低频漂移的上界必须覆盖到 2 Hz，否则真实的 KI 过大波形会被误判成中频。
JITTER_HZ = 8.0        # 高于此值: 高频颤动 → KD 过大
DRIFT_HZ = 2.0         # 低于此值: 低频漂移 → KI 过大（覆盖 1~2 秒周期）
SLOW_RISE_S = 0.6      # 上升时间超过 0.6 秒视为响应迟缓 → KP 太小
OVERSHOOT_RATIO = 0.15 # 超调量超过 15% 视为超调振荡
DEADBAND_RATIO = 0.08  # 稳态误差大于跳变幅度 8% 视为有静差


def read_csv(path):
    """读取 CSV，返回 (表头列表, 列数据字典)。自动识别分隔符。"""
    try:
        f = open(path, "r", encoding="utf-8-sig", newline="")
    except FileNotFoundError:
        raise SystemExit(f"错误: 找不到文件 {path}")
    except UnicodeDecodeError:
        f = open(path, "r", encoding="gbk", newline="")
    with f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
            rows = list(csv.reader(f, dialect))
        except csv.Error:
            rows = list(csv.reader(f, delimiter=","))

    if not rows:
        raise SystemExit("错误: CSV 为空")

    header = [h.strip() for h in rows[0]]
    # 无表头的情况（整行都是数字）→ 命名为 col0/col1/...
    if not any(re.search(r"[^\d.eE+-]", h or "") for h in header):
        header = [f"col{i}" for i in range(len(header))]
        data_rows = rows
    else:
        data_rows = rows[1:]

    cols = {h: [] for h in header}
    for r in data_rows:
        if not r:
            continue
        for i, h in enumerate(header):
            if i >= len(r):
                continue
            v = r[i].strip()
            if v == "":
                continue
            try:
                cols[h].append(float(v))
            except ValueError:
                pass
    return header, cols


def guess_cols(header, cols):
    """
    无表头时按「变化形态」猜列。
    判据不能只看跳变剧烈程度 —— 积分列(误差累积)的跳变同样剧烈，会被误判成实际值列。
    实际值列的真正特征是：与目标列的阶跃时刻对齐、并朝目标值收敛。

    返回 (目标列名, 实际列名) 或 None。
    """
    cand = [h for h in header if len(cols[h]) > 20]
    if len(cand) < 2:
        return None

    # 1) 目标列 = 含明确阶跃的列。
    #    注意不能用"前后两半均值差"来打分：阶跃通常发生在前 1/3 处
    #    （前面是静止基线，后面是长段收敛），两半均值差会被压缩得很小
    #    （实测真实目标列只得 0.33 分，低于阈值而漏判）。
    #    改用「首个显著跳变前后的均值差 / 全距」。
    def step_score(name):
        v = cols[name]
        idx = _first_step_index(v)
        if idx is None:
            return 0.0
        rng = max(v) - min(v)
        if rng < 1e-9:
            return 0.0
        before = v[:idx]
        after = v[idx:]
        if not before or not after:
            return 0.0
        m1 = sum(before) / len(before)
        m2 = sum(after) / len(after)
        return abs(m2 - m1) / rng

    target_col = max(cand, key=step_score)
    if step_score(target_col) < 0.5:
        return None    # 没有明显的阶跃特征，放弃猜测

    t_step = _first_step_index(cols[target_col])

    # 2) 实际值列 = 在目标阶跃点之后有明显响应、且朝目标终值收敛的列。
    #    不能用"平滑度"当主判据：积分列是误差的累积，天然比实际值更平滑，
    #    靠平滑度会选反（实测把积分列认成了实际值列）。
    #    可靠判据是方向一致性：实际值列在阶跃后的变化方向必须与目标一致。
    t_step = _first_step_index(cols[target_col])
    goal = cols[target_col][-1]
    pre_goal = cols[target_col][t_step - 1] if t_step else goal

    best, best_score = None, -1e18
    for c in cand:
        if c == target_col:
            continue
        v = cols[c]
        pre = v[t_step - 1] if t_step else v[0]
        post = v[min(t_step + int(0.05 * len(v)), len(v) - 1)] if t_step else v[-1]

        # 方向一致性：实际值必须朝目标方向移动（分数 0~1）
        want = 1.0 if goal > pre_goal else (-1.0 if goal < pre_goal else 0.0)
        moved = post - pre
        if want == 0 or want * moved <= 0:
            continue                       # 方向不对，不是实际值列
        direction = min(1.0, abs(moved) / (abs(goal - pre_goal) + 1e-9))

        # 收敛度：末值离目标越近越好
        gap = abs(v[-1] - goal) / (abs(goal - pre_goal) + 1e-9)

        score = direction * 2.0 - gap
        if score > best_score:
            best_score, best = score, c
    return target_col, best


def _first_step_index(v):
    """找出序列中第一个显著跳变的位置（用于判断其他列是否有响应）。"""
    n = len(v)
    if n < 10:
        return None
    diffs = [abs(v[i] - v[i - 1]) for i in range(1, n)]
    mad = sorted(diffs)[len(diffs) // 2] or 1e-9
    rng = max(v) - min(v)
    thr = max(6 * mad, 0.15 * rng)
    for i in range(1, n):
        if abs(v[i] - v[i - 1]) > thr:
            return i
    return None


def pick_col(header, cols, *keywords):
    """按关键词模糊匹配列名。"""
    for kw in keywords:
        for h in header:
            if kw.lower() in h.lower() and len(cols[h]) > 2:
                return h
    return None


def detrend(x):
    """去均值 + 去线性趋势，得到振荡分量。"""
    n = len(x)
    if n < 3:
        return [v - (sum(x) / n) for v in x]
    mx = sum(range(n)) / n
    my = sum(x) / n
    num = sum((i - mx) * (x[i] - my) for i in range(n))
    den = sum((i - mx) ** 2 for i in range(n))
    slope = num / den if den else 0.0
    intercept = my - slope * mx
    return [x[i] - (intercept + slope * i) for i in range(n)]


def dominant_freq(x, fs, noise_gate=0.02):
    """
    用 DFT 全频扫描估计主频与幅度（纯 Python）。
    相比零穿越法可靠得多：零穿越法在信号叠加趋势、或接近奈奎斯特频率时会严重
    低估主频（实测 14 Hz 信号被低估成 6.2 Hz）。

    noise_gate: 相对噪声门槛（相对于该段信号全距）。幅度低于此值视为"无明显振荡"，
                返回 0 Hz。这一步很关键 —— 否则单调收敛或已稳定的波形也会被
                报出一个虚假的低频主频（实测无振荡信号被误报为 0.55 Hz 低频漂移）。

    返回 (主频 Hz, 幅度)
    """
    y = detrend(x)
    n = len(y)
    if n < 16:
        return 0.0, 0.0

    span = max(y) - min(y)
    # 幅度门槛：小于全距的 noise_gate 视为噪声，不构成振荡
    if span < 1e-12 or span < noise_gate * abs(max(x) - min(x)):
        return 0.0, 0.0

    # 扫描范围：低频漂移最快约 1 秒周期(1 Hz)，留到 0.3 Hz；
    # 高频颤动取到 0.4*fs（超过奈奎斯特一半意义不大）
    f_lo, f_hi = 0.3, 0.4 * fs
    steps = 1200
    df = (f_hi - f_lo) / steps

    mags = []
    for k in range(steps + 1):
        f = f_lo + k * df
        omega = 2.0 * math.pi * f / fs
        c, s = math.cos(omega), math.sin(omega)
        c_step, s_step = math.cos(omega * n), math.sin(omega * n)
        re, im = 0.0, 0.0
        cr, ci, sr, si = 1.0, 0.0, 1.0, 0.0
        for v in y:
            re += v * cr
            im -= v * si
            cr, ci = cr * c - ci * s, cr * s + ci * c
            sr, si = sr * c_step - si * s_step, sr * s_step + si * c_step
        mags.append(math.hypot(re, im) / n * 2.0)

    best_k = max(range(len(mags)), key=lambda i: mags[i])

    # 抛物线插值细化峰值位置，消除扫描栅格带来的频率偏差
    best_f = f_lo + best_k * df
    if 0 < best_k < len(mags) - 1:
        y0, y1, y2 = mags[best_k - 1], mags[best_k], mags[best_k + 1]
        denom = (y0 - 2 * y1 + y2)
        if denom != 0:
            delta = 0.5 * (y0 - y2) / denom
            if -1 <= delta <= 1:
                best_f += delta * df

    return best_f, mags[best_k]


def find_steps(actual):
    """
    找出目标跳变（阶跃）区段，返回跳变后的索引列表。

    必须在「目标序列」上调用，不能在误差序列上调用：
    误差里含大幅振荡时，振荡造成的每半周跳变都会被误判为阶跃，
    导致最后一个"阶跃"落在数据末尾，稳态分析段被压缩成噪声尾部。
    """
    n = len(actual)
    if n < 10:
        return []
    diffs = [abs(actual[i] - actual[i - 1]) for i in range(1, n)]
    if not diffs:
        return []
    mad = sorted(diffs)[len(diffs) // 2] or 1e-9
    rng = max(actual) - min(actual)
    thr = max(6 * mad, 0.15 * rng)
    steps = []
    for i in range(1, n):
        if abs(actual[i] - actual[i - 1]) > thr:
            # 合并 5 个采样点内的连续跳变，只保留跳变后的第一个点
            if not steps or i - steps[-1] > 5:
                steps.append(i)
    return steps


def analyze_step(actual, start, end, target, fs):
    """
    分析一个阶跃响应区段:
      - 超调量 = (峰值 - 目标) / 目标
      - 上升时间 = 首次到达目标 90% 的时间
    """
    seg = actual[start:end]
    if len(seg) < 5:
        return None
    goal = target[start:end]
    step_amp = goal[-1] - actual[start]
    if abs(step_amp) < 1e-9:
        return None

    # 峰值（在目标方向上超出最多的点）
    prog = [a - actual[start] for a in seg]
    peak_idx = max(range(len(prog)), key=lambda i: prog[i] * (1 if step_amp > 0 else -1))
    overshoot = (prog[peak_idx] - step_amp) / abs(step_amp)

    # 上升时间（到 90%），换算成秒：rise_idx 是段内下标，采样周期为 1/fs 秒
    rise_idx = None
    for i, p in enumerate(prog):
        if abs(p) >= 0.9 * abs(step_amp):
            rise_idx = i
            break
    rise_sec = (rise_idx / fs) if rise_idx is not None else None
    # 整段结束时仍未到达 90% —— 响应确实很慢（或根本没到位）
    reached_90 = rise_idx is not None
    # 最终收敛到的位置与目标的差距（用于区分「慢」和「有静差」）
    final_prog = prog[-1]
    return {
        "step_amp": step_amp,
        "overshoot": overshoot,
        "rise_sec": rise_sec,
        "reached_90": reached_90,
        "final_prog_ratio": final_prog / step_amp if abs(step_amp) > 1e-12 else None,
        "seg_len_sec": len(seg) / fs,
    }


def verdict(freq, amp, overshoot, rise_sec, steady_err, step_amp,
            integral, integral_growing, reached_90, final_ratio):
    """
    根据 SKILL.md 表1/表2 给出判定与建议。
    判定顺序 = 「现象」的因果优先级：振荡类现象比静态偏差更显眼，先判振荡。

    返回 (现象描述, 建议列表)
    """
    advice = []

    # 1) 高频颤动 → KD 相对 KP 过大（最明确的故障现象，优先判）
    if freq > JITTER_HZ:
        desc = f"高频颤动（主频 {freq:.1f} Hz）→ KD 相对 KP 过大"
        advice.append("表1 第2行：减小 KD，或同时减小 KP")
        advice.append("若减小 KD 后仍抖，先在视觉侧加中值/卡尔曼滤波，不要继续压 KD")
        return desc, advice

    # 2) 低频漂移/来回晃 → KI 过大
    #    优先于超调判定：KI 造成的往复摆动外观上很像"超调"，但病因完全不同，
    #    按低频处理才对（表1 第3行说"增大 KD"是针对 KP 相对 KD 过大的中频摆动）。
    if freq > 0 and freq <= DRIFT_HZ:
        desc = f"低频漂移/来回晃（主频 {freq:.2f} Hz，周期约 {1/freq:.1f} s）→ KI 过大"
        advice.append("表2 第2行：减小 KI")
        advice.append("低频抖动优先减 KI，不要先去动 KP/KD")
        if integral_growing:
            advice.append("积分项持续增大 → 若减小 KI 后误差消不掉，确认 KI 极性是否反了")
        return desc, advice

    # 3) 超调振荡 → KP 过大或 KD 不足（无明显振荡成分时，纯超调属于此类）
    if overshoot is not None and overshoot > OVERSHOOT_RATIO:
        desc = f"超调振荡（超调量 {overshoot*100:+.0f}%）→ KP 过大或 KD 不足"
        advice.append("表2 第1行：减小 KP，或增大 KD")
        advice.append("若已减小 KI 且系统反应变迟钝，也需回头微调 KP/KD（表1 第4行：增加 KD）")
        return desc, advice

    # 4) 静差：能稳定追踪（有明确阶跃响应），但收敛位置偏离目标
    #    用 final_ratio（实际收敛/跳变幅度）判定，比稳态误差更可靠。
    if step_amp and final_ratio is not None:
        deficit = 1.0 - abs(final_ratio)
        if deficit > DEADBAND_RATIO:
            desc = f"存在静差（最终只走到目标的 {final_ratio*100:.0f}%）"
            advice.append("静差 → 加 KI（先给 0.001 或 0.1×KP 起步）")
            if integral_growing:
                advice.append("积分项持续增大但误差没消 → 确认 KI 极性是否反了")
            return desc, advice

    # 5) 响应迟缓 → KP 太小。用绝对时间判断，与数据长度无关。
    if reached_90 is False:
        desc = "响应迟缓：记录结束前仍未到达目标的 90% → KP 太小"
        advice.append("表1 第1行：增大 KP，同时可能需要稍增 KD")
        return desc, advice
    if rise_sec is not None and rise_sec > SLOW_RISE_S:
        desc = f"响应迟缓（上升时间 {rise_sec:.2f} s）→ KP 太小"
        advice.append("表1 第1行：增大 KP，同时可能需要稍增 KD")
        return desc, advice

    # 6) 中频摆动（2~8 Hz）
    if freq and DRIFT_HZ < freq <= JITTER_HZ:
        desc = f"中频摆动（主频 {freq:.1f} Hz）"
        advice.append("表1 第3行：增大 KD，或适当减小 KP")
        return desc, advice

    desc = "未匹配到明显异常（波形接近单调收敛）"
    advice.append("若仍有轻微残差，先确认死区与输出限幅是否生效，再考虑加 KI")
    return desc, advice


def main():
    ap = argparse.ArgumentParser(
        description="VOFA+ 波形判读（对应 pid-gimbal-tuning skill 的表1/表2）")
    ap.add_argument("csv", help="VOFA+ DataLoader 导出的 CSV")
    ap.add_argument("--target-col", help="目标坐标列名")
    ap.add_argument("--actual-col", help="实际/当前坐标列名")
    ap.add_argument("--integral-col", help="积分项列名（强烈建议提供）")
    ap.add_argument("--fs", type=float, default=30.0, help="控制频率 Hz，默认 30")
    ap.add_argument("--window", type=float, default=2.0, help="阶跃区段长度（秒），默认 2")
    args = ap.parse_args()

    header, cols = read_csv(args.csv)

    tcol = args.target_col or pick_col(header, cols, "target", "目标", "goal", "setpoint", "ref")
    acol = args.actual_col or pick_col(header, cols, "opencv", "actual", "current",
                                       "实际", "当前", "real", "pos")
    icol = args.integral_col or pick_col(header, cols, "integral", "积分")

    # 都认不出来时，尝试按形态猜（主要服务于无表头的导出文件）
    if (not tcol or not acol) and not (args.target_col and args.actual_col):
        guessed = guess_cols(header, cols)
        if guessed:
            tcol = tcol or guessed[0]
            acol = acol or guessed[1]
            print(f"[提示] 按数据形态推断列: 目标={tcol} 实际={acol}（如需指定请用 --target-col/--actual-col）")

    if not tcol or not acol:
        print("错误: 找不到目标列或实际列。")
        print("  读到的列:", header)
        print("  用 --target-col / --actual-col 显式指定。")
        sys.exit(1)

    target = cols[tcol]
    actual = cols[acol]
    n = min(len(target), len(actual))
    target, actual = target[:n], actual[:n]
    err = [target[i] - actual[i] for i in range(n)]

    integral_growing = False
    if icol:
        integral = cols[icol]
        # 「持续增大」的判据：末值绝对值显著大于首值，且样本量够
        if len(integral) > max(4, n // 2) and abs(integral[-1]) > abs(integral[0]):
            integral_growing = abs(integral[-1]) - abs(integral[0]) > 1e-9
        integral_note = (f"积分项: 首={integral[0]:.3g} 末={integral[-1]:.3g} "
                         f"→ {'持续增大，需警惕积分饱和' if integral_growing else '未持续增大'}")
    else:
        integral_note = ("未提供积分列（--integral-col）。"
                         "积分项是否持续增大是判断该加/减 KI 最直接的依据。")

    # 阶跃检测必须在目标序列上做（原因见 find_steps 的注释）
    steps = find_steps(target)
    seg_len = max(10, int(args.window * args.fs))
    quiet_start = 0
    if steps:
        last_step = max(steps)
        # 从最后一个阶跃之后留 1.5 个区段长度作为稳定段，跳过上升沿暂态
        quiet_start = last_step + int(seg_len * 1.5)
    if quiet_start + seg_len > n:
        quiet_start = max(0, n - seg_len)
    if quiet_start <= 0:
        # 数据里没有阶跃（全程稳态或单调），直接用末尾段
        quiet_start = max(0, n - seg_len)

    quiet = err[quiet_start:quiet_start + seg_len]
    # 噪声门槛以「目标跳变幅度」为基准：小于 2% 跳变幅度的波动不算振荡
    step_amp_ref = max(abs(target[i] - target[i - 1]) for i in range(1, n)) if n > 1 else 1.0
    freq, amp = dominant_freq(quiet, args.fs, noise_gate=0.02)
    if step_amp_ref > 0 and amp < 0.02 * step_amp_ref:
        freq, amp = 0.0, 0.0

    # 阶跃响应分析：基准点取阶跃前最后一个采样，分析区间延伸到数据末尾。
    # 不能只用固定长度的短窗口 —— 超调峰值和慢上升往往发生在几秒之后，
    # 窗口太短会漏掉峰值（实测超调波形被测成 -2.7%，实际 +18%）。
    overshoot, rise_sec, step_amp = None, None, 0.0
    reached_90, final_ratio = None, None
    for s in steps:
        if s > 0 and s < n - 20:
            r = analyze_step(actual, s - 1, n, target, args.fs)
            if r and abs(r["step_amp"]) > 1e-6:
                overshoot = r["overshoot"]
                rise_sec = r["rise_sec"]
                step_amp = r["step_amp"]
                reached_90 = r["reached_90"]
                final_ratio = r["final_prog_ratio"]
                break

    # 稳态误差（末尾 10% 区间）
    tail = err[int(n * 0.9):]
    steady_err = sum(tail) / len(tail) if tail else 0.0

    desc, advice = verdict(freq, amp, overshoot, rise_sec, steady_err,
                          step_amp, cols.get(icol), integral_growing,
                          reached_90, final_ratio)

    print("=" * 60)
    print("VOFA+ 波形判读结果")
    print("=" * 60)
    print(f"数据点数     : {n}   控制频率: {args.fs} Hz")
    print(f"目标列 / 实际列: {tcol} / {acol}")
    print()
    print("【定量指标】")
    print(f"  主导现象     : {desc}")
    print(f"  振荡主频: {freq:.2f} Hz" + (f"  (周期 {1/freq:.2f} s)" if freq else "  (无明显周期)"))
    print(f"  振荡幅度     : ±{amp:.4g}")
    if overshoot is not None:
        print(f"  超调量       : {overshoot*100:+.1f}%")
    if rise_sec is not None:
        print(f"  上升时间     : {rise_sec:.2f} s")
    elif reached_90 is False:
        print(f"  上升时间     : 记录结束前未达到目标的 90%（{(n - steps[0])/args.fs:.1f} s 仍未到位）")
    print(f"  稳态误差     : {steady_err:+.4g}")
    print(f"  {integral_note}")
    print()
    print("【调参建议】（对应 SKILL.md 表1 / 表2）")
    for a in advice:
        print(f"  · {a}")
    print()
    print("提醒: 每次只改一个参数，改完记录现象再改下一个。")
    print("=" * 60)


if __name__ == "__main__":
    main()