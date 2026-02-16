# spectro_pro.py
# ------------------------------------------------------------
# Robust DIY Spectrophotometer Program (Serial -> Absorbance -> Concentration)
#
# Key features:
# - Auto port + auto baud detection (optional)
# - Extremely tolerant serial parsing (plain, key=value, CSV, JSON-like)
# - Outlier rejection (Hampel filter) + smoothing (moving average)
# - Dark + Blank capture with persistent config
# - Calibration from JSON (linear or polynomial A vs C)
# - Quality checks + stable logging
#
# Requirements:
#   pip install pyserial numpy
#
# Examples:
#   python spectro_pro.py capture --what dark  --port auto --baud auto
#   python spectro_pro.py capture --what blank --port auto --baud auto
#   python spectro_pro.py measure --port auto --baud auto
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("ERROR: Missing pyserial. Install with: pip install pyserial", file=sys.stderr)
    raise


# -------------------------
# Parsing: tolerate anything
# -------------------------

FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
KEYVAL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_\-]*)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

# Priority keys if device sends labeled values
DEFAULT_KEY_PRIORITY = [
    "adc", "raw", "counts", "count", "intensity", "i", "signal", "sig", "value", "v"
]


def parse_line(line: str) -> Tuple[List[float], Dict[str, float]]:
    """
    Returns:
      numbers: all floats found in the line
      kv: dictionary of key->float from patterns like key=123 or key:123
    Supports JSON-ish content too (we try json.loads if it looks like JSON).
    """
    line = line.strip()
    kv: Dict[str, float] = {}

    # Try JSON first (fast check)
    if line.startswith("{") and line.endswith("}"):
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (int, float)) and isinstance(k, str):
                        kv[k.lower()] = float(v)
                # still also collect numbers
        except Exception:
            pass

    # key=value / key:value scanning
    for m in KEYVAL_RE.finditer(line):
        kv[m.group(1).lower()] = float(m.group(2))

    # all numbers
    nums = [float(x) for x in FLOAT_RE.findall(line)]
    return nums, kv


def choose_value(
    nums: List[float],
    kv: Dict[str, float],
    key: Optional[str],
    field: Optional[int],
    key_priority: List[str]
) -> Optional[float]:
    """
    Select one number to represent intensity.
    - If --key is given: use that key from kv
    - Else if kv has priority keys: choose first present
    - Else if --field is given: choose that index from nums
    - Else: choose last number in the line
    """
    if key:
        return kv.get(key.lower(), None)

    for k in key_priority:
        if k in kv:
            return kv[k]

    if field is not None:
        if 0 <= field < len(nums):
            return nums[field]
        return None

    if nums:
        return nums[-1]
    return None


# -------------------------
# Config & Calibration
# -------------------------

@dataclass
class DeviceConfig:
    dark: float = 0.0       # mean dark counts
    blank: float = 0.0      # mean blank counts (I0)
    eps: float = 1e-12
    adc_max: Optional[float] = None  # for saturation warnings (e.g., 1023, 4095, 65535)

    @staticmethod
    def load(path: Path) -> "DeviceConfig":
        if not path.exists():
            return DeviceConfig()
        d = json.loads(path.read_text(encoding="utf-8"))
        return DeviceConfig(
            dark=float(d.get("dark", 0.0)),
            blank=float(d.get("blank", 0.0)),
            eps=float(d.get("eps", 1e-12)),
            adc_max=(float(d["adc_max"]) if d.get("adc_max", None) is not None else None),
        )

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {"dark": self.dark, "blank": self.blank, "eps": self.eps, "adc_max": self.adc_max},
                indent=2
            ),
            encoding="utf-8"
        )


@dataclass
class Calibration:
    # Default linear: A = slope*C + intercept
    model: str = "linear_A_vs_C"
    slope: float = 1.0
    intercept: float = 0.0
    units: str = "mg/L"
    # Poly: A = a0 + a1*C + a2*C^2 + ...
    poly_coeffs_A_vs_C: Optional[List[float]] = None
    valid_C_range: Optional[Tuple[float, float]] = (0.0, 100.0)

    @staticmethod
    def load(path: Path) -> "Calibration":
        if not path.exists():
            template = {
                "model": "linear_A_vs_C",
                "slope": 1.0,
                "intercept": 0.0,
                "units": "mg/L",
                "poly_coeffs_A_vs_C": None,
                "valid_C_range": [0.0, 100.0]
            }
            path.write_text(json.dumps(template, indent=2), encoding="utf-8")
            return Calibration()

        d = json.loads(path.read_text(encoding="utf-8"))
        rng = d.get("valid_C_range", None)
        vr = None
        if isinstance(rng, list) and len(rng) == 2:
            vr = (float(rng[0]), float(rng[1]))

        return Calibration(
            model=str(d.get("model", "linear_A_vs_C")),
            slope=float(d.get("slope", 1.0)),
            intercept=float(d.get("intercept", 0.0)),
            units=str(d.get("units", "mg/L")),
            poly_coeffs_A_vs_C=d.get("poly_coeffs_A_vs_C", None),
            valid_C_range=vr if vr else None
        )

    def concentration_from_absorbance(self, A: float) -> float:
        if self.model == "linear_A_vs_C":
            if abs(self.slope) < 1e-15:
                raise ValueError("Calibration slope is ~0. Fix calibration.json slope.")
            return (A - self.intercept) / self.slope

        if self.model == "poly_A_vs_C":
            if not self.poly_coeffs_A_vs_C:
                raise ValueError("poly_coeffs_A_vs_C missing for poly_A_vs_C.")
            # Solve polynomial: a0 + a1*C + a2*C^2 + ... - A = 0
            coeffs = list(self.poly_coeffs_A_vs_C)
            coeffs[0] -= A
            roots = np.roots(list(reversed(coeffs)))
            real_roots = [float(r.real) for r in roots if abs(r.imag) < 1e-8]

            if self.valid_C_range:
                lo, hi = self.valid_C_range
                real_roots = [r for r in real_roots if lo - 1e-9 <= r <= hi + 1e-9]

            if not real_roots:
                raise ValueError("No valid real root found. Check calibration range/model.")
            # Choose root closest to mid-range (more stable)
            if self.valid_C_range:
                mid = 0.5 * (self.valid_C_range[0] + self.valid_C_range[1])
                real_roots.sort(key=lambda r: abs(r - mid))
            return real_roots[0]

        raise ValueError(f"Unknown model: {self.model}")


# -------------------------
# Signal processing (robust)
# -------------------------

def hampel_filter(values: np.ndarray, k: int = 7, t0: float = 3.0) -> np.ndarray:
    """
    Hampel filter for outlier removal.
    values: 1D array
    k: half-window size
    t0: threshold in MAD units
    """
    n = len(values)
    if n < 2 * k + 1:
        return values

    x = values.copy()
    for i in range(k, n - k):
        window = x[i - k:i + k + 1]
        med = np.median(window)
        mad = np.median(np.abs(window - med)) + 1e-12
        if abs(x[i] - med) > t0 * 1.4826 * mad:
            x[i] = med
    return x


def compute_absorbance(I_corr: float, I0_corr: float, eps: float) -> float:
    # Transmittance T = I/I0
    T = max(I_corr / max(I0_corr, eps), eps)
    return -math.log10(T)


# -------------------------
# Serial helpers
# -------------------------

def list_serial_ports() -> List[str]:
    ports = []
    for p in list_ports.comports():
        ports.append(p.device)
    return ports


def auto_choose_port() -> str:
    ports = list_serial_ports()
    if not ports:
        raise RuntimeError("No serial ports found. Check USB connection / drivers.")
    if len(ports) == 1:
        return ports[0]
    # If multiple, choose the one that looks like Arduino/USB-serial (heuristic)
    preferred = []
    for p in list_ports.comports():
        desc = (p.description or "").lower()
        hwid = (p.hwid or "").lower()
        if ("arduino" in desc) or ("usb" in desc) or ("acm" in p.device.lower()) or ("ch340" in desc) or ("cp210" in desc) or ("ftdi" in desc) or ("usb" in hwid):
            preferred.append(p.device)
    return preferred[0] if preferred else ports[0]


def open_serial(port: str, baud: int, timeout: float = 0.2) -> serial.Serial:
    ser = serial.Serial(port=port, baudrate=baud, timeout=timeout)
    # Many boards reset when serial opens; give time
    time.sleep(1.8)
    ser.reset_input_buffer()
    return ser


def detect_baud(port: str, baud_candidates: List[int], seconds_each: float,
                key: Optional[str], field: Optional[int], key_priority: List[str]) -> int:
    """
    Try opening with each baud and see if we can parse numbers reliably.
    """
    best_baud = None
    best_score = -1

    for baud in baud_candidates:
        try:
            ser = open_serial(port, baud, timeout=0.2)
        except Exception:
            continue

        parsed = 0
        total = 0
        t_end = time.time() + seconds_each

        try:
            while time.time() < t_end:
                line = ser.readline().decode(errors="ignore")
                total += 1
                nums, kv = parse_line(line)
                v = choose_value(nums, kv, key, field, key_priority)
                if v is not None and np.isfinite(v):
                    parsed += 1
            score = parsed
            if score > best_score:
                best_score = score
                best_baud = baud
        finally:
            ser.close()

    if best_baud is None:
        raise RuntimeError("Could not detect baud rate. Try specifying --baud manually.")
    return best_baud


# -------------------------
# Modes
# -------------------------

def capture_level(ser: serial.Serial, samples: int,
                  key: Optional[str], field: Optional[int], key_priority: List[str],
                  scale: float, offset: float) -> Tuple[float, float]:
    vals: List[float] = []
    t0 = time.time()
    while len(vals) < samples:
        line = ser.readline().decode(errors="ignore")
        nums, kv = parse_line(line)
        v = choose_value(nums, kv, key, field, key_priority)
        if v is None:
            continue
        v = v * scale + offset
        vals.append(float(v))

    arr = np.array(vals, dtype=float)
    arr = hampel_filter(arr, k=max(3, min(7, samples // 20)), t0=3.0)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    dt = time.time() - t0
    print(f"[OK] captured mean={mean:.6f}, sd={sd:.6f}, n={samples}, time={dt:.2f}s")
    return mean, sd


def cmd_capture(args: argparse.Namespace) -> None:
    cfg_path = Path(args.config)
    cfg = DeviceConfig.load(cfg_path)
    if args.adc_max is not None:
        cfg.adc_max = args.adc_max

    port = auto_choose_port() if args.port == "auto" else args.port
    baud = args.baud
    if baud == 0:
        baud = detect_baud(
            port,
            baud_candidates=args.baud_candidates,
            seconds_each=args.baud_probe_seconds,
            key=args.key,
            field=args.field,
            key_priority=args.key_priority,
        )
        print(f"[INFO] Auto-detected baud: {baud}")

    ser = open_serial(port, baud, timeout=0.2)
    try:
        print(f"[INFO] CAPTURE {args.what.upper()} | port={port} baud={baud}")
        mean, sd = capture_level(
            ser, args.samples,
            key=args.key, field=args.field, key_priority=args.key_priority,
            scale=args.scale, offset=args.offset
        )
        if args.what == "dark":
            cfg.dark = mean
        else:
            cfg.blank = mean
        cfg.save(cfg_path)
        print(f"[OK] saved: {cfg_path}")
        print(f"     dark={cfg.dark:.6f}, blank={cfg.blank:.6f}, adc_max={cfg.adc_max}")
    finally:
        ser.close()


def cmd_measure(args: argparse.Namespace) -> None:
    cfg_path = Path(args.config)
    cal_path = Path(args.calibration)

    cfg = DeviceConfig.load(cfg_path)
    cal = Calibration.load(cal_path)

    port = auto_choose_port() if args.port == "auto" else args.port
    baud = args.baud
    if baud == 0:
        baud = detect_baud(
            port,
            baud_candidates=args.baud_candidates,
            seconds_each=args.baud_probe_seconds,
            key=args.key,
            field=args.field,
            key_priority=args.key_priority,
        )
        print(f"[INFO] Auto-detected baud: {baud}")

    # safety checks
    if cfg.blank == 0.0 and not args.allow_unset:
        print("[ERROR] blank not set. Run: capture --what blank", file=sys.stderr)
        sys.exit(2)

    out_csv = Path(args.csv) if args.csv else Path(f"measure_{time.strftime('%Y%m%d_%H%M%S')}.csv")

    ser = open_serial(port, baud, timeout=0.2)

    # rolling window for smoothing
    win = deque(maxlen=args.window)

    # for stats / quality
    last_print = 0.0
    start_time = time.time()
    rows = 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    f = out_csv.open("w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow([
        "timestamp", "raw_selected", "raw_filtered",
        "I_corr", "I0_corr", "absorbance", f"concentration({cal.units})",
        "flags"
    ])

    print("[INFO] MEASURE (live)")
    print(f"       port={port} baud={baud}")
    print(f"       dark={cfg.dark:.6f}, blank={cfg.blank:.6f}")
    print(f"       calibration: {cal.model} | units={cal.units}")
    print(f"       logging: {out_csv}")
    print("       Ctrl+C to stop.\n")

    try:
        while True:
            line = ser.readline().decode(errors="ignore")
            nums, kv = parse_line(line)
            v = choose_value(nums, kv, args.key, args.field, args.key_priority)
            if v is None or not np.isfinite(v):
                continue

            v = v * args.scale + args.offset
            win.append(float(v))

            # wait until window fills enough
            if len(win) < max(5, min(args.window, 10)):
                continue

            # robust clean
            arr = np.array(win, dtype=float)
            arr2 = hampel_filter(arr, k=max(3, min(7, len(arr)//3)), t0=3.0)
            raw_filtered = float(np.mean(arr2))

            # quality metrics
            sd = float(np.std(arr2, ddof=1)) if len(arr2) > 1 else 0.0
            sem = sd / math.sqrt(len(arr2)) if len(arr2) > 1 else 0.0

            # corrections
            I_corr = raw_filtered - cfg.dark
            I0_corr = cfg.blank - cfg.dark

            flags = []
            if cfg.adc_max is not None:
                if raw_filtered > 0.98 * cfg.adc_max:
                    flags.append("SATURATING")
                if raw_filtered < 0.02 * cfg.adc_max:
                    flags.append("VERY_LOW")

            if I0_corr <= cfg.eps:
                if args.allow_unset:
                    A = float("nan")
                    C = float("nan")
                    flags.append("BLANK_UNSET_OR_BAD")
                else:
                    raise RuntimeError("Invalid blank after dark correction. Re-capture dark/blank.")
            else:
                # clamp corrected intensity
                if I_corr <= cfg.eps:
                    flags.append("I_CORR<=0")
                    I_corr_clamped = cfg.eps
                else:
                    I_corr_clamped = I_corr

                A = compute_absorbance(I_corr_clamped, I0_corr, cfg.eps)

                try:
                    C = cal.concentration_from_absorbance(A)
                except Exception as e:
                    C = float("nan")
                    flags.append(f"CALC_FAIL:{type(e).__name__}")

            # stability flag
            if sem > 0 and abs(raw_filtered) > 0:
                rsem = sem / max(abs(raw_filtered), 1e-12)
                if rsem > args.rsem_warn:
                    flags.append("UNSTABLE_SIGNAL")

            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            w.writerow([
                ts,
                f"{v:.6f}",
                f"{raw_filtered:.6f}",
                f"{I_corr:.6f}",
                f"{I0_corr:.6f}",
                f"{A:.6f}",
                f"{C:.6f}",
                "|".join(flags) if flags else ""
            ])
            f.flush()
            rows += 1

            now = time.time()
            if now - last_print >= args.print_every:
                last_print = now
                hz = rows / max(now - start_time, 1e-9)
                print(
                    f"{ts} | raw={raw_filtered:10.4f} | A={A:8.5f} | "
                    f"C={C:10.5f} {cal.units} | sd={sd:.4f} sem={sem:.4f} | {hz:.1f} Hz "
                    + (f"| {','.join(flags)}" if flags else "")
                )

    except KeyboardInterrupt:
        print("\n[INFO] stopped by user.")
    finally:
        f.close()
        ser.close()
        print(f"[OK] saved: {out_csv}")


# -------------------------
# CLI
# -------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Robust spectrophotometer serial reader + analyzer")

    p.add_argument("--port", default="auto", help="Serial port (COM5, /dev/ttyACM0) or 'auto'")
    p.add_argument("--baud", default="auto",
                   help="Baud rate (e.g., 9600) or 'auto'")
    p.add_argument("--config", default="config.json", help="Config JSON for dark/blank")
    p.add_argument("--calibration", default="calibration.json", help="Calibration JSON")
    p.add_argument("--key", default=None, help="Prefer a named key from serial like adc/raw/intensity")
    p.add_argument("--field", type=int, default=None,
                   help="Pick Nth number from the line (0-based). Example: --field 0")
    p.add_argument("--scale", type=float, default=1.0, help="Scale incoming value (v = v*scale + offset)")
    p.add_argument("--offset", type=float, default=0.0, help="Offset incoming value")
    p.add_argument("--adc-max", type=float, default=None, help="ADC max for saturation warnings (1023,4095,etc)")

    p.add_argument("--baud-probe-seconds", type=float, default=2.0,
                   help="Seconds to test each baud in auto mode")
    p.add_argument("--baud-candidates", type=int, nargs="+",
                   default=[115200, 57600, 38400, 19200, 9600],
                   help="Baud candidates for auto detect")

    # key priority can be customized
    p.add_argument("--key-priority", nargs="+", default=DEFAULT_KEY_PRIORITY,
                   help="Priority keys when line contains key=value pairs or JSON")

    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="Capture dark or blank and save in config")
    c.add_argument("--what", choices=["dark", "blank"], required=True)
    c.add_argument("--samples", type=int, default=300)
    c.set_defaults(func=cmd_capture)

    m = sub.add_parser("measure", help="Live measurement + logging")
    m.add_argument("--window", type=int, default=25, help="Smoothing window length")
    m.add_argument("--print-every", type=float, default=0.5, help="Console update interval seconds")
    m.add_argument("--csv", default=None, help="Output CSV path")
    m.add_argument("--allow-unset", action="store_true", help="Allow measuring even if blank is 0")
    m.add_argument("--rsem-warn", type=float, default=0.01,
                   help="Warn if relative SEM > this (signal instability)")
    m.set_defaults(func=cmd_measure)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Normalize baud
    if isinstance(args.baud, str) and args.baud.lower() == "auto":
        args.baud = 0
    else:
        args.baud = int(args.baud)

    args.func(args)


if __name__ == "__main__":
    main()
