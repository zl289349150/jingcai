# -*- coding: utf-8 -*-
"""predict_all.py — 足球预测系统全面增强版（模块一~五整合 + 部署）
用法:
  python predict_all.py            # 训练全部可得联赛 + 评估 + 生成 predictions.json（含模块演示）
环境变量:
  API_FOOTBALL_KEY  # 可选；未设置时 伤停/裁判 标"未取得"
"""
import io, os, sys, json, time, sqlite3, re, unicodedata
from pathlib import Path
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np
import pandas as pd
from scipy.stats import poisson
from curl_cffi import requests as cr

print("=" * 78)
print("足球预测系统全面增强版  predict_all.py")
print("=" * 78)
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = str(BASE_DIR / "models")
REPORT_DIR = BASE_DIR / "outputs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
OUT_JSON = str(BASE_DIR / "predictions.json")
QUALITY_REPORT = REPORT_DIR / "odds_quality_report.json"
ENV_REPORT = REPORT_DIR / "environment_correlation_report.json"
QUALITY_ROWS = []
ENV_ROWS = []

BJ_TZ = ZoneInfo("Asia/Shanghai")

def current_season_codes(now=None):
    """Return the current season plus two preceding seasons."""
    now = now or datetime.now(BJ_TZ)
    start = now.year if now.month >= 7 else now.year - 1
    return [f"{start % 100:02d}{(start + 1) % 100:02d}",
            f"{(start - 1) % 100:02d}{start % 100:02d}",
            f"{(start - 2) % 100:02d}{(start - 1) % 100:02d}"]

SEASON_CODES = current_season_codes()

# ---------------- 联赛配置（模块二）----------------
LEAGUES = {
    "E0":  {"name": "英超", "fd": "E0",  "seasons": SEASON_CODES, "xg": True},
    "SP1": {"name": "西甲", "fd": "SP1", "seasons": SEASON_CODES, "xg": False},
    "I1":  {"name": "意甲", "fd": "I1",  "seasons": SEASON_CODES, "xg": False},
    "D1":  {"name": "德甲", "fd": "D1",  "seasons": SEASON_CODES, "xg": False},
    "F1":  {"name": "法甲", "fd": "F1",  "seasons": SEASON_CODES, "xg": False},
    "N1":  {"name": "荷甲", "fd": "N1",  "seasons": SEASON_CODES, "xg": False},
    "P1":  {"name": "葡超", "fd": "P1",  "seasons": SEASON_CODES, "xg": False},
}
NO_DATA_LEAGUES = {"日职(J1)": "football-data 无 J1 数据文件", "韩K(K1)": "football-data 无 K1 数据文件", "澳超(AUS)": "football-data 无 AUS 数据文件"}

# EPL 城市坐标（模块五-天气用；其余联赛天气未取得）
CITY_MAP = {
    "Arsenal": ("London", 51.5074, -0.1278), "Chelsea": ("London", 51.5074, -0.1278),
    "Tottenham Hotspur": ("London", 51.5074, -0.1278), "West Ham United": ("London", 51.5074, -0.1278),
    "Manchester City": ("Manchester", 53.4808, -2.2426), "Manchester United": ("Manchester", 53.4808, -2.2426),
    "Liverpool": ("Liverpool", 53.4084, -2.9916), "Everton": ("Liverpool", 53.4084, -2.9916),
    "Newcastle United": ("Newcastle", 54.9783, -1.6178), "Aston Villa": ("Birmingham", 52.4862, -1.8904),
    "Birmingham": ("Birmingham", 52.4862, -1.8904),
    "Leicester City": ("Leicester", 52.6369, -1.1398), "Leeds United": ("Leeds", 53.8008, -1.5491),
    "Brighton & Hove Albion": ("Brighton", 50.8225, -0.1372), "Southampton": ("Southampton", 50.9097, -1.4044),
    "Crystal Palace": ("London", 51.5074, -0.1278), "Bournemouth": ("Bournemouth", 50.7192, -1.8808),
    "Brentford": ("London", 51.5074, -0.1278), "Fulham": ("London", 51.5074, -0.1278),
    "Burnley": ("Burnley", 53.7893, -2.2405), "Nottingham Forest": ("Nottingham", 52.9548, -1.1581),
    "Wolverhampton Wanderers": ("Wolverhampton", 52.5869, -2.1287), "Luton Town": ("Luton", 51.8787, -0.4200),
}
SHORT_CITY = {
    "Man United": ("Manchester", 53.4808, -2.2426), "Man City": ("Manchester", 53.4808, -2.2426),
    "Newcastle": ("Newcastle", 54.9783, -1.6178), "Wolves": ("Wolverhampton", 52.5869, -2.1287),
    "Nott'm Forest": ("Nottingham", 52.9548, -1.1581), "Tottenham": ("London", 51.5074, -0.1278),
    "West Ham": ("London", 51.5074, -0.1278), "Brighton": ("Brighton", 50.8225, -0.1372),
    "Leeds": ("Leeds", 53.8008, -1.5491), "Leicester": ("Leicester", 52.6369, -1.1398),
    "Luton": ("Luton", 51.8787, -0.4200), "Sheffield United": ("Sheffield", 53.3811, -1.4701),
    "Bournemouth": ("Bournemouth", 50.7192, -1.8808), "Aston Villa": ("Birmingham", 52.4862, -1.8904),
    "Crystal Palace": ("London", 51.5074, -0.1278), "Southampton": ("Southampton", 50.9097, -1.4044),
    "Brentford": ("London", 51.5074, -0.1278), "Burnley": ("Burnley", 53.7893, -2.2405),
    "Fulham": ("London", 51.5074, -0.1278), "Everton": ("Liverpool", 53.4084, -2.9916),
    "Arsenal": ("London", 51.5074, -0.1278), "Chelsea": ("London", 51.5074, -0.1278),
    "Liverpool": ("Liverpool", 53.4084, -2.9916),
}
CITY_MAP.update(SHORT_CITY)
DERBY = {("Manchester United", "Manchester City"), ("Arsenal", "Tottenham Hotspur"),
         ("Liverpool", "Everton"), ("Newcastle United", "Sunderland")}
INTL_BREAKS = [(3, 18, 30), (6, 3, 15), (9, 1, 10), (10, 6, 15), (11, 10, 19)]  # 月,起,止（近似FIFA窗口）

# ---------------- 工具 ----------------
def norm(n):
    return str(n).strip().lower().replace("'", "").replace(".", "").replace("&", "and")

def clean_prob(h, d, a):
    """基本比例去水；异常赔率返回 NaN，禁止带病数据进入模型。"""
    odds = np.asarray([h, d, a], dtype=float)
    if (not np.isfinite(odds).all()) or (odds <= 1.01).any() or (odds > 100).any():
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    inv = 1.0 / odds
    return inv / inv.sum()


def odds_quality_report(data, league_code="all"):
    """检查赔率列、异常值、缺失和去水归一性，结果可复核且不参与预测。"""
    cols = [c for c in ("B365H", "B365D", "B365A", "PSH", "PSD", "PSA", "AvgCH", "AvgCD", "AvgCA") if c in data.columns]
    out = {"league": league_code, "rows": int(len(data)), "columns": {}, "selected_source": {"pinnacle": 0, "bet365": 0, "none": 0}}
    for c in cols:
        s = pd.to_numeric(data[c], errors="coerce")
        out["columns"][c] = {
            "missing": int(s.isna().sum()),
            "valid": int(((s > 1.01) & (s <= 100)).sum()),
            "extreme_low": int((s <= 1.01).sum()),
            "extreme_high": int((s > 100).sum()),
            "min": None if s.dropna().empty else float(s.min()),
            "max": None if s.dropna().empty else float(s.max()),
        }
    triples = []
    for _, row in data.iterrows():
        pin = [row.get("PSH"), row.get("PSD"), row.get("PSA")]
        b365 = [row.get("B365H"), row.get("B365D"), row.get("B365A")]
        source = "pinnacle" if all(pd.notna(x) and 1.01 < float(x) <= 100 for x in pin) else ("bet365" if all(pd.notna(x) and 1.01 < float(x) <= 100 for x in b365) else "none")
        out["selected_source"][source] += 1
        if source != "none":
            triples.append(clean_prob(*(pin if source == "pinnacle" else b365)))
    if triples:
        probs = np.asarray(triples)
        sums = probs.sum(axis=1)
        out["devig"] = {
            "valid_samples": int(len(probs)),
            "probability_min": [float(x) for x in probs.min(axis=0)],
            "probability_max": [float(x) for x in probs.max(axis=0)],
            "normalization_max_abs_error": float(np.max(np.abs(sums - 1.0))),
            "normalization_pass": bool(np.allclose(sums, 1.0, atol=1e-9)),
        }
    else:
        out["devig"] = {"valid_samples": 0, "normalization_pass": False}
    QUALITY_ROWS.append(out)
    return out

def download_fd(code, season):
    last_error = None
    for _ in range(3):
        try:
            r = cr.get(f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv", impersonate="chrome", timeout=40, headers=HEADERS)
            if r.status_code == 404:
                print(f"    {season}/{code}.csv 未发布，跳过（不编造当前赛季数据）")
                return None
            r.raise_for_status()
            df = pd.read_csv(io.BytesIO(r.content), encoding="latin-1")
            df["Season"] = season
            return df
        except Exception as e:
            last_error = e
            time.sleep(2)
    print(f"    {season}/{code}.csv 下载失败，跳过：{type(last_error).__name__ if last_error else '未知错误'}")
    return None


# ---------------- 特征工程（模块二核心，泛化版）----------------
def build_features(data, with_xg=False):
    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
    data = data.dropna(subset=["Date", "FTHG", "FTAG"]).sort_values("Date").reset_index(drop=True)
    base_h = ["Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG",
              "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]
    xg_h = ["home_xg", "away_xg"] if with_xg else []
    home = data[base_h + xg_h].copy()
    hcols = ["Date", "Season", "Team", "Opp", "GF", "GA", "HTGF", "HTGA", "S", "SA", "ST", "STA",
             "C", "CA", "F", "FA", "Y", "YA", "R", "RA"] + (["Team_xg", "Opp_xg"] if with_xg else [])
    home.columns = hcols
    home["HomeFlag"] = 1
    base_a = ["Date", "Season", "AwayTeam", "HomeTeam", "FTAG", "FTHG", "HTAG", "HTHG",
              "AS", "HS", "AST", "HST", "AC", "HC", "AF", "HF", "AY", "HY", "AR", "HR"]
    xg_a = ["away_xg", "home_xg"] if with_xg else []
    away = data[base_a + xg_a].copy()
    away.columns = hcols
    away["HomeFlag"] = 0
    log = pd.concat([home, away], ignore_index=True).sort_values(["Team", "Date"]).reset_index(drop=True)
    STAT_COLS = ["GF", "GA", "HTGF", "S", "SA", "ST", "STA", "C", "CA", "F", "FA", "Y", "YA", "R", "RA"]
    def roll(g, cols, n, pfx):
        for c in cols:
            log[pfx + c + f"_{n}"] = g[c].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
    g = log.groupby("Team", sort=False)
    roll(g, STAT_COLS, 5, "all_"); roll(g, STAT_COLS, 10, "all_"); roll(g, STAT_COLS, 15, "all_")
    if with_xg:
        for c, n in [("Team_xg", 5), ("Team_xg", 10), ("Opp_xg", 5), ("Opp_xg", 10)]:
            log[f"all_xg_{c}_{n}"] = g[c].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
    HOME_COLS = ["GF", "GA", "S", "ST", "C", "F", "Y", "R"]
    gh = log[log["HomeFlag"] == 1].groupby("Team", sort=False)
    for c in HOME_COLS:
        log["home_" + c + "_5"] = gh[c].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    ga = log[log["HomeFlag"] == 0].groupby("Team", sort=False)
    for c in HOME_COLS:
        log["away_" + c + "_5"] = ga[c].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    # H2H
    meetings = {}
    out = {k: [] for k in ["H2H_HW", "H2H_D", "H2H_AW", "H2H_goals"]}
    for _, r in data.iterrows():
        key = tuple(sorted([r["HomeTeam"], r["AwayTeam"]]))
        hist = [m for m in meetings.get(key, []) if m[0] < r["Date"]][-5:]
        hw = sum(1 for m in hist if (m[1] == r["HomeTeam"] and m[3] > m[4]) or (m[2] == r["HomeTeam"] and m[4] > m[3]))
        aw = sum(1 for m in hist if (m[1] == r["AwayTeam"] and m[3] > m[4]) or (m[2] == r["AwayTeam"] and m[4] > m[3]))
        out["H2H_HW"].append(hw); out["H2H_D"].append(len(hist) - hw - aw); out["H2H_AW"].append(aw)
        out["H2H_goals"].append(sum(m[3] + m[4] for m in hist))
        meetings.setdefault(key, []).append((r["Date"], r["HomeTeam"], r["AwayTeam"], r["FTHG"], r["FTAG"]))
    for k, v in out.items():
        data[k] = v
    # Elo
    ELO, played = {}, {}
    eh_l, ea_l = [], []
    for _, r in data.iterrows():
        eh, ea = ELO.get(r["HomeTeam"], 1500.0), ELO.get(r["AwayTeam"], 1500.0)
        eh_l.append(eh); ea_l.append(ea)
        exp_h = 1 / (1 + 10 ** ((ea - eh) / 400.0))
        res_h = 1 if r["FTHG"] > r["FTAG"] else (0.5 if r["FTHG"] == r["FTAG"] else 0)
        kh = 40 if played.get(r["HomeTeam"], 0) < 30 else 20
        ka = 40 if played.get(r["AwayTeam"], 0) < 30 else 20
        mov = np.log1p(abs(r["FTHG"] - r["FTAG"]))
        ELO[r["HomeTeam"]] = eh + kh * mov * (res_h - exp_h)
        ELO[r["AwayTeam"]] = ea + ka * mov * ((1 - res_h) - (1 - exp_h))
        played[r["HomeTeam"]] = played.get(r["HomeTeam"], 0) + 1
        played[r["AwayTeam"]] = played.get(r["AwayTeam"], 0) + 1
    data["H_elo"], data["A_elo"] = eh_l, ea_l
    data["elo_diff"] = data["H_elo"] - data["A_elo"]
    # 赛程密度
    td = {}
    dh, da = [], []
    for _, r in data.iterrows():
        dh.append(sum(1 for d in td.get(r["HomeTeam"], []) if r["Date"] - d <= timedelta(days=7)))
        da.append(sum(1 for d in td.get(r["AwayTeam"], []) if r["Date"] - d <= timedelta(days=7)))
        td.setdefault(r["HomeTeam"], []).append(r["Date"]); td.setdefault(r["AwayTeam"], []).append(r["Date"])
    data["H_dens"], data["A_dens"] = dh, da
    # 市场赔率去水（B365 开盘 + Pinnacle）
    b365 = np.array([clean_prob(x, y, z) for x, y, z in zip(data["B365H"], data["B365D"], data["B365A"])])
    pin = np.array([clean_prob(x, y, z) for x, y, z in
                    zip(data["PSH"].fillna(data["B365H"]), data["PSD"].fillna(data["B365D"]), data["PSA"].fillna(data["B365A"]))])
    data["mkt_b365_h"], data["mkt_b365_d"], data["mkt_b365_a"] = b365[:, 0], b365[:, 1], b365[:, 2]
    data["mkt_pin_h"], data["mkt_pin_d"], data["mkt_pin_a"] = pin[:, 0], pin[:, 1], pin[:, 2]
    log = log.set_index(["Date", "Team"])
    xg_feats = ([f"all_xg_Team_xg_{n}" for n in (5, 10)] + [f"all_xg_Opp_xg_{n}" for n in (5, 10)]) if with_xg else []
    feat_cols = ([f"all_{c}_{n}" for c in STAT_COLS for n in (5, 10, 15)] +
                 [f"home_{c}_5" for c in HOME_COLS] + [f"away_{c}_5" for c in HOME_COLS] + xg_feats)
    for c in feat_cols:
        data["H_" + c] = [log.loc[(d, h), c] if (d, h) in log.index else np.nan for d, h in zip(data["Date"], data["HomeTeam"])]
        data["A_" + c] = [log.loc[(d, a), c] if (d, a) in log.index else np.nan for d, a in zip(data["Date"], data["AwayTeam"])]
    data["y"] = np.where(data["FTHG"] > data["FTAG"], 0, np.where(data["FTHG"] == data["FTAG"], 1, 2))
    FEATURES = ([f"H_all_{c}_{n}" for c in STAT_COLS for n in (5, 10, 15)] +
                [f"A_all_{c}_{n}" for c in STAT_COLS for n in (5, 10, 15)] +
                ([f"H_all_xg_Team_xg_{n}" for n in (5, 10)] + [f"A_all_xg_Team_xg_{n}" for n in (5, 10)] +
                 [f"H_all_xg_Opp_xg_{n}" for n in (5, 10)] + [f"A_all_xg_Opp_xg_{n}" for n in (5, 10)]) if with_xg else [] +
                [f"H_home_{c}_5" for c in HOME_COLS] + [f"A_away_{c}_5" for c in HOME_COLS] +
                ["H2H_HW", "H2H_D", "H2H_AW", "H2H_goals", "H_elo", "A_elo", "elo_diff", "H_dens", "A_dens"] +
                ["mkt_b365_h", "mkt_b365_d", "mkt_b365_a", "mkt_pin_h", "mkt_pin_d", "mkt_pin_a"])
    data = data.dropna(subset=FEATURES + ["y"]).reset_index(drop=True)
    return data, FEATURES

# ---------------- 训练与评估（模块二）----------------
def train_league(code, cfg):
    import os as _os, pickle
    print(f"\n--- 联赛 {cfg['name']}（{code}）---")
    data_pkl = f"{MODEL_DIR}/{code}_data.pkl"
    meta_path = f"{MODEL_DIR}/{code}_meta.json"
    cache_valid = False
    if _os.path.exists(data_pkl) and _os.path.exists(meta_path):
        try:
            cache_meta = json.load(open(meta_path, encoding="utf-8"))
            cache_valid = cache_meta.get("seasons") == list(cfg["seasons"])
        except Exception:
            cache_valid = False
    if _os.path.exists(data_pkl) and cache_valid:
        with open(data_pkl, "rb") as f:
            data, FEATURES = pickle.load(f)
        print(f"  从缓存加载数据（{len(data)} 场，特征 {len(FEATURES)}）")
    else:
        frames = [download_fd(cfg["fd"], s) for s in cfg["seasons"]]
        frames = [frame for frame in frames if frame is not None and len(frame)]
        if not frames:
            if _os.path.exists(data_pkl):
                with open(data_pkl, "rb") as f:
                    data, FEATURES = pickle.load(f)
                cache_valid = True
                print(f"  当前赛季文件未取得，回退到最近可用缓存（{len(data)} 场；不生成当前赛季虚假数据）")
            else:
                raise RuntimeError(f"{code} 没有可用的 Football-data 数据")
        else:
            data = pd.concat(frames, ignore_index=True)
            odds_quality_report(data, code)
            use_xg = bool(cfg.get("xg"))
            if use_xg:
                import json as _json
                xg_path = BASE_DIR / "xg_cache.json"
                cache = _json.load(open(xg_path, encoding="utf-8")) if xg_path.exists() else {}
                data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
                data["_k"] = [f"{d.strftime('%Y-%m-%d')}|{norm(h)}|{norm(a)}" for d, h, a in zip(data["Date"], data["HomeTeam"], data["AwayTeam"])]
                data["home_xg"] = [(cache.get(k) or {}).get("h") for k in data["_k"]]
                data["away_xg"] = [(cache.get(k) or {}).get("a") for k in data["_k"]]
                matched = int(data[["home_xg", "away_xg"]].notna().all(axis=1).sum())
                if matched < 50:
                    use_xg = False
                    data = data.drop(columns=["_k", "home_xg", "away_xg"])
                    print(f"    xG 缓存仅匹配 {matched} 场，低于 50，跳过 xG 特征（不编造）")
                else:
                    data = data.drop(columns=["_k"]).dropna(subset=["home_xg", "away_xg"])
                    print(f"    xG 合并：保留 {len(data)} 场")
            data, FEATURES = build_features(data, with_xg=use_xg)
            with open(data_pkl, "wb") as f:
                pickle.dump((data, FEATURES), f)
            json.dump({"seasons": list(cfg["seasons"]), "with_xg": use_xg}, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print(f"  特征构建完成并缓存（{len(data)} 场，特征 {len(FEATURES)}）")
    if not any(item.get("league") == code for item in QUALITY_ROWS):
        odds_quality_report(data, code)
    import xgboost as xgb
    data = data.sort_values("Date").reset_index(drop=True)
    split = int(len(data) * 0.90)
    train, test = data.iloc[:split], data.iloc[split:]
    cpath, rhpath, rapath = f"{MODEL_DIR}/{code}_clf.json", f"{MODEL_DIR}/{code}_regh.json", f"{MODEL_DIR}/{code}_rega.json"
    if cache_valid and _os.path.exists(cpath) and _os.path.exists(rhpath) and _os.path.exists(rapath):
        clf = xgb.XGBClassifier(); clf.load_model(cpath)
        reg_h = xgb.XGBRegressor(); reg_h.load_model(rhpath)
        reg_a = xgb.XGBRegressor(); reg_a.load_model(rapath)
        print("  从缓存加载模型")
    else:
        clf = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, objective="multi:softprob", num_class=3,
                                eval_metric="mlogloss", tree_method="hist", n_jobs=4, verbosity=0)
        clf.fit(train[FEATURES], train["y"])
        reg_h = xgb.XGBRegressor(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, tree_method="hist", n_jobs=4, verbosity=0)
        reg_a = xgb.XGBRegressor(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, tree_method="hist", n_jobs=4, verbosity=0)
        reg_h.fit(train[FEATURES], train["FTHG"]); reg_a.fit(train[FEATURES], train["FTAG"])
        clf.save_model(cpath); reg_h.save_model(rhpath); reg_a.save_model(rapath)
    p = clf.predict_proba(test[FEATURES])
    mkt = test[["mkt_b365_h", "mkt_b365_d", "mkt_b365_a"]].to_numpy()
    mkt = mkt / mkt.sum(axis=1, keepdims=True)
    yt = test["y"].values
    def onehot(v, k): return 1.0 if v == k else 0.0
    b_model = np.mean([np.sum((p[i] - np.array([onehot(yt[i], k) for k in range(3)])) ** 2) for i in range(len(p))])
    b_mkt = np.mean([np.sum((mkt[i] - np.array([onehot(yt[i], k) for k in range(3)])) ** 2) for i in range(len(mkt))])
    eps = 1e-9
    ll_model = -np.mean([np.log(max(p[i, yt[i]], eps)) for i in range(len(p))])
    ll_mkt = -np.mean([np.log(max(mkt[i, yt[i]], eps)) for i in range(len(mkt))])
    print(f"  测试 {len(test)} 场 | 模型Brier {b_model:.4f} vs 市场Brier {b_mkt:.4f} | 模型LL {ll_model:.4f} vs 市场LL {ll_mkt:.4f}")
    env_report = environment_correlation(data, code)
    print(f"  环境评分 Pearson r={env_report.get('pearson_r')} | {env_report.get('decision')}")
    return {"code": code, "name": cfg["name"], "data": data, "FEATURES": FEATURES, "clf": clf,
            "reg_h": reg_h, "reg_a": reg_a, "test": test, "brier_model": b_model, "brier_mkt": b_mkt,
            "ll_model": ll_model, "ll_mkt": ll_mkt, "env_report": env_report}

# ---------------- 模块一：伤停（API-Football）----------------
class InjuryModule:
    def __init__(self):
        self.key = os.environ.get("API_FOOTBALL_KEY", "").strip()
        self.available = bool(self.key)
    def apply(self, home, away, lam_h, lam_a):
        if not self.available:
            return lam_h, lam_a, "伤停数据未取得（未配置 API_FOOTBALL_KEY）", 0.0, False, []
        # 有 key 时的真实流程（示例：api-football /injuries?fixture=xxx）
        # inj = requests.get(f"https://v3.football.api-sports.io/injuries?fixture={fid}",
        #                    headers={"x-apisports-key": self.key}).json()
        # 解析出双方伤停：{player, position, reason}
        # 影响规则（按指令）：
        #   主力门将缺阵: 对方进球期望 +10%
        #   主力中卫缺阵: 本队失球期望 +8%
        #   主力前锋缺阵: 本队进球期望 -7%
        #   核心中场缺阵: 本队进球和失球各 ±5%
        #   两名以上主力缺阵: 冷门警报
        injuries_h, injuries_a = [], []  # 从接口解析
        impact = 0.0
        cold = False
        for team_inj, side in ((injuries_h, 0), (injuries_a, 1)):
            key_starters = [p for p in team_inj if p.get("主力")]
            if len(key_starters) >= 2:
                cold = True
            for p in key_starters:
                pos = p.get("position", "")
                if pos == "门将" and side == 1:
                    lam_h *= 1.10; impact += 10
                if pos == "门将" and side == 0:
                    lam_a *= 1.10; impact += 10
                if pos == "中卫":
                    (lam_a if side == 0 else lam_h) and None
                    if side == 0: lam_a *= 1.08
                    else: lam_h *= 1.08
                    impact += 8
                if pos == "前锋":
                    if side == 0: lam_h *= 0.93
                    else: lam_a *= 0.93
                    impact += 7
                if pos == "中场":
                    if side == 0:
                        lam_h *= 0.95; lam_a *= 1.05
                    else:
                        lam_a *= 0.95; lam_h *= 1.05
                    impact += 5
        if cold:
            desc = "伤停冷门警报：双方合计2名以上主力缺阵，方向概率降权"
        else:
            desc = f"伤停影响评分 {min(impact, 10):.0f}/10"
        return lam_h, lam_a, desc, min(impact, 10), cold, (injuries_h, injuries_a)

# ---------------- 模块三：盘口变化 ----------------
def odds_movement(row):
    """B365 开盘 vs AvgC 最新快照；文字只描述赔率，不把概率变化当赔率变化。"""
    try:
        o = np.array([row["B365H"], row["B365D"], row["B365A"]], dtype=float)
        c = np.array([row["AvgCH"], row["AvgCD"], row["AvgCA"]], dtype=float)
        if not np.isfinite(o).all() or not np.isfinite(c).all() or (o <= 1.01).any() or (c <= 1.01).any():
            return "盘口变化：未取得（暂无收盘盘）", "未知", False, 0.0
        # 热门方只在主/客胜中选开盘赔率更低者，避免把平局当热门方。
        fav = 0 if o[0] <= o[2] else 2
        odds_chg = (c - o) / o
        hot_chg = float(odds_chg[fav])
        draw_chg = float(odds_chg[1])
        descs = []
        cold = False
        if hot_chg < -0.05:
            descs.append(f"热门方赔率下降{abs(hot_chg)*100:.1f}% → 市场信心增强，方向可信度上调")
        elif hot_chg > 0.05:
            descs.append(f"热门方赔率上升{hot_chg*100:.1f}% → 市场信心减弱，冷门警报")
            cold = True
        if draw_chg < -0.08:
            descs.append(f"平局赔率下降{abs(draw_chg)*100:.1f}% → 市场防平，平局概率上调")
        if not descs:
            descs.append("盘口变化幅度小，无明显信号")
        conf = "强" if hot_chg < -0.05 else ("弱" if hot_chg > 0.05 else "中性")
        return "；".join(descs), conf, cold, hot_chg
    except Exception:
        return "盘口变化：未取得", "未知", False, 0.0

# ---------------- 模块四：投注策略 ----------------
def cold_risk_from_signals(odds_cold, injury_score, env_score_value, direction_disagree):
    """冷门风险必须输出具体来源，不能只给高/中/低。"""
    sources = []
    if odds_cold:
        sources.append("盘口异常（热门方赔率反向上升）")
    if injury_score >= 6:
        sources.append(f"伤停影响评分 {injury_score:.0f}/10")
    if env_score_value <= 3:
        sources.append(f"环境评分偏低（{env_score_value:.1f}/10）")
    if direction_disagree:
        sources.append("模型与市场方向不一致且 Edge 为负")
    level = "高" if len(sources) >= 2 else ("中" if sources else "低")
    return level, (sources or ["无明显冷门信号"])


def strategy(p_model, p_market, odds_home, draw, away, cold_risk, cold_sources=None):
    odds = np.array([odds_home, draw, away], dtype=float)
    mkt = np.array(p_market, dtype=float); mkt = mkt / mkt.sum()
    k = int(np.argmax(p_model))
    edge = p_model[k] - mkt[k]
    lab = ["主胜", "平局", "客胜"]
    big_fav = mkt.max() >= 0.70
    # 动作与仓位严格联动：高 Edge 也必须经过冷门风险分层。
    # 低风险的高价值信号才允许“可介入”；中风险降级为轻仓，高风险观望。
    if edge >= 0.10 and cold_risk == "高":
        action, stake_pct = "观望", 0.0
    elif edge >= 0.10 and cold_risk == "中":
        action, stake_pct = "轻仓观察", 1.5
    elif edge >= 0.10 and cold_risk == "低":
        action, stake_pct = "可介入", 3.0
    elif 0.05 <= edge < 0.10 and cold_risk in ("低", "中"):
        action, stake_pct = "轻仓观察", 1.0
    elif edge < 0:
        action, stake_pct = "放弃", 0.0
    else:
        action, stake_pct = "观望", 0.0
    # 大热场次只有 Edge >= 5% 才能推荐。
    if big_fav and edge < 0.05:
        action, stake_pct = "观望", 0.0
    # 凯利（1/4）
    b = odds[k] - 1
    if b > 0 and np.isfinite(p_model[k]):
        kelly = max((odds[k] * p_model[k] - 1) / b, 0.0)
    else:
        kelly = 0.0
    # 凯利只保留为理论参考，不得覆盖风控仓位。
    note = "理论仓位，实际不建议介入" if action in ("观望", "放弃") else "按风险规则计算的参考仓位"
    return {"方向": lab[k], "edge": edge,
            "edge_by_outcome": [float(x) for x in (np.asarray(p_model) - mkt)],
            "action": action, "kelly": kelly, "建议仓位": f"{stake_pct:.1f}%",
            "stake_pct": stake_pct, "stake_note": note, "大热": big_fav,
            "cold_sources": cold_sources or []}

# ---------------- 模块五：环境与心理 ----------------
def get_weather(city, lat, lon, date):
    try:
        d = date.strftime("%Y-%m-%d")
        u = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
             f"&start_date={d}&end_date={d}&daily=temperature_2m_max,precipitation_probability_max,wind_speed_10m_max&timezone=auto")
        r = cr.get(u, impersonate="chrome", timeout=20, headers=HEADERS)
        j = r.json()["daily"]
        t = j["temperature_2m_max"][0]; pp = j["precipitation_probability_max"][0]; ws = j["wind_speed_10m_max"][0]
        tags = []
        if pp is not None and pp > 50: tags.append("雨战")
        if t is not None and t > 30: tags.append("高温")
        if ws is not None and ws > 30: tags.append("大风")
        if not tags: tags.append("无影响")
        return f"天气：{city} {t:.0f}℃ 降水{pp:.0f}% 风速{ws:.0f}km/h → {'/'.join(tags)}"
    except Exception as e:
        return "天气：未取得"

def final_rank(df, season, team):
    sub = df[df["Season"] == season]
    pts = {}
    for _, r in sub.iterrows():
        hp, ap = (3, 0) if r["FTHG"] > r["FTAG"] else ((1, 1) if r["FTHG"] == r["FTAG"] else (0, 3))
        pts[r["HomeTeam"]] = pts.get(r["HomeTeam"], 0) + hp
        pts[r["AwayTeam"]] = pts.get(r["AwayTeam"], 0) + ap
    ranked = sorted(pts, key=lambda t: -pts[t])
    return ranked.index(team) + 1

def motivation(rank):
    if rank <= 4: return "高（争冠/欧战）"
    if rank <= 6: return "中高（欧战资格）"
    if rank >= 17: return "高（保级）"
    return "中（中游）"

def intl_break_flag(date):
    for mo, sd, ed in INTL_BREAKS:
        start = datetime(date.year, mo, sd); end = datetime(date.year, mo, ed)
        if start <= date <= end + timedelta(days=3):
            return "国脚疲劳风险（国家队比赛日后3天内）"
    return ""

def streak_flag(team, df, date):
    sub = df[(df["Date"] < date) & ((df["HomeTeam"] == team) | (df["AwayTeam"] == team))].tail(5)
    if len(sub) == 0: return "无近期数据"
    w = 0
    for _, r in sub.iterrows():
        is_home = r["HomeTeam"] == team
        gf, ga = (r["FTHG"], r["FTAG"]) if is_home else (r["FTAG"], r["FTHG"])
        if gf > ga: w += 1
        elif gf < ga: w -= 1
    return f"近5场净胜{w}场" + ("（状态好）" if w >= 3 else ("（状态差）" if w <= -3 else ""))

def env_score(weather, motivation, streak, cold_risk):
    s = 5.0
    if "雨战" in weather or "大风" in weather: s -= 1
    if "高温" in weather: s -= 0.5
    if motivation in ("高（争冠/欧战）", "高（保级）"): s += 1
    if "状态好" in streak: s += 1
    if "状态差" in streak: s -= 1
    if cold_risk == "高": s -= 2
    elif cold_risk == "中": s -= 1
    return max(0, min(10, s))


def environment_correlation(data, league_code):
    """用历史 CSV 可取得字段做环境代理评分，并报告与赛果的 Pearson 相关。

    Football-data 没有历史天气、裁判心理和更衣室字段，因此这里明确标记为
    proxy：赛程密度 + 近期状态 + 动机排名。若相关性低于 0.05，环境信号不
    参与推荐权重，只保留展示和冷门审查用途。
    """
    d = data.copy()
    required = ["FTHG", "FTAG", "H_dens", "A_dens"]
    if any(c not in d.columns for c in required):
        result = {"league": league_code, "status": "未取得", "reason": "历史数据缺少环境代理字段"}
        ENV_ROWS.append(result)
        return result
    y = np.where(d["FTHG"] > d["FTAG"], 1.0, np.where(d["FTHG"] == d["FTAG"], 0.5, 0.0))
    stress = np.clip(pd.to_numeric(d["H_dens"], errors="coerce").fillna(0) + pd.to_numeric(d["A_dens"], errors="coerce").fillna(0) - 1.0, 0, 4)
    form_h = pd.to_numeric(d["H_all_GF_5"], errors="coerce").fillna(0) if "H_all_GF_5" in d.columns else pd.Series(0.0, index=d.index)
    form_a = pd.to_numeric(d["A_all_GF_5"], errors="coerce").fillna(0) if "A_all_GF_5" in d.columns else pd.Series(0.0, index=d.index)
    score = np.clip(5.0 - 0.8 * stress + np.clip(form_h - form_a, -2, 2) * 0.35, 0, 10)
    valid = np.isfinite(score) & np.isfinite(y)
    corr = float(np.corrcoef(score[valid], y[valid])[0, 1]) if valid.sum() >= 3 and np.std(score[valid]) > 0 else None
    result = {
        "league": league_code, "status": "已计算" if corr is not None else "未取得",
        "method": "proxy（历史 CSV 无天气/裁判/心理字段）", "samples": int(valid.sum()),
        "pearson_r": corr, "abs_r": None if corr is None else abs(corr),
        "decision": "保留环境信号" if corr is not None and abs(corr) >= 0.05 else "降低权重：不参与推荐，仅用于展示/冷门审查",
    }
    ENV_ROWS.append(result)
    return result


# ---------------- 官方竞彩当前在售比赛池 ----------------
LOTTERY_DB = BASE_DIR / "runtime" / "football-analysis" / "data" / "football_analysis.sqlite3"
UCL_HISTORY_URL = "https://raw.githubusercontent.com/CharlieGnomo/champions_uefa_data/master/matches.csv"
UCL_HISTORY_CACHE = REPORT_DIR / "ucl_matches.csv"

UCL_TEAM_ALIASES = {
    "lask林茨": "lask", "lask": "lask", "lask linz": "lask", "lasklinz": "lask",
    "凯尔特人": "celticfc", "celtic": "celticfc", "celticfc": "celticfc",
    "博德闪耀": "fk bodoglimt", "fk bodoglimt": "fk bodoglimt", "bodoglimt": "fk bodoglimt",
    "奈梅亨": "nec nijmegen", "nijmegen": "nec nijmegen", "nec nijmegen": "nec nijmegen",
}

def _ucl_team_key(name):
    text = str(name or "").strip().lower()
    if text in UCL_TEAM_ALIASES:
        return UCL_TEAM_ALIASES[text]
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return UCL_TEAM_ALIASES.get(text, text)


def load_ucl_history():
    """Load real UEFA match results; this source has results but no odds/xG."""
    try:
        if UCL_HISTORY_CACHE.exists():
            raw = UCL_HISTORY_CACHE.read_bytes()
        else:
            resp = cr.get(UCL_HISTORY_URL, impersonate="chrome", timeout=45, headers=HEADERS)
            resp.raise_for_status()
            raw = resp.content
            UCL_HISTORY_CACHE.write_bytes(raw)
        df = pd.read_csv(io.BytesIO(raw), sep=";")
        df = df[df["status"].astype(str).str.upper().eq("FINISHED")].copy()
        df["FTHG"] = pd.to_numeric(df["ft1"], errors="coerce")
        df["FTAG"] = pd.to_numeric(df["ft2"], errors="coerce")
        df["HomeTeam"] = df["t1_name"].astype(str)
        df["AwayTeam"] = df["t2_name"].astype(str)
        df = df.dropna(subset=["FTHG", "FTAG", "HomeTeam", "AwayTeam"])
        return df[["date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].reset_index(drop=True)
    except Exception as exc:
        print(f"  欧冠历史数据读取失败：{exc}")
        return pd.DataFrame()


def load_ucl_domestic_support():
    """Load recent domestic results for UCL teams when UEFA history is sparse."""
    frames = []
    sources = [
        ("AUT", "https://www.football-data.co.uk/new/AUT.csv"),
        ("NOR", "https://www.football-data.co.uk/new/NOR.csv"),
    ]
    for comp, url in sources:
        try:
            cache = REPORT_DIR / f"ucl_{comp.lower()}.csv"
            if cache.exists():
                raw = cache.read_bytes()
            else:
                resp = cr.get(url, impersonate="chrome", timeout=45, headers=HEADERS)
                resp.raise_for_status(); raw = resp.content; cache.write_bytes(raw)
            d = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
            d = d.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam", "HG": "FTHG", "AG": "FTAG"})
            d["FTHG"] = pd.to_numeric(d["FTHG"], errors="coerce"); d["FTAG"] = pd.to_numeric(d["FTAG"], errors="coerce")
            d = d.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
            d["_comp"] = comp; frames.append(d)
        except Exception as exc:
            print(f"  欧冠辅助数据 {comp} 未取得：{exc}")
    # 本项目已同步的荷甲历史/当前赛季数据，覆盖奈梅亨。
    try:
        n1_path = BASE_DIR / "models" / "N1_data.pkl"
        if n1_path.exists():
            import pickle
            n1, _ = pickle.load(open(n1_path, "rb"))
            n1 = n1.rename(columns={"HomeTeam": "HomeTeam", "AwayTeam": "AwayTeam", "FTHG": "FTHG", "FTAG": "FTAG"})
            n1 = n1[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].copy(); n1["_comp"] = "NED"; frames.append(n1)
    except Exception as exc:
        print(f"  欧冠辅助数据 NED 未取得：{exc}")
    # 苏格兰仅取当前/前两季，避免大量旧数据掩盖当前状态。
    for season in SEASON_CODES:
        try:
            resp = cr.get(f"https://www.football-data.co.uk/mmz4281/{season}/SC0.csv", impersonate="chrome", timeout=25, headers=HEADERS)
            if resp.status_code != 200: continue
            d = pd.read_csv(io.BytesIO(resp.content), encoding="latin-1")
            d = d.rename(columns={"HomeTeam": "HomeTeam", "AwayTeam": "AwayTeam", "FTHG": "FTHG", "FTAG": "FTAG"})
            d["FTHG"] = pd.to_numeric(d["FTHG"], errors="coerce"); d["FTAG"] = pd.to_numeric(d["FTAG"], errors="coerce")
            d = d.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
            d["_comp"] = "SCO"; frames.append(d)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def ucl_pool_models(pool):
    """Independent Poisson model for UCL pool, based on real UEFA historical goals.

    No market odds are used as model features. Teams with fewer than 3 historical
    UCL matches are kept as low-coverage/observation only.
    """
    hist = load_ucl_history()
    if hist.empty:
        return {}
    hist["_h"] = hist["HomeTeam"].map(_ucl_team_key)
    hist["_a"] = hist["AwayTeam"].map(_ucl_team_key)
    base_h = float(hist["FTHG"].mean())
    base_a = float(hist["FTAG"].mean())
    domestic = load_ucl_domestic_support()
    if not domestic.empty:
        domestic["_h"] = domestic["HomeTeam"].map(_ucl_team_key)
        domestic["_a"] = domestic["AwayTeam"].map(_ucl_team_key)
    def stats(team):
        t = _ucl_team_key(team)
        h = hist[hist["_h"] == t]; a = hist[hist["_a"] == t]
        n = len(h) + len(a)
        if n == 0:
            return {"n": 0, "ha": np.nan, "hd": np.nan, "aa": np.nan, "ad": np.nan}
        return {"n": n, "ha": h["FTHG"].mean() if len(h) else np.nan,
                "hd": h["FTAG"].mean() if len(h) else np.nan,
                "aa": a["FTAG"].mean() if len(a) else np.nan,
                "ad": a["FTHG"].mean() if len(a) else np.nan}
    def domestic_stats(team):
        if domestic.empty: return None
        t = _ucl_team_key(team); h = domestic[domestic["_h"] == t]; a = domestic[domestic["_a"] == t]
        n = len(h) + len(a)
        if n == 0: return None
        return {"n": n, "ha": h["FTHG"].mean() if len(h) else np.nan,
                "hd": h["FTAG"].mean() if len(h) else np.nan,
                "aa": a["FTAG"].mean() if len(a) else np.nan,
                "ad": a["FTHG"].mean() if len(a) else np.nan}
    def shrink(value, n, prior):
        if not np.isfinite(value):
            return prior
        return (n * float(value) + 10.0 * prior) / (n + 10.0)
    dom_h = float(domestic["FTHG"].mean()) if not domestic.empty else base_h
    dom_a = float(domestic["FTAG"].mean()) if not domestic.empty else base_a
    def profile(team):
        sources = []
        us = stats(team)
        if us["n"]:
            sources.append((us, base_h, base_a, min(us["n"], 30)))
        ds = domestic_stats(team)
        if ds:
            sources.append((ds, dom_h, dom_a, min(ds["n"], 30)))
        if not sources:
            return {"n": 0, "attack_h": 1.0, "def_h": 1.0, "attack_a": 1.0, "def_a": 1.0}
        total = sum(float(x[3]) for x in sources)
        vals = []
        for s, bh, ba, weight in sources:
            ha = shrink(s["ha"], s["n"], bh); hd = shrink(s["hd"], s["n"], ba)
            aa = shrink(s["aa"], s["n"], ba); ad = shrink(s["ad"], s["n"], bh)
            vals.append((weight, ha / bh, hd / ba, aa / ba, ad / bh))
        return {"n": int(sum(s["n"] for s, _, _, _ in sources)),
                "attack_h": sum(w * x for w, x, _, _, _ in vals) / total,
                "def_h": sum(w * x for w, _, x, _, _ in vals) / total,
                "attack_a": sum(w * x for w, _, _, x, _ in vals) / total,
                "def_a": sum(w * x for w, _, _, _, x in vals) / total}
    out = {}
    for item in pool:
        hp, ap = profile(item["home"]), profile(item["away"])
        lam_h = max(0.05, base_h * hp["attack_h"] * ap["def_a"])
        lam_a = max(0.05, base_a * ap["attack_a"] * hp["def_h"])
        ph = poisson.pmf(np.arange(11), lam_h); pa = poisson.pmf(np.arange(11), lam_a)
        probs = np.array([
            sum(ph[i] * pa[j] for i in range(11) for j in range(11) if i > j),
            sum(ph[i] * pa[i] for i in range(11)),
            sum(ph[i] * pa[j] for i in range(11) for j in range(11) if i < j),
        ])
        coverage = min(hp["n"], ap["n"])
        out[item["lottery_id"]] = {
            "model_prob": probs / probs.sum(), "exp_goals": [lam_h, lam_a],
            "coverage": coverage, "coverage_home": hp["n"], "coverage_away": ap["n"],
            "sources": ["UEFA历史赛果", "本土联赛赛果支持"],
            "model_status": "欧冠混合模型（欧冠历史+本土联赛）" if coverage >= 10 else "欧冠低覆盖（仅方向参考）" if coverage >= 3 else "欧冠历史样本不足",
        }
    return out

def _parse_bj_time(value):
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=BJ_TZ)
        return dt.astimezone(BJ_TZ)
    except Exception:
        return None


def _odds_move_detail(opening, latest):
    """将官方赔率初盘与最新快照统一为可供两个前端使用的结构。"""
    detail = {
        "opening": opening,
        "latest": latest,
        "change": None,
        "hot_outcome": None,
        "hot_change": None,
        "draw_change": None,
        "signal": "未取得",
        "cold_alert": False,
        "text": "盘口变化：未取得",
    }
    try:
        if not opening or not latest:
            if latest:
                detail["latest"] = latest
                detail["text"] = "初盘未取得；最新赔率：" + "/".join(f"{float(x):.2f}" for x in latest)
            return detail
        o = np.asarray(opening, dtype=float); c = np.asarray(latest, dtype=float)
        if not np.isfinite(o).all() or not np.isfinite(c).all() or (o <= 1.01).any() or (c <= 1.01).any():
            detail["text"] = "盘口变化：赔率快照异常或未取得"
            return detail
        chg = (c - o) / o
        fav = int(np.argmin([o[0], o[2]]))
        fav_outcome = "主胜" if fav == 0 else "客胜"
        hot_chg = float(chg[fav]); draw_chg = float(chg[1])
        detail.update({
            "opening": [float(x) for x in o], "latest": [float(x) for x in c],
            "change": [float(x) for x in chg], "hot_outcome": fav_outcome,
            "hot_change": hot_chg, "draw_change": draw_chg,
        })
        parts = ["初盘 " + "/".join(f"{x:.2f}" for x in o) + " → 最新 " + "/".join(f"{x:.2f}" for x in c)]
        if hot_chg < -0.05:
            parts.append(f"热门方赔率下降{abs(hot_chg)*100:.1f}% → 市场信心增强")
            detail["signal"] = "市场信心增强"
        elif hot_chg > 0.05:
            parts.append(f"热门方赔率上升{hot_chg*100:.1f}% → 市场信心减弱，冷门警报")
            detail["signal"] = "市场信心减弱"
            detail["cold_alert"] = True
        if draw_chg < -0.08:
            parts.append(f"平局赔率下降{abs(draw_chg)*100:.1f}% → 市场防平")
        if len(parts) == 1:
            parts.append("变化未超过提示阈值")
            detail["signal"] = "无明显变化"
        detail["text"] = "；".join(parts)
    except Exception:
        pass
    return detail


def load_lottery_pool(hours=36):
    """Read the latest official open fixtures and odds from the synced SQLite store."""
    if not LOTTERY_DB.exists():
        print(f"  竞彩数据库不存在：{LOTTERY_DB}")
        return []
    now = datetime.now(BJ_TZ)
    start = now - timedelta(hours=1)
    end = now + timedelta(hours=hours)
    try:
        con = sqlite3.connect(str(LOTTERY_DB))
        con.row_factory = sqlite3.Row
        fixtures = con.execute(
            "SELECT * FROM fixtures WHERE status='open' ORDER BY kickoff_at"
        ).fetchall()
        pool = []
        for f in fixtures:
            kickoff = _parse_bj_time(f["kickoff_at"])
            if kickoff is None or not (start <= kickoff <= end):
                continue
            odds_rows = con.execute(
                "SELECT * FROM odds_snapshots WHERE lottery_id=? ORDER BY observed_at DESC, id DESC",
                (f["lottery_id"],),
            ).fetchall()
            one_rows = [o for o in odds_rows if str(o["market"]).lower() in {"1x2", "胜平负"}]
            one = one_rows[0] if one_rows else None
            latest_one = one_rows[0] if one_rows else None
            opening_one = one_rows[-1] if one_rows else None
            handicap = next((o for o in odds_rows if str(o["market"]).lower() in {"让球胜平负", "hhad"}), None)
            market_odds = None
            market_prob = None
            if one and all(one[k] is not None for k in ("home", "draw", "away")):
                market_odds = [float(one["home"]), float(one["draw"]), float(one["away"])]
                p = clean_prob(*market_odds)
                if np.isfinite(p).all():
                    market_prob = [round(float(x), 6) for x in p]
            pool.append({
                "lottery_id": f["lottery_id"], "league": f["competition"],
                "home": f["home_team"], "away": f["away_team"],
                "kickoff_at": kickoff.isoformat(timespec="minutes"),
                "market_odds": market_odds, "market_prob": market_prob,
                "opening_market_odds": None if opening_one is None else [opening_one[k] for k in ("home", "draw", "away")],
                "latest_market_odds": None if latest_one is None else [latest_one[k] for k in ("home", "draw", "away")],
                "odds_move_detail": _odds_move_detail(
                    None if opening_one is None else [opening_one[k] for k in ("home", "draw", "away")],
                    None if latest_one is None else [latest_one[k] for k in ("home", "draw", "away")],
                ),
                "odds_source": None if one is None else one["source"],
                "odds_stage": None if one is None else one["stage"],
                "handicap": None if handicap is None else {
                    "line": handicap["handicap"], "home": handicap["home"],
                    "draw": handicap["draw"], "away": handicap["away"],
                    "source": handicap["source"], "stage": handicap["stage"],
                },
            })
        con.close()
        return pool
    except Exception as exc:
        print(f"  竞彩数据库读取失败：{exc}")
        return []


def rows_from_lottery_pool(pool, model_map=None):
    """Create rows for the shared official pool with explicit model fallback semantics."""
    rows = []
    for item in pool:
        mp = item.get("market_prob")
        if mp and np.isfinite(mp).all():
            direction = ["主胜", "平局", "客胜"][int(np.argmax(mp))]
            market_text = [round(float(x), 4) for x in mp]
        else:
            direction, market_text = "无明显方向", None
        kickoff = _parse_bj_time(item["kickoff_at"])
        date = kickoff.strftime("%Y-%m-%d") if kickoff else datetime.now(BJ_TZ).strftime("%Y-%m-%d")
        odds_detail = item.get("odds_move_detail") or _odds_move_detail(
            item.get("opening_market_odds"), item.get("latest_market_odds") or item.get("market_odds")
        )
        odds_desc = odds_detail.get("text") or "官方赔率未取得"
        model = (model_map or {}).get(item["lottery_id"])
        row = {
            "lottery_id": item["lottery_id"], "league": item["league"], "date": date,
            "kickoff_bjt": item["kickoff_at"], "home": item["home"], "away": item["away"],
            "model_status": "模型待训练/样本不足，暂用市场概率", "model_prob": market_text, "market_prob": market_text,
            "model_confidence": "低", "model_prob_source": "market_fallback",
            "edge": 0.0 if market_text is not None else None,
            "edge_by_outcome": [0.0, 0.0, 0.0] if market_text is not None else None, "direction": direction,
            "action": "观望", "stake": "0.0%", "stake_pct": 0.0,
            "stake_note": "模型待训练/样本不足，暂用市场概率；实际不建议介入", "big_fav": bool(mp and max(mp) >= .70),
            "exp_goals": None, "over25": None, "home_rank": None,
            "injury": "未取得", "injury_score": None, "odds_move": odds_desc,
            "odds_move_detail": odds_detail, "market_conf": "官方竞彩", "cold_risk": "未取得",
            "cold_risk_sources": ["伤停未取得", "环境未取得", "盘口异常未取得"], "weather": "未取得",
            "referee": "未取得", "tactics": "未取得", "motivation": "未取得",
            "intl_break": "未取得", "streak": "未取得", "env_score": None,
            "actual": None, "score": None,
            "prediction_view": {"model_status": "模型待训练/样本不足，暂用市场概率", "model_confidence": "低",
                                "model_prob": market_text, "direction": direction, "edge": 0.0 if market_text is not None else None,
                                "action": "观望", "stake_pct": 0.0, "cold_risk": "未取得"},
            "market_view": {"odds": item.get("market_odds"), "market_prob": market_text,
                            "opening_odds": item.get("opening_market_odds"), "latest_odds": item.get("latest_market_odds") or item.get("market_odds"),
                            "odds_move": odds_detail,
                            "odds_source": item.get("odds_source"), "odds_stage": item.get("odds_stage"),
                            "handicap": item.get("handicap")},
        }
        if model and market_text is not None:
            model_prob = np.asarray(model["model_prob"], dtype=float)
            market_prob = np.asarray(market_text, dtype=float)
            coverage = int(model.get("coverage", 0))
            # 欧冠历史样本不足时完全回退到市场去水概率，禁止输出固定占位概率。
            if coverage < 10:
                fallback_status = "欧冠样本不足，暂用市场概率"
                row.update({
                    "model_status": fallback_status, "model_prob": [round(float(x), 4) for x in market_prob],
                    "market_prob": [round(float(x), 4) for x in market_prob], "model_confidence": "低",
                    "model_prob_source": "market_fallback", "edge": 0.0,
                    "edge_by_outcome": [0.0, 0.0, 0.0], "direction": direction,
                    "action": "观望", "stake": "0.0%", "stake_pct": 0.0,
                    "stake_note": "历史样本不足，暂用市场概率；实际不建议介入",
                    "cold_risk": "未取得", "cold_risk_sources": ["伤停未取得", "环境未取得", "盘口异常未取得"],
                    "model_coverage": {"home": int(model.get("coverage_home", 0)), "away": int(model.get("coverage_away", 0)),
                                       "total": coverage, "minimum_for_intervention": 10},
                    "model_sources": model.get("sources", ["欧冠历史+本土联赛"]),
                    "prediction_view": {"model_status": fallback_status, "model_confidence": "低",
                                        "model_prob": [round(float(x), 4) for x in market_prob], "direction": direction,
                                        "edge": 0.0, "action": "观望", "stake_pct": 0.0, "cold_risk": "未取得"},
                })
            else:
                cold_risk, cold_sources = cold_risk_from_signals(
                    bool(odds_detail.get("cold_alert")), 0.0, 5.0, False
                )
                st = strategy(model_prob, market_prob, *(item.get("market_odds") or [np.nan, np.nan, np.nan]),
                              cold_risk, cold_sources)
                edge = float(st["edge"])
                row.update({
                    "model_status": model["model_status"], "model_prob": [round(float(x), 4) for x in model_prob],
                    "market_prob": [round(float(x), 4) for x in market_prob], "model_confidence": "高",
                    "model_prob_source": "trained_model", "edge": round(edge, 4),
                    "edge_by_outcome": [round(float(x), 4) for x in st["edge_by_outcome"]],
                    "direction": st["方向"], "action": st["action"], "stake": f"{st['stake_pct']:.1f}%",
                    "stake_pct": st["stake_pct"], "stake_note": st["stake_note"], "exp_goals": [round(float(x), 2) for x in model["exp_goals"]],
                    "over25": round(float(1 - sum(poisson.pmf(i, model["exp_goals"][0]) * poisson.pmf(j, model["exp_goals"][1])
                                                  for i in range(11) for j in range(11) if i + j <= 2)), 4),
                    "injury": "未取得", "injury_score": None, "cold_risk": cold_risk,
                    "cold_risk_sources": cold_sources,
                    "model_coverage": {"home": int(model.get("coverage_home", 0)), "away": int(model.get("coverage_away", 0)),
                                       "total": coverage, "minimum_for_intervention": 10},
                    "model_sources": model.get("sources", ["欧冠历史+本土联赛"]),
                    "prediction_view": {"model_status": model["model_status"], "model_confidence": "高",
                                        "model_prob": [round(float(x), 4) for x in model_prob], "direction": st["方向"],
                                        "edge": round(edge, 4), "action": st["action"], "stake_pct": st["stake_pct"],
                                        "cold_risk": cold_risk, "coverage_home": model["coverage_home"],
                                        "coverage_away": model["coverage_away"]},
                })
        rows.append(row)
    return rows

# ---------------- 主流程 ----------------
def main():
    results = {}
    for code, cfg in LEAGUES.items():
        try:
            results[code] = train_league(code, cfg)
        except Exception as e:
            print(f"  ⚠️ {cfg['name']} 训练失败：{repr(e)[:120]} → 该联赛标'未取得'")
    print("\n" + "=" * 78)
    print("各联赛模型评估汇总（模块二）")
    print("=" * 78)
    print(f"{'联赛':<6}{'测试场次':>8}{'模型Brier':>12}{'市场Brier':>12}{'模型LL':>10}{'市场LL':>10}")
    for code, r in results.items():
        print(f"{r['name']:<6}{len(r['test']):>8}{r['brier_model']:>12.4f}{r['brier_mkt']:>12.4f}{r['ll_model']:>10.4f}{r['ll_mkt']:>10.4f}")
    for lg, why in NO_DATA_LEAGUES.items():
        print(f"{lg}: 未取得（{why}）")

    print("\n" + "=" * 78)
    print("最近一轮（演示）详细报告 + 模块一/三/四/五整合 → predictions.json")
    print("=" * 78)
    injury = InjuryModule()
    print(f"  模块一(伤停): {'已启用(API_FOOTBALL_KEY)' if injury.available else '未取得（未配置 API_FOOTBALL_KEY 环境变量）'}")
    preds = []
    for code, r in results.items():
        test = r["test"]
        last_date = test["Date"].max()
        last = test[test["Date"] == last_date]
        p = r["clf"].predict_proba(last[r["FEATURES"]])
        mkt = last[["mkt_b365_h", "mkt_b365_d", "mkt_b365_a"]].to_numpy(); mkt = mkt / mkt.sum(axis=1, keepdims=True)
        eg_h = np.maximum(r["reg_h"].predict(last[r["FEATURES"]]), 0.05)
        eg_a = np.maximum(r["reg_a"].predict(last[r["FEATURES"]]), 0.05)
        for i, (_, row) in enumerate(last.iterrows()):
            om_desc, om_conf, om_cold, hot_chg = odds_movement(row)
            # 模块五
            w = "天气：未取得"
            if code == "E0" and row["HomeTeam"] in CITY_MAP:
                cname, lat, lon = CITY_MAP[row["HomeTeam"]]
                w = get_weather(cname, lat, lon, row["Date"])
            rank = final_rank(r["data"], row["Season"], row["HomeTeam"])
            mot = motivation(rank)
            ib = intl_break_flag(row["Date"])
            streak = streak_flag(row["HomeTeam"], r["data"], row["Date"])
            # 伤停数据可得时量化；未配置 key 明确标注未取得，不替换成伪数据。
            eg_h_i, eg_a_i, injury_desc, injury_score, injury_cold, injury_items = injury.apply(
                row["HomeTeam"], row["AwayTeam"], float(eg_h[i]), float(eg_a[i]))
            base_env = env_score(w, mot, streak, "低")
            direction_disagree = int(np.argmax(p[i])) != int(np.argmax(mkt[i])) and (p[i].max() - mkt[i].max()) < 0
            cold_risk, cold_sources = cold_risk_from_signals(om_cold or injury_cold, injury_score, base_env, direction_disagree)
            es = env_score(w, mot, streak, cold_risk)
            # 动作与仓位联动；高风险 + 高 Edge 只降为轻仓观察，不直接吞掉价值信号。
            st = strategy(p[i], mkt[i], row["B365H"], row["B365D"], row["B365A"], cold_risk, cold_sources)
            actual = "主胜" if row["FTHG"] > row["FTAG"] else ("平局" if row["FTHG"] == row["FTAG"] else "客胜")
            over = sum(poisson.pmf(i2, eg_h_i) * poisson.pmf(j2, eg_a_i) for i2 in range(11) for j2 in range(11) if i2 + j2 > 2)
            preds.append({
                "league": r["name"], "date": row["Date"].strftime("%Y-%m-%d"),
                "home": row["HomeTeam"], "away": row["AwayTeam"],
                "model_status": "历史训练模型", "model_confidence": "高", "model_prob_source": "trained_model",
                "model_prob": [round(float(x), 3) for x in p[i]],
                "market_prob": [round(float(x), 3) for x in mkt[i]],
                "direction": st["方向"], "edge": round(st["edge"], 4), "action": st["action"],
                "edge_by_outcome": [round(float(x), 4) for x in st["edge_by_outcome"]],
                "stake": st["建议仓位"], "stake_pct": st["stake_pct"], "stake_note": st["stake_note"], "big_fav": bool(st["大热"]),
                "exp_goals": [round(float(eg_h_i), 2), round(float(eg_a_i), 2)],
                "over25": round(float(over), 3), "home_rank": int(rank),
                "injury": injury_desc, "injury_score": round(float(injury_score), 1), "odds_move": om_desc, "market_conf": om_conf,
                "cold_risk": cold_risk, "cold_risk_sources": cold_sources, "weather": w, "referee": "未取得（需 API-Football key）",
                "tactics": "未取得（无风格标签数据源）", "motivation": mot,
                "intl_break": ib or "无", "streak": streak, "env_score": round(es, 1),
                "actual": actual, "score": f"{int(row['FTHG'])}-{int(row['FTAG'])}",
                "prediction_view": {"model_status": "历史训练模型", "model_confidence": "高",
                                    "model_prob": [round(float(x), 3) for x in p[i]],
                                    "edge": round(st["edge"], 4), "direction": st["方向"],
                                    "action": st["action"], "stake_pct": st["stake_pct"],
                                    "cold_risk": cold_risk, "env_score": round(es, 1)},
                "market_view": {"odds": [float(row["B365H"]), float(row["B365D"]), float(row["B365A"])],
                                "market_prob": [round(float(x), 3) for x in mkt[i]],
                                 "odds_source": "Bet365/历史快照", "odds_stage": "historical",
                                 "odds_move": {"text": om_desc, "signal": om_conf, "cold_alert": bool(om_cold)},
                                "handicap": None},
            })
    # 官方竞彩池是两个前端的共同比赛源；模型未覆盖的赛事明确略过，不生成估算概率。
    lottery_pool = load_lottery_pool(hours=36)
    data_mode = "竞彩官方在售比赛池" if lottery_pool else "历史验证回退（竞彩当前池未取得）"
    if lottery_pool:
        # 仅对欧冠场次使用欧冠历史+本土联赛混合模型。
        # 其他竞彩网联赛若没有对应历史训练样本，必须明确标记为“模型未覆盖”，
        # 不能套用欧冠分支的兜底概率或状态文案。
        ucl_pool = [item for item in lottery_pool if "欧冠" in str(item.get("league", ""))]
        ucl_map = ucl_pool_models(ucl_pool)
        preds = rows_from_lottery_pool(lottery_pool, ucl_map)
        (REPORT_DIR / "current_pool.json").write_text(
            json.dumps({"generated_at": datetime.now(BJ_TZ).isoformat(timespec="seconds"),
                        "hours": 36, "fixtures": lottery_pool}, ensure_ascii=False, indent=2),
            encoding="utf-8")
    for row in preds:
        row["data_mode"] = data_mode
    with open(QUALITY_REPORT, "w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.now().isoformat(timespec="seconds"), "leagues": QUALITY_ROWS}, f, ensure_ascii=False, indent=2)
    with open(ENV_REPORT, "w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.now().isoformat(timespec="seconds"), "leagues": ENV_ROWS}, f, ensure_ascii=False, indent=2)
    quality_md = ["# 赔率数据质量检查报告", "", f"生成时间：{datetime.now().isoformat(timespec='seconds')}", "", "| 联赛 | 样本 | Pinnacle完整 | Bet365完整 | 去水样本 | 概率范围(主/平/客) | 归一化 |", "|---|---:|---:|---:|---:|---|---|"]
    for q in QUALITY_ROWS:
        dv = q.get("devig", {})
        mins = dv.get("probability_min", [None, None, None]); maxs = dv.get("probability_max", [None, None, None])
        rng = "未取得" if mins[0] is None else " / ".join(f"{a:.3f}-{b:.3f}" for a, b in zip(mins, maxs))
        quality_md.append(f"| {q.get('league')} | {q.get('rows', 0)} | {q.get('selected_source', {}).get('pinnacle', 0)} | {q.get('selected_source', {}).get('bet365', 0)} | {dv.get('valid_samples', 0)} | {rng} | {'通过' if dv.get('normalization_pass') else '未取得'} |")
    quality_md += ["", "异常值定义：赔率 ≤1.01 或 >100 视为异常并剔除；优先使用 Pinnacle 三列完整快照，否则回退 Bet365。去水采用 1/赔率 后按三项归一化。"]
    (REPORT_DIR / "odds_quality_report.md").write_text("\n".join(quality_md), encoding="utf-8")
    env_md = ["# 环境评分相关性报告", "", "历史 Football-data 不含天气、裁判和心理字段，以下为赛程密度 + 近期状态 + 动机排名代理评分。", "", "| 联赛 | 样本 | Pearson r | |r| | 决策 |", "|---|---:|---:|---:|---|"]
    for e in ENV_ROWS:
        env_md.append(f"| {e.get('league')} | {e.get('samples', 0)} | {e.get('pearson_r', '未取得')} | {e.get('abs_r', '未取得')} | {e.get('decision', '未取得')} |")
    (REPORT_DIR / "environment_correlation_report.md").write_text("\n".join(env_md), encoding="utf-8")
    json.dump(preds, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # 静态前端与根目录保持同一份数据，避免部署后页面继续读取旧快照。
    frontend_json = BASE_DIR / "frontend" / "predictions.json"
    try:
        frontend_json.write_text(json.dumps(preds, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError as exc:
        print(f"  前端数据镜像写入失败：{exc}")
    # 打印最近一轮前几条
    for p in preds[:6]:
        print(f"\n[{p['league']}] {p['date']} {p['home']} vs {p['away']}")
        print(f"  模型 {p['model_prob']} | 市场 {p['market_prob']} | 方向 {p['direction']} | Edge {p['edge']}")
        print(f"  动作 {p['action']} | 建议仓位 {p['stake']} | 冷门风险 {p['cold_risk']}（来源：{'、'.join(p['cold_risk_sources']) or '无'}） | 环境评分 {p['env_score']}")
        print(f"  盘口: {p['odds_move']}")
        print(f"  天气: {p['weather']} | 伤停: {p['injury']} | 裁判: {p['referee']} | 战术: {p['tactics']}")
        print(f"  实际: {p['score']}（{p['actual']}）")
    print(f"\n已生成 {OUT_JSON}（共 {len(preds)} 场）")

if __name__ == "__main__":
    main()
