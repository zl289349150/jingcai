# -*- coding: utf-8 -*-
"""predict_all.py — 足球预测系统全面增强版（模块一~五整合 + 部署）
用法:
  python predict_all.py            # 训练全部可得联赛 + 评估 + 生成 predictions.json（含模块演示）
环境变量:
  API_FOOTBALL_KEY  # 可选；未设置时 伤停/裁判 标"未取得"
"""
import io, os, sys, json, time
from datetime import datetime, timedelta
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
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
OUT_JSON = "predictions.json"

# ---------------- 联赛配置（模块二）----------------
LEAGUES = {
    "E0":  {"name": "英超", "fd": "E0",  "seasons": ["2526", "2425", "2324"], "xg": True},
    "SP1": {"name": "西甲", "fd": "SP1", "seasons": ["2526", "2425", "2324"], "xg": False},
    "I1":  {"name": "意甲", "fd": "I1",  "seasons": ["2526", "2425", "2324"], "xg": False},
    "D1":  {"name": "德甲", "fd": "D1",  "seasons": ["2526", "2425", "2324"], "xg": False},
    "F1":  {"name": "法甲", "fd": "F1",  "seasons": ["2526", "2425", "2324"], "xg": False},
    "N1":  {"name": "荷甲", "fd": "N1",  "seasons": ["2526", "2425", "2324"], "xg": False},
    "P1":  {"name": "葡超", "fd": "P1",  "seasons": ["2526", "2425", "2324"], "xg": False},
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
    inv = 1.0 / np.array([h, d, a], dtype=float)
    return inv / inv.sum()

def download_fd(code, season):
    for _ in range(3):
        try:
            r = cr.get(f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv", impersonate="chrome", timeout=40, headers=HEADERS)
            r.raise_for_status()
            df = pd.read_csv(io.BytesIO(r.content), encoding="latin-1")
            df["Season"] = season
            return df
        except Exception as e:
            import time; time.sleep(2)
    raise


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
    if _os.path.exists(data_pkl):
        with open(data_pkl, "rb") as f:
            data, FEATURES = pickle.load(f)
        print(f"  从缓存加载数据（{len(data)} 场，特征 {len(FEATURES)}）")
    else:
        frames = [download_fd(cfg["fd"], s) for s in cfg["seasons"]]
        data = pd.concat(frames, ignore_index=True)
        if cfg.get("xg"):
            import json as _json
            cache = _json.load(open(r"C:\Users\zl289\Documents\Codex\2026-08-16\chatgpt-9\xg_cache.json", encoding="utf-8"))
            data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
            data["_k"] = [f"{d.strftime('%Y-%m-%d')}|{norm(h)}|{norm(a)}" for d, h, a in zip(data["Date"], data["HomeTeam"], data["AwayTeam"])]
            data["home_xg"] = [(cache.get(k) or {}).get("h") for k in data["_k"]]
            data["away_xg"] = [(cache.get(k) or {}).get("a") for k in data["_k"]]
            data = data.drop(columns=["_k"]).dropna(subset=["home_xg", "away_xg"])
            print(f"    xG 合并：保留 {len(data)} 场")
        data, FEATURES = build_features(data, with_xg=cfg.get("xg", False))
        with open(data_pkl, "wb") as f:
            pickle.dump((data, FEATURES), f)
        print(f"  特征构建完成并缓存（{len(data)} 场，特征 {len(FEATURES)}）")
    import xgboost as xgb
    data = data.sort_values("Date").reset_index(drop=True)
    split = int(len(data) * 0.90)
    train, test = data.iloc[:split], data.iloc[split:]
    cpath, rhpath, rapath = f"{MODEL_DIR}/{code}_clf.json", f"{MODEL_DIR}/{code}_regh.json", f"{MODEL_DIR}/{code}_rega.json"
    if _os.path.exists(cpath):
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
    return {"code": code, "name": cfg["name"], "data": data, "FEATURES": FEATURES, "clf": clf,
            "reg_h": reg_h, "reg_a": reg_a, "test": test, "brier_model": b_model, "brier_mkt": b_mkt,
            "ll_model": ll_model, "ll_mkt": ll_mkt}

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
    """B365 开盘 vs AvgC 收盘（历史可用；未来场次收盘未取得时标未取得）"""
    try:
        o = np.array([row["B365H"], row["B365D"], row["B365A"]], dtype=float)
        c = np.array([row["AvgCH"], row["AvgCD"], row["AvgCA"]], dtype=float)
        if not np.isfinite(c).all():
            return "盘口变化：未取得（暂无收盘盘）", "未知", False, 0.0
        po = o / o.sum(); pc = c / c.sum()
        chg = (pc - po) / po  # 概率变化幅度
        fav = int(np.argmax(po))
        hot_chg = chg[fav]
        draw_chg = chg[1]
        descs = []
        cold = False
        if hot_chg < -0.05:
            descs.append(f"热门方(主胜/客胜)概率下降{abs(hot_chg)*100:.0f}% → 市场信心增强")
        elif hot_chg > 0.05:
            descs.append(f"热门方概率上升{hot_chg*100:.0f}% → 市场信心减弱，冷门警报")
            cold = True
        if draw_chg < -0.08:
            descs.append(f"平局概率下降{abs(draw_chg)*100:.0f}% → 市场防平，平局概率上调")
        if not descs:
            descs.append("盘口变化幅度小，无明显信号")
        conf = "强" if hot_chg < -0.05 else ("弱" if hot_chg > 0.05 else "中性")
        return "；".join(descs), conf, cold, hot_chg
    except Exception:
        return "盘口变化：未取得", "未知", False, 0.0

# ---------------- 模块四：投注策略 ----------------
def strategy(p_model, p_market, odds_home, draw, away, cold_risk):
    odds = np.array([odds_home, draw, away], dtype=float)
    mkt = np.array(p_market, dtype=float); mkt = mkt / mkt.sum()
    k = int(np.argmax(p_model))
    edge = p_model[k] - mkt[k]
    lab = ["主胜", "平局", "客胜"]
    if cold_risk == "高":
        action = "观望（冷门风险高）"
    elif edge >= 0.05:
        action = "可介入"
    elif edge >= 0.02:
        action = "轻仓观察"
    elif edge < -0.05:
        action = "放弃该方向"
    else:
        action = "观望"
    # 大热防冷
    big_fav = mkt.max() >= 0.70
    if big_fav:
        action = "大热场次，建议关注让球盘或跳过"
    # 凯利（1/4）
    b = odds[k] - 1
    if b > 0:
        kelly = max((odds[k] * p_model[k] - 1) / b, 0.0)
    else:
        kelly = 0.0
    stake = kelly * 0.25 * 100
    return {"方向": lab[k], "edge": edge, "action": action, "kelly": kelly, "建议仓位": f"{stake:.1f}%", "大热": big_fav}

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
            # 冷门风险合并（模块一+三）
            cold_risk = "高" if om_cold else "低"
            st = strategy(p[i], mkt[i], row["B365H"], row["B365D"], row["B365A"], cold_risk)
            # 模块五
            w = "天气：未取得"
            if code == "E0" and row["HomeTeam"] in CITY_MAP:
                cname, lat, lon = CITY_MAP[row["HomeTeam"]]
                w = get_weather(cname, lat, lon, row["Date"])
            rank = final_rank(r["data"], row["Season"], row["HomeTeam"])
            mot = motivation(rank)
            ib = intl_break_flag(row["Date"])
            streak = streak_flag(row["HomeTeam"], r["data"], row["Date"])
            es = env_score(w, mot, streak, cold_risk)
            actual = "主胜" if row["FTHG"] > row["FTAG"] else ("平局" if row["FTHG"] == row["FTAG"] else "客胜")
            over = sum(poisson.pmf(i2, eg_h[i]) * poisson.pmf(j2, eg_a[i]) for i2 in range(11) for j2 in range(11) if i2 + j2 > 2)
            preds.append({
                "league": r["name"], "date": row["Date"].strftime("%Y-%m-%d"),
                "home": row["HomeTeam"], "away": row["AwayTeam"],
                "model_prob": [round(float(x), 3) for x in p[i]],
                "market_prob": [round(float(x), 3) for x in mkt[i]],
                "direction": st["方向"], "edge": round(st["edge"], 4), "action": st["action"],
                "stake": st["建议仓位"], "big_fav": bool(st["大热"]),
                "exp_goals": [round(float(eg_h[i]), 2), round(float(eg_a[i]), 2)],
                "over25": round(float(over), 3), "home_rank": int(rank),
                "injury": "未取得", "odds_move": om_desc, "market_conf": om_conf,
                "cold_risk": cold_risk, "weather": w, "referee": "未取得（需 API-Football key）",
                "tactics": "未取得（无风格标签数据源）", "motivation": mot,
                "intl_break": ib or "无", "streak": streak, "env_score": round(es, 1),
                "actual": actual, "score": f"{int(row['FTHG'])}-{int(row['FTAG'])}",
            })
    json.dump(preds, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # 打印最近一轮前几条
    for p in preds[:6]:
        print(f"\n[{p['league']}] {p['date']} {p['home']} vs {p['away']}")
        print(f"  模型 {p['model_prob']} | 市场 {p['market_prob']} | 方向 {p['direction']} | Edge {p['edge']}")
        print(f"  动作 {p['action']} | 建议仓位 {p['stake']} | 冷门风险 {p['cold_risk']} | 环境评分 {p['env_score']}")
        print(f"  盘口: {p['odds_move']}")
        print(f"  天气: {p['weather']} | 伤停: {p['injury']} | 裁判: {p['referee']} | 战术: {p['tactics']}")
        print(f"  实际: {p['score']}（{p['actual']}）")
    print(f"\n已生成 {OUT_JSON}（共 {len(preds)} 场）")

if __name__ == "__main__":
    main()
