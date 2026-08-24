# -*- coding: utf-8 -*-
"""竞彩每日网页生成器（云端版，纯标准库，无第三方依赖）
数据源：中国竞彩网官方接口 webapi.sporttery.cn
用法：python jingcai_fetch.py   → 生成 ./site/index.html
"""
import html
import json
import ssl
from datetime import datetime, timedelta
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

SHANGHAI = ZoneInfo("Asia/Shanghai")
BASE_URL = "https://webapi.sporttery.cn/gateway/jc/football/getMatchCalculatorV1.qry"
POOL_CODES = "hhad,had,crs,ttg,hafu"

def fetch_payload():
    query = urlencode({"poolCode": POOL_CODES, "channel": "c"})
    url = f"{BASE_URL}?{query}"
    req = Request(url, headers={
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Referer": "https://m.sporttery.cn/mjc/jsq/zqspf/",
        "Origin": "https://m.sporttery.cn",
    })
    with urlopen(req, timeout=25, context=ssl.create_default_context()) as resp:
        body = resp.read()
    payload = json.loads(body)
    if payload.get("errorCode") not in (None, "0", 0):
        raise RuntimeError(f"官方接口错误：{payload.get('errorMessage')}")
    return payload

def parse(payload):
    value = payload.get("value") or {}
    groups = value.get("matchInfoList") or []
    out = []
    seen = set()
    for g in groups:
        for item in (g.get("subMatchList") or []):
            if not isinstance(item, dict):
                continue
            num = item.get("matchNumStr") or item.get("matchNum")
            home = item.get("homeTeamAllName") or item.get("homeTeamAbbName") or item.get("homeTeamName")
            away = item.get("awayTeamAllName") or item.get("awayTeamAbbName") or item.get("awayTeamName")
            league = item.get("leagueAbbName") or item.get("leagueName") or "未知联赛"
            status = str(item.get("matchStatus") or item.get("sellStatus") or "Selling").lower()
            if status not in ("selling", "1", "true"):
                continue
            if not num or not home or not away or num in seen:
                continue
            seen.add(num)
            md = item.get("matchDate"); mt = item.get("matchTime")
            ko = None
            if md and mt:
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
                    try:
                        ko = datetime.strptime(f"{md} {mt}", fmt).replace(tzinfo=SHANGHAI); break
                    except ValueError:
                        continue
            if ko is None:
                continue
            had = item.get("had") or {}
            hhad = item.get("hhad") or {}
            def od(block):
                try:
                    return float(block["h"]), float(block["d"]), float(block["a"])
                except (KeyError, TypeError, ValueError):
                    return None
            o1 = od(had)
            oh = od(hhad)
            goal_line = hhad.get("goalLine")
            try:
                goal_line = float(goal_line)
            except (TypeError, ValueError):
                goal_line = None
            out.append({"num": num, "league": str(league), "home": str(home), "away": str(away),
                        "ko": ko, "o1": o1, "oh": oh, "gl": goal_line})
    return out

def imp_probs(h, d, a):
    inv = [1.0/h, 1.0/d, 1.0/a]
    s = sum(inv)
    return [x/s for x in inv]

LABELS = ["主胜", "平局", "客胜"]

def opinion(it):
    o1 = it["o1"]; oh = it["oh"]
    if o1 is None:
        return "暂无官方赔率，无法解读"
    ph, pd_, pa = imp_probs(*o1)
    probs = [ph, pd_, pa]
    k = max(range(3), key=lambda i: probs[i])
    pick = LABELS[k]
    parts = [f"官方胜平负 {o1[0]:.2f}/{o1[1]:.2f}/{o1[2]:.2f} → 主胜{ph:.0%}、平{pd_:.0%}、客胜{pa:.0%}。"]
    if probs[k] >= 0.50:
        conf = "信心较高"
    elif probs[k] >= 0.40:
        conf = "信心一般"
    else:
        conf = "偏谨慎"
    parts.append(f"市场更看好{pick}（{probs[k]:.0%}），{conf}。")
    if pd_ >= 0.30:
        parts.append(f"平局概率{pd_:.0%}不低，有防平价值。")
    if oh is not None and it["gl"] is not None:
        hv = it["gl"]
        hh, hd_, ha = oh
        who = "主队让" if hv < 0 else ("客队让" if hv > 0 else "平手")
        parts.append(f"官方让球：{who}{abs(hv):g}球（{hh:.2f}/{hd_:.2f}/{ha:.2f}）。")
        if hv < 0:
            m = min([(hh, "让球方大胜"), (hd_, "刚好卡盘"), (ha, "让球方难覆盖")], key=lambda t: t[0])
        else:
            m = min([(hh, "让球方难覆盖"), (hd_, "刚好卡盘"), (ha, "让球方大胜")], key=lambda t: t[0])
        parts.append(f"盘口最低赔率在「{m[1]}」档，市场倾向{m[1]}。")
    parts.append("注：仅基于官方赔率解读，未接入球队攻防/伤停数据，仅供参考。")
    return " ".join(parts)

def main():
    now = datetime.now(SHANGHAI)
    items = parse(fetch_payload())
    items.sort(key=lambda x: x["ko"])
    groups = {}
    for it in items:
        d = it["ko"].strftime("%Y-%m-%d")
        it["passed"] = it["ko"] < now
        it["day"] = d
        groups.setdefault(d, []).append(it)

    def row(it):
        ko = it["ko"].strftime("%m-%d %H:%M")
        o1 = it["o1"]
        odds = f"{o1[0]:.2f} / {o1[1]:.2f} / {o1[2]:.2f}" if o1 else "—"
        oh = it["oh"]
        hd = f"{it['gl']:g}球　{oh[0]:.2f}/{oh[1]:.2f}/{oh[2]:.2f}" if oh and it["gl"] is not None else "—"
        badge = "<span class='tag r'>已开赛</span>" if it["passed"] else "<span class='tag g'>未开赛</span>"
        esc = html.escape
        return (f"<tr><td><b>{esc(it['num'])}</b></td><td>{esc(it['league'])}</td>"
                f"<td>{esc(it['home'])} <span class='vs'>vs</span> {esc(it['away'])}</td>"
                f"<td>{ko}</td><td>{odds}</td><td>{hd}</td><td>{badge}</td>"
                f"<td style='font-size:12.5px;color:#374151;max-width:360px'>{esc(opinion(it))}</td></tr>")

    sections = ""
    for d, its in sorted(groups.items()):
        rows = "".join(row(it) for it in its)
        n_up = sum(1 for it in its if not it["passed"])
        weekday = ["周一","周二","周三","周四","周五","周六","周日"][datetime.strptime(d, "%Y-%m-%d").weekday()]
        sections += (f"<div class='card'><h2>{weekday}（{d}）· {len(its)} 场 · 未开赛 {n_up}</h2>"
                     f"<table class='tbl'><thead><tr><th>场次</th><th>联赛</th><th>对阵</th>"
                     f"<th>开赛(北京)</th><th>官方胜平负</th><th>官方让球</th><th>状态</th><th>大白话解读</th></tr></thead>"
                     f"<tbody>{rows}</tbody></table></div>")

    total_up = sum(1 for it in items if not it["passed"])
    page = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>今日竞彩 · 每日自动更新</title><style>
body{{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;background:#f6f7f9;color:#111827;margin:0}}
.wrap{{max-width:1200px;margin:0 auto;padding:24px 16px 60px}}
header{{background:linear-gradient(135deg,#b91c1c,#dc2626);color:#fff;padding:30px 24px;border-radius:14px;margin-bottom:20px}}
h1{{margin:0 0 6px;font-size:25px}} header p{{margin:2px 0;opacity:.94;font-size:13.5px}}
.card{{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:18px 20px;margin-bottom:18px}}
h2{{font-size:17px;margin:0 0 10px;color:#b91c1c;border-left:4px solid #dc2626;padding-left:10px}}
.tbl{{width:100%;border-collapse:collapse;font-size:13px}} .tbl th,.tbl td{{border:1px solid #e5e7eb;padding:7px 8px;text-align:left}}
.tbl th{{background:#fef2f2;color:#b91c1c}} .tbl tr:nth-child(even) td{{background:#fafbfc}}
.vs{{color:#9ca3af;font-size:12px}} .tag{{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600}}
.tag.g{{background:#dcfce7;color:#15803d}} .tag.r{{background:#fee2e2;color:#b91c1c}}
footer{{text-align:center;color:#9ca3af;font-size:12px;margin-top:26px}}
</style></head><body><div class="wrap">
<header><h1>🎯 今日竞彩（中国大陆竞彩网开售场次）</h1>
<p>数据源：中国竞彩网官方接口 webapi.sporttery.cn · 共 {len(items)} 场（未开赛 {total_up} 场）· 更新于 {now.strftime('%Y-%m-%d %H:%M')}（北京时间）</p>
<p>每天自动更新 · 所有内容仅为官方赔率解读与概率分析，不构成任何投注建议</p></header>
{sections}
<footer>数据来自中国竞彩网官方接口 · 每天固定时间云端自动更新 · 仅概率与风险分析，不构成投注建议</footer>
</div></body></html>"""
    import os
    os.makedirs("site", exist_ok=True)
    with open("site/index.html", "w", encoding="utf-8") as f:
        f.write(page)
    print(f"OK 已生成 site/index.html（{len(items)} 场，未开赛 {total_up}）")

if __name__ == "__main__":
    main()
