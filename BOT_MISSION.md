# 🤖 AI Forex Bot v2 — Mission Document

## 🎯 Why We Built This

**Satu tujuan:** ***$100 Streaming Challenge.***

Kita punya akun demo $8,566. Kita ingin membuktikan bahwa bot AI bisa trading forex secara **konsisten profit** dengan **risk management disiplin** — lalu di-streaming secara **jujur** (no fake profits, no hidden DD).

Goal akhir: **Bot yang bisa dijalankan dengan modal $100 real dan menghasilkan profit konsisten.**

---

## 🧠 Core Philosophy

1. **Honest Live Trading** — Semua keputusan terekam, semua loss diakui, tidak ada simulasi di LIVE mode.
2. **Data-Driven** — Setiap perubahan fitur/model wajib divalidasi dengan data (walk-forward, OOS, backtest).
3. **Risk First** — Profit kedua. Safety override, position sizing, drawdown limit adalah prioritas.
4. **Ensemble Over Single Model** — H4 trend + H1 entry + M5 timing, bukan satu model ajaib.
5. **Self-Learning** — Bot belajar dari trade outcomes dan recency-weighted retraining.

---

## 🏛️ Architecture Pillars

| Pillar | Apa | File Kunci |
|--------|-----|-----------|
| **MT5 Connector** | Live data, order execution | `data/mt5_connector.py` |
| **Market Data Engine** | Caching, routing, storage | `data/market_data_engine.py` |
| **Decision Engine** | Old single-TF logic | `decision/decision_engine.py` |
| **Ensemble v2** | **NEW** H4+H1+M5 weighted voting | `ensemble/` |
| **ML Models** | XGBoost, RF, LGBM, LSTM per TF | `ml/`, `models/` |
| **Risk Manager** | Position sizing, DD limits | `risk/risk_manager.py` |
| **Exit Engine** | ATR trailing, time-based exit | `trading/exit_engine.py` |
| **Self-Learning** | Simulation → trade memory → retrain | `simulation/`, `learning/` |
| **Dashboard** | Real-time status web UI | `web_dashboard/` |

---

## 🔑 Key Decisions (Jangan Diubah Tanpa Diskusi)

1. **M5 = entry timeframe.** Higher TFs (M15/M30/H1/H4) hanya context features.
2. **Trend override ada TAPI digate ML** — STRONG_BULLISH→BUY hanya jika ML sell_prob < 0.50.
3. **Safety override:** RSI>70 → no BUY, RSI<30 → no SELL. Threshold dinamis berdasarkan trend (BULLISH→85, BEARISH→15).
4. **Scale-out aktif** — 2 posisi: TP1 di ATR×1.0, RUN di ATR×2.5 + trailing.
5. **Ensemble v2 menggantikan decision engine lama** saat `ENSEMBLE_MODE=true`.
6. **Weight voting ensemble:** H4 50%, H1 35%, M5 15%.
7. **Recency weighting:** 0-30d=1.0, 31-90d=0.8, 91-180d=0.6, >365d=0.2.
8. **Chronological split only** — no random shuffle.
9. **SQLite untuk trade memory** — atomic writes, query performance, zero dependency.

---

## ⚠️ Golden Rules (Wajib Diingat)

| # | Rule |
|---|------|
| 1 | **Jangan coding sebelum diskusi** jika menyentuh trading logic, decision engine, AI model, risk management, exit logic, position sizing, training pipeline, self-learning, atau MT5 integration. |
| 2 | **Risk Specialist punya HAK VETO.** Kalau Risk bilang REJECT, STOP. |
| 3 | **Semua perubahan trading wajib bukti data.** Walk-forward, OOS, backtest. |
| 4 | **Semua perubahan model wajib validasi OOS.** |
| 5 | **Semua perubahan live trading wajib safety check.** |
| 6 | **Data > Opini.** Kalau bukti tidak cukup → NEED MORE DATA. |
| 7 | **Stabilitas > Profit sementara.** |

---

## 📊 Current Status (per 16 June 2026)

- **Mode:** LIVE | **Balance:** $8,566.13 | **DD:** -9.1%
- **Ensemble v2:** ✅ Running (H4 47.6%, H1 54.2%, M5 60.6%)
- **Old M5 models:** Masih jalan sebagai fallback, 5 TF ensembles
- **Bug terbaru fixed:** `rr_ratio` undefined (risk_manager.py), `no_trade` KeyError (main.py+ensemble)
- **Yang perlu dimonitor:** Ensemble performance di berbagai market regime, dashboard update untuk ensemble stats

---

## 🚀 Next Big Goal

**$100 Streaming Challenge:**
1. Verifikasi ensemble stabil 1-2 minggu di demo
2. Test dengan akun $100 real (modal kecil, risk kecil)
3. Stream hasilnya — jujur, transparan, edukatif
4. Scale up kalau konsisten profit
