import os
import json
from pathlib import Path
from dotenv import load_dotenv

from core.exceptions import ConfigError

load_dotenv()


class Config:
    _instance = None
    _loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._loaded:
            return
        self._loaded = True
        self._config = {}
        self._load_from_env()
        self._validate()

    def _load_from_env(self):
        self._config = {
            "account": {
                "balance": self._get_float("ACCOUNT_BALANCE", 100.0),
                "currency": self._get_str("ACCOUNT_CURRENCY", "USD"),
                "leverage": self._get_int("ACCOUNT_LEVERAGE", 100),
                "trading_mode": self._get_str("TRADING_MODE", "simulation"),
                "is_demo": self._get_bool("IS_DEMO", True),
                "allow_real": self._get_bool("ALLOW_REAL_TRADING", False),
                "learn_only": self._get_bool("LEARN_ONLY", False),
                "user_id": self._get_str("USER_ID", ""),
                "password": self._get_str("PASSWORD", ""),
                "server": self._get_str("SERVER", ""),
                "server_ip": self._get_str("SERVER_IP", ""),
            },
            "trading": {
                "pairs": self._get_list("TRADING_PAIRS", ["EURUSD"]),
                "timeframes": self._get_list("TRADING_TIMEFRAMES", ["M5", "M15", "M30", "H1", "H4"]),
                "adaptive_timeframe": self._get_bool("ADAPTIVE_TIMEFRAME", True),
                "use_multi_timeframe": self._get_bool("USE_MULTI_TIMEFRAME", True),
            },
            "risk": {
                "max_risk_pct": self._get_float("MAX_RISK_PCT", 0.005),
                "max_daily_loss_pct": self._get_float("MAX_DAILY_LOSS_PCT", 0.03),
                "max_open_positions": self._get_int("MAX_OPEN_POSITIONS", 1),
                "rr_ratio": self._get_float("RR_RATIO", 1.0),
                "use_dynamic_sl": self._get_bool("USE_DYNAMIC_SL", True),
                "use_dynamic_tp": self._get_bool("USE_DYNAMIC_TP", True),
                "sl_pips": self._get_float("SL_PIPS", 30.0),
                "tp_pips": self._get_float("TP_PIPS", 60.0),
                "trailing_activate": self._get_float("TRAILING_ACTIVATE", 20.0),
                "trailing_distance": self._get_float("TRAILING_DISTANCE", 10.0),
                "trailing_atr_multiplier": self._get_float("TRAILING_ATR_MULTIPLIER", 1.5),
                "max_hold_hours": self._get_int("MAX_HOLD_HOURS", 12),
                "use_dynamic_risk": self._get_bool("USE_DYNAMIC_RISK", True),
            },
            "ai_filter": {
                "min_confidence": self._get_float("MIN_CONFIDENCE", 0.55),
                "min_market_score": self._get_int("MIN_MARKET_SCORE", 50),
                "allow_no_trade": self._get_bool("ALLOW_NO_TRADE", True),
                "max_spread_pips": self._get_float("MAX_SPREAD_PIPS", 2.0),
            },
            "learning": {
                "enabled": self._get_bool("ENABLE_SELF_LEARNING", True),
                "auto_retrain": self._get_bool("ENABLE_AUTO_RETRAIN", True),
                "retrain_after_trades": self._get_int("RETRAIN_AFTER_TRADES", 50),
                "retrain_interval_hours": self._get_int("RETRAIN_INTERVAL_HOURS", 24),
                "model_versioning": self._get_bool("MODEL_VERSIONING", True),
                "concept_drift": self._get_bool("CONCEPT_DRIFT_DETECTION", True),
                "adaptive_memory": self._get_bool("ADAPTIVE_MEMORY", True),
                "weight_new_data": self._get_bool("WEIGHT_NEW_DATA", True),
            },
            "ml": {
                "enable_xgboost": self._get_bool("ENABLE_XGBOOST", True),
                "enable_random_forest": self._get_bool("ENABLE_RANDOM_FOREST", True),
                "enable_lightgbm": self._get_bool("ENABLE_LIGHTGBM", True),
                "enable_lstm": self._get_bool("ENABLE_LSTM", True),
                "enable_voting_ensemble": self._get_bool("ENABLE_VOTING_ENSEMBLE", True),
            },
            "training": {
                "confirm_training": self._get_str("CONFIRM_TRAINING", "yes"),
                "historical_years": self._get_int("HISTORICAL_YEARS", 2),
                "rolling_window_days": self._get_int("ROLLING_TRAINING_WINDOW_DAYS", 180),
            },
            "analysis": {
                "market_structure": self._get_bool("ENABLE_MARKET_STRUCTURE", True),
                "support_resistance": self._get_bool("ENABLE_SUPPORT_RESISTANCE", True),
                "breakout_detection": self._get_bool("ENABLE_BREAKOUT_DETECTION", True),
                "breakdown_detection": self._get_bool("ENABLE_BREAKDOWN_DETECTION", True),
                "price_action": self._get_bool("ENABLE_PRICE_ACTION", True),
                "candle_pattern": self._get_bool("ENABLE_CANDLE_PATTERN", True),
                "volatility_analysis": self._get_bool("ENABLE_VOLATILITY_ANALYSIS", True),
                "trend_analysis": self._get_bool("ENABLE_TREND_ANALYSIS", True),
                "market_regime": self._get_bool("ENABLE_MARKET_REGIME", True),
            },
            "news": {
                "enabled": self._get_bool("NEWS_ENABLED", False),
                "weight": self._get_float("NEWS_WEIGHT", 0.15),
                "api_key": self._get_str("NEWS_API_KEY", ""),
            },
            "llm": {
                "enabled": self._get_bool("ENABLE_LLM", False),
                "weight": self._get_float("LLM_WEIGHT", 0.20),
                "api_key": self._get_str("LLM_API_KEY", ""),
                "api_url": self._get_str("LLM_API_URL", ""),
                "model": self._get_str("LLM_MODEL", ""),
            },
            "profit_target": {
                "percent": self._get_float("PROFIT_TARGET_PERCENT", 0.3),
            },
            "telegram": {
                "enabled": self._get_bool("TELEGRAM_ENABLED", False),
                "bot_token": self._get_str("TELEGRAM_BOT_TOKEN", ""),
                "chat_id": self._get_str("TELEGRAM_CHAT_ID", ""),
                "notify_open": self._get_bool("TELEGRAM_NOTIFY_OPEN", True),
                "notify_close": self._get_bool("TELEGRAM_NOTIFY_CLOSE", True),
                "notify_trailing": self._get_bool("TELEGRAM_NOTIFY_TRAILING", True),
                "notify_emergency": self._get_bool("TELEGRAM_NOTIFY_EMERGENCY", True),
                "notify_error": self._get_bool("TELEGRAM_NOTIFY_ERROR", True),
                "notify_retrain": self._get_bool("TELEGRAM_NOTIFY_RETRAIN", True),
                "notify_daily": self._get_bool("TELEGRAM_NOTIFY_DAILY", True),
                "notify_weekly": self._get_bool("TELEGRAM_NOTIFY_WEEKLY", True),
                "notify_monthly": self._get_bool("TELEGRAM_NOTIFY_MONTHLY", True),
                "notify_heartbeat": self._get_bool("TELEGRAM_NOTIFY_HEARTBEAT", True),
            },
            "emergency": {
                "caution_dd": self._get_float("EMERGENCY_CAUTION_DD", 0.03),
                "danger_dd": self._get_float("EMERGENCY_DANGER_DD", 0.04),
                "critical_dd": self._get_float("EMERGENCY_CRITICAL_DD", 0.05),
                "auto_close_positions": self._get_bool("EMERGENCY_AUTO_CLOSE_POSITIONS", True),
                "notify_telegram": self._get_bool("EMERGENCY_NOTIFY_TELEGRAM", True),
            },
            "backtest": {
                "walk_forward": self._get_bool("ENABLE_WALK_FORWARD_TEST", True),
                "monte_carlo": self._get_bool("ENABLE_MONTE_CARLO_TEST", True),
                "out_of_sample": self._get_bool("ENABLE_OUT_OF_SAMPLE_TEST", True),
            },
            "logging": {
                "level": self._get_str("LOG_LEVEL", "INFO"),
                "save_ai_reasoning": self._get_bool("SAVE_AI_REASONING", True),
                "save_trade_history": self._get_bool("SAVE_TRADE_HISTORY", True),
                "save_market_state": self._get_bool("SAVE_MARKET_STATE", True),
            },
            "diagnostic": {
                "show_confidence": self._get_bool("SHOW_AI_CONFIDENCE", True),
                "show_market_score": self._get_bool("SHOW_MARKET_SCORE", True),
                "show_regime": self._get_bool("SHOW_REGIME", True),
                "show_timeframe": self._get_bool("SHOW_SELECTED_TIMEFRAME", True),
                "show_model_version": self._get_bool("SHOW_MODEL_VERSION", True),
                "show_learning_status": self._get_bool("SHOW_LEARNING_STATUS", True),
            },
        }

    def _get_str(self, key: str, default: str) -> str:
        return os.getenv(key, default)

    def _get_int(self, key: str, default: int) -> int:
        try:
            return int(os.getenv(key, str(default)))
        except (ValueError, TypeError):
            return default

    def _get_float(self, key: str, default: float) -> float:
        try:
            return float(os.getenv(key, str(default)))
        except (ValueError, TypeError):
            return default

    def _get_bool(self, key: str, default: bool) -> bool:
        val = os.getenv(key)
        if val is None:
            return default
        return val.lower() in ("true", "1", "yes")

    def _get_list(self, key: str, default: list) -> list:
        val = os.getenv(key)
        if not val:
            return default
        return [x.strip() for x in val.split(",") if x.strip()]

    def _validate(self):
        acct = self._config["account"]
        if acct["trading_mode"] == "live" and not acct["allow_real"]:
            if not acct.get("learn_only"):
                raise ConfigError("Real trading not allowed. Set ALLOW_REAL_TRADING=true in .env or LEARN_ONLY=true")
        if not acct["is_demo"] and acct["trading_mode"] == "live":
            if not acct["server"] or not acct["user_id"]:
                raise ConfigError("Live mode requires SERVER and USER_ID")
        if self._config["risk"]["max_risk_pct"] > 0.02:
            raise ConfigError("MAX_RISK_PCT cannot exceed 2%")
        if self._config["risk"]["max_daily_loss_pct"] > 0.10:
            raise ConfigError("MAX_DAILY_LOSS_PCT cannot exceed 10%")
        pairs = self._config["trading"]["pairs"]
        for p in pairs:
            base = p.split(".")[0]
            if len(base) < 6 or not base.isalpha():
                raise ConfigError(f"Invalid trading pair: {p}")

    @property
    def account(self) -> dict:
        return self._config["account"]

    @property
    def trading(self) -> dict:
        return self._config["trading"]

    @property
    def risk(self) -> dict:
        return self._config["risk"]

    @property
    def ai_filter(self) -> dict:
        return self._config["ai_filter"]

    @property
    def learning(self) -> dict:
        return self._config["learning"]

    @property
    def ml(self) -> dict:
        return self._config["ml"]

    @property
    def training(self) -> dict:
        return self._config["training"]

    @property
    def analysis(self) -> dict:
        return self._config["analysis"]

    @property
    def news(self) -> dict:
        return self._config["news"]

    @property
    def llm(self) -> dict:
        return self._config["llm"]

    @property
    def profit_target(self) -> dict:
        return self._config["profit_target"]

    @property
    def telegram(self) -> dict:
        return self._config["telegram"]

    @property
    def emergency(self) -> dict:
        return self._config["emergency"]

    @property
    def backtest(self) -> dict:
        return self._config["backtest"]

    @property
    def logging(self) -> dict:
        return self._config["logging"]

    @property
    def diagnostic(self) -> dict:
        return self._config["diagnostic"]

    def get_dynamic_min_confidence(self, balance: float) -> float:
        if balance < 200:
            return 0.65
        elif balance < 500:
            return 0.55
        elif balance < 2000:
            return 0.50
        elif balance < 5000:
            return 0.45
        return 0.40

    def get_dynamic_max_positions(self, balance: float) -> int:
        if balance < 500:
            return 1
        elif balance < 2000:
            return 2
        return 3

    def to_dict(self) -> dict:
        return self._config

    def __getitem__(self, key):
        return self._config[key]


config = Config()
