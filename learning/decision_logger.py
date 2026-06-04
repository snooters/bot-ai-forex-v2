import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from core.constants import DECISION_LOG_DIR, REASONING_LOG_DIR
from utils.logger import get_logger


class DecisionLogger:
    def __init__(self):
        self.logger = get_logger("decision_logger")
        self._log_dir = Path(DECISION_LOG_DIR)
        self._reasoning_dir = Path(REASONING_LOG_DIR)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._reasoning_dir.mkdir(parents=True, exist_ok=True)

    def log_decision(self, symbol: str, decision: Dict):
        timestamp = datetime.now().isoformat()
        filename = f"decision_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        context = decision.get("context", {})
        record = {
            "timestamp": timestamp,
            "symbol": symbol,
            "decision": {
                "action": decision.get("action", "HOLD"),
                "confidence": decision.get("confidence", 0),
                "market_score": decision.get("market_score", 0),
                "no_trade": decision.get("no_trade", True),
                "no_trade_reasons": decision.get("no_trade_reasons", []),
                "reasons": decision.get("reasons", []),
            },
            "context": {
                "trend": context.get("trend"),
                "regime": context.get("regime"),
                "volatility": context.get("volatility"),
                "momentum": context.get("momentum"),
                "timeframe": context.get("timeframe"),
                "price": context.get("price"),
                "spread": context.get("spread"),
                "feature_summary": context.get("feature_summary", {}),
            },
            "ml_signal": decision.get("ml_signal", {}),
            "account_status": decision.get("account_status", {}),
        }

        filepath = self._log_dir / filename
        try:
            with open(filepath, "w") as f:
                json.dump(record, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save decision log: {e}")

    def log_reasoning(self, symbol: str, reasoning_data: Dict):
        timestamp = datetime.now().isoformat()
        filename = f"reasoning_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        record = {
            "timestamp": timestamp,
            "symbol": symbol,
            "reasoning": reasoning_data,
        }
        filepath = self._reasoning_dir / filename
        try:
            with open(filepath, "w") as f:
                json.dump(record, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save reasoning log: {e}")

    def get_recent_decisions(self, symbol: str, n: int = 10) -> List[Dict]:
        pattern = f"decision_{symbol}_"
        files = sorted(self._log_dir.glob(f"{pattern}*.json"), reverse=True)
        result = []
        for f in files[:n]:
            try:
                with open(f) as fh:
                    result.append(json.load(fh))
            except Exception:
                pass
        return result
