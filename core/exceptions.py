class BotError(Exception):
    pass

class ConfigError(BotError):
    pass

class DataError(BotError):
    pass

class MT5ConnectionError(DataError):
    pass

class MT5DataError(DataError):
    pass

class CacheError(DataError):
    pass

class FeatureError(BotError):
    pass

class ModelError(BotError):
    pass

class ModelTrainingError(ModelError):
    pass

class ModelPredictionError(ModelError):
    pass

class ModelNotFoundError(ModelError):
    pass

class DecisionError(BotError):
    pass

class RiskError(BotError):
    pass

class RiskLimitExceeded(RiskError):
    pass

class EmergencyStop(RiskError):
    pass

class ExecutionError(BotError):
    pass

class OrderRejectedError(ExecutionError):
    pass

class LLMError(BotError):
    pass

class LLMTimeoutError(LLMError):
    pass

class TelegramError(BotError):
    pass

class LearningError(BotError):
    pass

class BacktestError(BotError):
    pass

class DashboardError(BotError):
    pass
