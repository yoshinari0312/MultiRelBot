from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import List, Optional, Dict, Any
import os
from datetime import datetime

try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    # .env ファイルを読み込む（プロジェクトルートから）
    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(dotenv_path)
except ImportError:
    # dotenv がインストールされていない場合はスキップ
    pass

# タイムスタンプ付きディレクトリを作成
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_ROOT = os.path.join("static", "logs", timestamp_str)
os.makedirs(LOG_ROOT, exist_ok=True)


# ============ サブ設定 ============


@dataclass
class LLMCfg:
    provider: Optional[str] = None  # "azure", "openai", "ollama"
    azure_endpoint: Optional[str] = None
    azure_api_key: Optional[str] = None
    azure_api_version: Optional[str] = None
    azure_model: Optional[str] = None
    azure_embedding_deployment: Optional[str] = None
    azure_embedding_api_version: Optional[str] = None
    reasoning_effort: Optional[str] = None  # "minimal", "low", "medium", "high"
    enable_reasoning_param: Optional[bool] = None
    # タスク別モデル
    human_model: Optional[str] = None
    relation_model: Optional[str] = None
    robot_model: Optional[str] = None
    topic_model: Optional[str] = None
    intervention_model: Optional[str] = None
    # 温度パラメータ
    relation_temperature: Optional[float] = None
    # リトライ設定
    max_attempts: Optional[int] = 5
    base_backoff: Optional[float] = 0.5


@dataclass
class EnvCfg:
    """環境・シミュレーション設定"""

    personas: Optional[Dict[str, Any]] = (
        None  # Dict[str, Dict[str, Any]]（triggers含む）
    )
    debug: Optional[bool] = None
    max_history_human: Optional[int] = None  # 人間LLM用の最大会話履歴数
    max_history_relation: Optional[int] = None  # 関係性推定用の最大会話履歴数
    intervention_max_history: Optional[int] = None  # 介入判定用の最大会話履歴数


@dataclass
class PepperCfg:
    ip: Optional[str] = None
    port: Optional[int] = None
    use_robot: Optional[bool] = None
    robot_included: Optional[bool] = None


@dataclass
class InterventionCfg:
    mode: Optional[str] = None
    isolation_threshold: Optional[float] = None
    temperature: Optional[float] = None


@dataclass
class ParticipantsCfg:
    num_participants: Optional[int] = None
    speakers: Optional[Dict[str, Any]] = None


@dataclass
class RealtimeCfg:
    silence_duration: Optional[float] = None
    use_google_stt: Optional[Any] = None
    use_direct_stream: Optional[bool] = None
    diarization_threshold: Optional[int] = None
    skip_threshold_bytes: Optional[int] = None
    n_batch: Optional[int] = None
    utterances_per_session: Optional[int] = None
    analyze_every: Optional[int] = None
    robot_count_after_intervention: Optional[int] = None


@dataclass
class TopicManagerCfg:
    enable: Optional[bool] = None
    generation_prompt: Optional[str] = None


@dataclass
class ScorerCfg:
    """スコアラー設定"""

    backend: Optional[str] = None
    use_ema: Optional[bool] = None
    decay_factor: Optional[float] = None


@dataclass
class SimulationCfg:
    """シミュレーション専用設定"""

    max_human_utterances: Optional[int] = None
    stability_check_interval: Optional[int] = None
    consecutive_stable_threshold: Optional[int] = None
    num_episodes: Optional[int] = None


@dataclass
class AppConfig:
    """アプリケーション全体の設定"""

    env: EnvCfg = field(default_factory=EnvCfg)
    llm: LLMCfg = field(default_factory=LLMCfg)
    simulation: SimulationCfg = field(default_factory=SimulationCfg)
    pepper: PepperCfg = field(default_factory=PepperCfg)
    intervention: InterventionCfg = field(default_factory=InterventionCfg)
    participants: ParticipantsCfg = field(default_factory=ParticipantsCfg)
    realtime: RealtimeCfg = field(default_factory=RealtimeCfg)
    topic_manager: TopicManagerCfg = field(default_factory=TopicManagerCfg)
    scorer: ScorerCfg = field(default_factory=ScorerCfg)


# ============ ヘルパー関数 ============


def _filter_kwargs(cls, kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """dataclass の有効なフィールドだけをフィルタリング"""
    if kwargs is None:
        return {}
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in kwargs.items() if k in valid}


def _normalize(val: Any) -> Optional[str]:
    """環境変数などの文字列を正規化（空文字列→None）"""
    if isinstance(val, str):
        stripped = val.strip()
        return stripped if stripped else None
    return val


def _validate_required_fields(cfg: AppConfig) -> None:
    """必須フィールドのバリデーション"""
    missing_sections = []

    def _require(section_name: str, obj: Any, field_list: List[str]) -> None:
        for fname in field_list:
            val = getattr(obj, fname, None)
            if val is None or (isinstance(val, str) and not val.strip()):
                missing_sections.append(f"{section_name}.{fname}")

    # LLM設定のバリデーション
    _require("llm", cfg.llm, ["provider"])
    if (cfg.llm.provider or "").lower() == "azure":
        _require(
            "llm.azure",
            cfg.llm,
            ["azure_endpoint", "azure_api_key", "azure_api_version", "azure_model"],
        )

    if missing_sections:
        joined = "\n - " + "\n - ".join(missing_sections)
        raise ValueError("Missing required configuration values:" + joined)


def load_config(yaml_path: str = "config.local.yaml") -> AppConfig:
    """config.local.yaml を（あれば）読み、既定値＋環境変数でマージ"""
    if not yaml:
        raise ImportError("PyYAML is not installed. Please run: pip install pyyaml")

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}

    # env 設定
    env_dict = dict(y.get("env") or {})
    env = EnvCfg(**_filter_kwargs(EnvCfg, env_dict))

    # LLM 設定（環境変数で上書き）
    llm_raw = dict(y.get("llm") or {})
    llm_raw.setdefault("azure_endpoint", os.getenv("AZURE_ENDPOINT"))
    llm_raw.setdefault("azure_api_key", os.getenv("AZURE_API_KEY"))
    llm_raw.setdefault("azure_api_version", os.getenv("AZURE_API_VERSION"))
    llm_raw.setdefault("azure_model", os.getenv("AZURE_MODEL"))
    llm_raw.setdefault(
        "azure_embedding_deployment", os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
    )
    llm_raw.setdefault(
        "azure_embedding_api_version", os.getenv("AZURE_EMBEDDING_API_VERSION")
    )
    llm_raw.setdefault("reasoning_effort", os.getenv("REASONING_EFFORT"))
    if os.getenv("ENABLE_REASONING_PARAM"):
        llm_raw["enable_reasoning_param"] = os.getenv(
            "ENABLE_REASONING_PARAM"
        ).lower() in ("true", "1", "yes")
    llm_raw.setdefault("human_model", os.getenv("HUMAN_MODEL"))
    llm_raw.setdefault("relation_model", os.getenv("RELATION_MODEL"))
    llm_raw.setdefault("robot_model", os.getenv("ROBOT_MODEL"))
    llm_raw.setdefault("topic_model", os.getenv("TOPIC_MODEL"))
    llm_raw.setdefault("intervention_model", os.getenv("INTERVENTION_MODEL"))
    llm = LLMCfg(**_filter_kwargs(LLMCfg, llm_raw))

    # Simulation 設定
    simulation = SimulationCfg(**_filter_kwargs(SimulationCfg, y.get("simulation")))
    # Pepper 設定
    pepper = PepperCfg(**_filter_kwargs(PepperCfg, y.get("pepper")))
    # 介入設定
    intervention = InterventionCfg(
        **_filter_kwargs(InterventionCfg, y.get("intervention"))
    )
    # 参加者設定
    participants = ParticipantsCfg(
        **_filter_kwargs(ParticipantsCfg, y.get("participants"))
    )
    # リアルタイム（音声）設定
    realtime = RealtimeCfg(**_filter_kwargs(RealtimeCfg, y.get("realtime")))

    topic_manager = TopicManagerCfg(
        **_filter_kwargs(TopicManagerCfg, y.get("topic_manager"))
    )

    scorer = ScorerCfg(**_filter_kwargs(ScorerCfg, y.get("scorer")))

    app_cfg = AppConfig(
        env=env,
        llm=llm,
        simulation=simulation,
        pepper=pepper,
        intervention=intervention,
        participants=participants,
        realtime=realtime,
        topic_manager=topic_manager,
        scorer=scorer,
    )
    _validate_required_fields(app_cfg)
    return app_cfg


def get_config() -> AppConfig:
    """設定をロードして返す（キャッシュなし）"""
    return load_config()
