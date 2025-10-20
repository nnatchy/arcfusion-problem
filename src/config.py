"""
Configuration management for ArcFusion RAG System.
Loads settings from config.ini with environment variable overrides.
"""

import configparser
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
from dotenv import load_dotenv

# Load .env file
load_dotenv()


@dataclass
class SystemConfig:
    """System-wide configuration"""
    version: str
    pdf_directory: str
    checkpoint_db_name: str
    default_greeting: str


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    results_file: str
    history_file: str
    test_cases_file: str


@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: str
    model: str
    temperature: float


@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    model: str
    dimension: int
    batch_size: int


@dataclass
class PineconeConfig:
    """Pinecone vector database configuration"""
    index_name: str
    dimension: int
    metric: str
    cloud: str
    region: str
    namespace: str
    initialization_wait_seconds: int


@dataclass
class RAGConfig:
    """RAG pipeline configuration"""
    chunk_size: int
    chunk_overlap: int
    top_k_retrieval: int
    top_k_final: int
    use_reranking: bool
    reranker_model: str
    reranker_top_n: int
    recursive_retrieval: bool
    max_recursion_depth: int
    citation_extraction_enabled: bool
    follow_citations: bool
    max_citations_per_doc: int
    nested_retrieval_top_k: int
    upsert_batch_size: int
    chunk_sentence_window: int


@dataclass
class WebSearchConfig:
    """Web search configuration"""
    provider: str
    max_results: int
    search_depth: str
    include_raw_content: bool
    include_answer: bool
    excerpt_length: int


@dataclass
class SessionConfig:
    """Session management configuration"""
    storage_type: str
    checkpoint_dir: str
    backend: str
    postgres_connection_string: Optional[str] = None


@dataclass
class AgentConfig:
    """Agent behavior configuration"""
    max_iterations: int
    confidence_threshold: float
    reflection_quality_threshold: float
    orchestrator_quality_threshold: float
    enable_security_check: bool
    enable_reflection: bool
    # Agent-specific temperatures
    intent_router_temperature: float
    clarification_temperature: float
    router_temperature: float
    planner_temperature: float
    recursive_rag_temperature: float
    synthesis_temperature: float
    reflection_temperature: float
    web_search_temperature: float
    # History windows
    intent_router_history_window: int
    clarification_history_window: int
    router_history_window: int
    planner_history_window: int
    synthesis_history_window: int
    rag_history_window: int
    # Source limits
    synthesis_max_sources: int
    rag_max_sources: int
    reflection_max_sources: int
    source_preview_length: int


@dataclass
class APIConfig:
    """API server configuration"""
    host: str
    port: int
    cors_origins: list


@dataclass
class StreamlitConfig:
    """Streamlit UI configuration"""
    enabled: bool
    port: int
    title: str
    theme: str
    default_session_id: str
    example_query_1: str
    example_query_2: str
    example_query_3: str
    example_query_4: str


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str
    format: str
    colorize: bool
    rotation: str
    retention: str
    log_file: str


@dataclass
class MemoryConfig:
    """Memory and history management configuration"""
    max_history_tokens: int
    keep_recent_tokens: int
    summary_target_tokens: int
    char_to_token_ratio: int
    summary_temperature: float


@dataclass
class AgentPrompts:
    """All agent system and user prompts"""
    intent_router_system: str
    intent_router_user: str
    clarification_system: str
    clarification_user: str
    router_system: str
    router_user: str
    planner_system: str
    planner_user: str
    recursive_rag_system: str
    recursive_rag_user: str
    web_search_system: str
    web_search_user: str
    synthesis_system: str
    synthesis_user: str
    reflection_system: str
    reflection_user: str
    evaluation_system: str
    evaluation_user: str
    history_summarizer_system: str
    history_summarizer_user: str


class Config:
    """
    Central configuration management.
    Loads from config.ini and allows environment variable overrides.
    """

    def __init__(self, config_path: str = "config.ini"):
        self.config = configparser.ConfigParser()

        # Find config file
        if not Path(config_path).exists():
            # Try in parent directory
            config_path = Path(__file__).parent.parent / "config.ini"

        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.config.read(config_path)

        # Load environment variables (override config.ini)
        self._load_env_overrides()

        # Parse configurations
        self.system = self._parse_system_config()
        self.llm = self._parse_llm_config()
        self.embeddings = self._parse_embedding_config()
        self.pinecone = self._parse_pinecone_config()
        self.rag = self._parse_rag_config()
        self.web_search = self._parse_web_search_config()
        self.session = self._parse_session_config()
        self.memory = self._parse_memory_config()
        self.evaluation = self._parse_evaluation_config()
        self.agents = self._parse_agent_config()
        self.api = self._parse_api_config()
        self.streamlit = self._parse_streamlit_config()
        self.logging = self._parse_logging_config()
        self.prompts = self._parse_prompts()

        # API Keys from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")

        # Validate required keys
        self._validate_api_keys()

    def _load_env_overrides(self):
        """Override config values with environment variables (format: SECTION__KEY)"""
        for section in self.config.sections():
            for key in self.config[section]:
                env_key = f"{section.upper()}__{key.upper()}"
                env_value = os.getenv(env_key)
                if env_value:
                    self.config[section][key] = env_value

    def _parse_system_config(self) -> SystemConfig:
        section = self.config["system"]
        return SystemConfig(
            version=section.get("version"),
            pdf_directory=section.get("pdf_directory"),
            checkpoint_db_name=section.get("checkpoint_db_name"),
            default_greeting=section.get("default_greeting")
        )

    def _parse_llm_config(self) -> LLMConfig:
        section = self.config["llm"]
        return LLMConfig(
            provider=section.get("provider"),
            model=section.get("model"),
            temperature=section.getfloat("temperature")
        )

    def _parse_embedding_config(self) -> EmbeddingConfig:
        section = self.config["embeddings"]
        return EmbeddingConfig(
            model=section.get("model"),
            dimension=section.getint("dimension"),
            batch_size=section.getint("batch_size")
        )

    def _parse_pinecone_config(self) -> PineconeConfig:
        section = self.config["pinecone"]
        return PineconeConfig(
            index_name=section.get("index_name"),
            dimension=section.getint("dimension"),
            metric=section.get("metric"),
            cloud=section.get("cloud"),
            region=section.get("region"),
            namespace=section.get("namespace"),
            initialization_wait_seconds=section.getint("initialization_wait_seconds")
        )

    def _parse_rag_config(self) -> RAGConfig:
        section = self.config["rag"]
        return RAGConfig(
            chunk_size=section.getint("chunk_size"),
            chunk_overlap=section.getint("chunk_overlap"),
            top_k_retrieval=section.getint("top_k_retrieval"),
            top_k_final=section.getint("top_k_final"),
            use_reranking=section.getboolean("use_reranking"),
            reranker_model=section.get("reranker_model"),
            reranker_top_n=section.getint("reranker_top_n"),
            recursive_retrieval=section.getboolean("recursive_retrieval"),
            max_recursion_depth=section.getint("max_recursion_depth"),
            citation_extraction_enabled=section.getboolean("citation_extraction_enabled"),
            follow_citations=section.getboolean("follow_citations"),
            max_citations_per_doc=section.getint("max_citations_per_doc"),
            nested_retrieval_top_k=section.getint("nested_retrieval_top_k"),
            upsert_batch_size=section.getint("upsert_batch_size"),
            chunk_sentence_window=section.getint("chunk_sentence_window")
        )

    def _parse_web_search_config(self) -> WebSearchConfig:
        section = self.config["web_search"]
        return WebSearchConfig(
            provider=section.get("provider"),
            max_results=section.getint("max_results"),
            search_depth=section.get("search_depth"),
            include_raw_content=section.getboolean("include_raw_content"),
            include_answer=section.getboolean("include_answer"),
            excerpt_length=section.getint("excerpt_length")
        )

    def _parse_session_config(self) -> SessionConfig:
        section = self.config["session"]
        return SessionConfig(
            storage_type=section.get("storage_type"),
            checkpoint_dir=section.get("checkpoint_dir"),
            backend=section.get("backend"),
            postgres_connection_string=os.getenv("POSTGRES_CONNECTION_STRING")
        )

    def _parse_agent_config(self) -> AgentConfig:
        section = self.config["agents"]
        return AgentConfig(
            max_iterations=section.getint("max_iterations"),
            confidence_threshold=section.getfloat("confidence_threshold"),
            reflection_quality_threshold=section.getfloat("reflection_quality_threshold"),
            orchestrator_quality_threshold=section.getfloat("orchestrator_quality_threshold"),
            enable_security_check=section.getboolean("enable_security_check"),
            enable_reflection=section.getboolean("enable_reflection"),
            # Agent-specific temperatures
            intent_router_temperature=section.getfloat("intent_router_temperature"),
            clarification_temperature=section.getfloat("clarification_temperature"),
            router_temperature=section.getfloat("router_temperature"),
            planner_temperature=section.getfloat("planner_temperature"),
            recursive_rag_temperature=section.getfloat("recursive_rag_temperature"),
            synthesis_temperature=section.getfloat("synthesis_temperature"),
            reflection_temperature=section.getfloat("reflection_temperature"),
            web_search_temperature=section.getfloat("web_search_temperature"),
            # History windows
            intent_router_history_window=section.getint("intent_router_history_window"),
            clarification_history_window=section.getint("clarification_history_window"),
            router_history_window=section.getint("router_history_window"),
            planner_history_window=section.getint("planner_history_window"),
            synthesis_history_window=section.getint("synthesis_history_window"),
            rag_history_window=section.getint("rag_history_window"),
            # Source limits
            synthesis_max_sources=section.getint("synthesis_max_sources"),
            rag_max_sources=section.getint("rag_max_sources"),
            reflection_max_sources=section.getint("reflection_max_sources"),
            source_preview_length=section.getint("source_preview_length")
        )

    def _parse_api_config(self) -> APIConfig:
        section = self.config["api"]
        cors_origins = json.loads(section.get("cors_origins"))
        return APIConfig(
            host=section.get("host"),
            port=section.getint("port"),
            cors_origins=cors_origins
        )

    def _parse_streamlit_config(self) -> StreamlitConfig:
        section = self.config["streamlit"]
        return StreamlitConfig(
            enabled=section.getboolean("enabled"),
            port=section.getint("port"),
            title=section.get("title"),
            theme=section.get("theme"),
            default_session_id=section.get("default_session_id"),
            example_query_1=section.get("example_query_1"),
            example_query_2=section.get("example_query_2"),
            example_query_3=section.get("example_query_3"),
            example_query_4=section.get("example_query_4")
        )

    def _parse_logging_config(self) -> LoggingConfig:
        section = self.config["logging"]
        return LoggingConfig(
            level=section.get("level"),
            format=section.get("format"),
            colorize=section.getboolean("colorize"),
            rotation=section.get("rotation"),
            retention=section.get("retention"),
            log_file=section.get("log_file")
        )

    def _parse_memory_config(self) -> MemoryConfig:
        section = self.config["memory"]
        return MemoryConfig(
            max_history_tokens=section.getint("max_history_tokens"),
            keep_recent_tokens=section.getint("keep_recent_tokens"),
            summary_target_tokens=section.getint("summary_target_tokens"),
            char_to_token_ratio=section.getint("char_to_token_ratio"),
            summary_temperature=section.getfloat("summary_temperature")
        )

    def _parse_evaluation_config(self) -> EvaluationConfig:
        section = self.config["evaluation"]
        return EvaluationConfig(
            results_file=section.get("results_file"),
            history_file=section.get("history_file"),
            test_cases_file=section.get("test_cases_file")
        )

    def _parse_prompts(self) -> AgentPrompts:
        """Parse all agent prompts from config"""
        return AgentPrompts(
            intent_router_system=self.config.get("prompts.intent_router", "system"),
            intent_router_user=self.config.get("prompts.intent_router", "user"),
            clarification_system=self.config.get("prompts.clarification", "system"),
            clarification_user=self.config.get("prompts.clarification", "user"),
            router_system=self.config.get("prompts.router", "system"),
            router_user=self.config.get("prompts.router", "user"),
            planner_system=self.config.get("prompts.planner", "system"),
            planner_user=self.config.get("prompts.planner", "user"),
            recursive_rag_system=self.config.get("prompts.recursive_rag", "system"),
            recursive_rag_user=self.config.get("prompts.recursive_rag", "user"),
            web_search_system=self.config.get("prompts.web_search", "system"),
            web_search_user=self.config.get("prompts.web_search", "user"),
            synthesis_system=self.config.get("prompts.synthesis", "system"),
            synthesis_user=self.config.get("prompts.synthesis", "user"),
            reflection_system=self.config.get("prompts.reflection", "system"),
            reflection_user=self.config.get("prompts.reflection", "user"),
            evaluation_system=self.config.get("prompts.evaluation", "system"),
            evaluation_user=self.config.get("prompts.evaluation", "user"),
            history_summarizer_system=self.config.get("prompts.history_summarizer", "system"),
            history_summarizer_user=self.config.get("prompts.history_summarizer", "user"),
        )

    def _validate_api_keys(self):
        """Validate that required API keys are present"""
        missing = []

        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not self.pinecone_api_key:
            missing.append("PINECONE_API_KEY")
        if not self.tavily_api_key:
            missing.append("TAVILY_API_KEY")

        if missing:
            raise ValueError(
                f"Missing required API keys: {', '.join(missing)}. "
                f"Please set them in your .env file or environment variables."
            )


# Global config instance
config = Config()
