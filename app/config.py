from dataclasses import dataclass, field


@dataclass
class QueueConfig:
    video_max_size: int = 5
    sleep_time: float = 0.02
    process_delay: float = 0.01

@dataclass
class ModelConfig:
    video_weight: float = 0.5
    audio_weight: float = 0.5
    threshold: float = 0.5

@dataclass
class AppConfig:
    queue: QueueConfig = field(default_factory=QueueConfig())
    model: ModelConfig = field(default_factory=ModelConfig())
    camera_id: int = 0

