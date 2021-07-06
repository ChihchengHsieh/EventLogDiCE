from typing import List
from dataclasses import dataclass, field
from parameters.enum import (
    BPI2012ActivityType
)


@dataclass
class BPI2012Parameters(object):
    file_path: str = "./data/event_logs/BPI_Challenge_2012.xes"

    preprocessed_folder_path: str = "./data/preprocessed/BPI_Challenge_2012_with_resource"

    include_types: List[BPI2012ActivityType] = field(
        default_factory=lambda: [BPI2012ActivityType.A])

    include_complete_only: bool = True

    def __post_init__(self):
        self.include_types = [BPI2012ActivityType[t] if type(
            t) == str else t for t in self.include_types]


@dataclass
class BPI2012ScenarioParameters(object):
    file_path: str = "./data/event_logs/BPI_Challenge_2012.xes"

    preprocessed_folder_path: str = "./data/preprocessed/BPI_Challenge_2012_scenario"

    include_types: List[BPI2012ActivityType] = field(
        default_factory=lambda: [BPI2012ActivityType.A])

    include_complete_only: bool = True

    sample_times: int = 5

    def __post_init__(self):
        self.include_types = [BPI2012ActivityType[t] if type(
            t) == str else t for t in self.include_types]
