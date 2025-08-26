from enum import Enum


class ApplyTo(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    BOTH = "both"
