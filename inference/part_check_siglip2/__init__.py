"""SigLIP 2 제로샷 부위 일치 검증 모듈."""

from .part_check import PartCheckError, check_part, load_embedder

__all__ = ["PartCheckError", "check_part", "load_embedder"]
