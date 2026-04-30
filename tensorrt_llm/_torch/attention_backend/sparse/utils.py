from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.pyexecutor.resource_manager import (KVCacheManager,
                                                             KVCacheManagerV2)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import CpType

from .dsa import DSACacheManager, DSACacheManagerV2, DSATrtllmAttention
from .rocket import (RocketKVCacheManager, RocketTrtllmAttention,
                     RocketVanillaAttention)


def _dsa_v2_compatible(kv_cache_config) -> bool:
    """Return True iff the requested kv_cache_config is compatible with
    using KVCacheManagerV2 under DSA.

    Mirrors the V2 fallback guards in :py:class:`KvCacheCreator` and
    :py:class:`KVCacheManagerV2.__init__` so DSA-on-V2 never violates an
    invariant the V2 runtime asserts.
    """
    if kv_cache_config is None:
        return False
    if getattr(kv_cache_config, "event_buffer_max_size", 0) > 0:
        return False
    return True


def get_sparse_attn_kv_cache_manager(
        sparse_attn_config: "SparseAttentionConfig",
        kv_cache_config=None,
        mapping=None,
        max_beam_width: int = 1,
        kv_connector_manager=None):
    """Pick the cache manager class for a sparse-attention algorithm.

    For DSA, honor ``kv_cache_config.use_kv_cache_manager_v2`` when the
    requested config is compatible with V2; otherwise warn and fall back to
    V1 (``DSACacheManager``). Other sparse algorithms remain V1-only for now.
    """
    use_v2 = bool(
        kv_cache_config is not None
        and getattr(kv_cache_config, "use_kv_cache_manager_v2", False))
    # V2 doesn't yet support beam search, kv connectors, or STAR CP — share
    # the same guards used by KVCacheManagerV2.__init__ so we bail early.
    cp_is_star = (mapping is not None
                  and mapping.cp_config.get('cp_type') == CpType.STAR)
    if use_v2 and (kv_connector_manager is not None or max_beam_width > 1
                   or cp_is_star or not _dsa_v2_compatible(kv_cache_config)):
        use_v2 = False

    if sparse_attn_config.algorithm == "rocket":
        return RocketKVCacheManager
    elif sparse_attn_config.algorithm == "dsa":
        if use_v2:
            return DSACacheManagerV2
        if kv_cache_config is not None and getattr(
                kv_cache_config, "use_kv_cache_manager_v2", False):
            logger.warning(
                "DSA: use_kv_cache_manager_v2 was requested but the current "
                "config is not compatible (beam_width>1, kv_connector, "
                "event_buffer, or STAR CP). Falling back to DSACacheManager (V1)."
            )
        return DSACacheManager
    elif sparse_attn_config.algorithm == "skip_softmax":
        return KVCacheManagerV2 if use_v2 else KVCacheManager
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm: {sparse_attn_config.algorithm}"
        )


def get_vanilla_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config.algorithm == "rocket":
        return RocketVanillaAttention
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm in vanilla attention backend: {sparse_attn_config.algorithm}"
        )


def get_trtllm_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config.algorithm == "rocket":
        return RocketTrtllmAttention
    elif sparse_attn_config.algorithm == "dsa":
        return DSATrtllmAttention
    elif sparse_attn_config.algorithm == "skip_softmax":
        return TrtllmAttention
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm in trtllm attention backend: {sparse_attn_config.algorithm}"
        )


def get_flashinfer_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    raise ValueError(
        f"Unsupported sparse attention algorithm in flashinfer attention backend: {sparse_attn_config.algorithm}"
    )
