import copy
import datetime
import inspect
import json
import numpy as np
import random
import torch
import hashlib
import base64

from dataclasses import asdict, is_dataclass
from typing import Any

from toolchemy.utils.datestimes import datetime_to_str


DEFAULT_SEED = 1337

MEGABYTE = 1024 ** 2
GIGABYTE = 1024 ** 3


def seed_init_fn(x, only_deterministic: bool = False):
    seed = DEFAULT_SEED + x
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if only_deterministic:
        torch.use_deterministic_algorithms(True)


def bytes_to_str(byte_data: bytes) -> str:
    for encoding in ['utf-8', 'latin-1', 'ascii']:
        try:
            return byte_data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Unknown encoding")


def pp_cast(msg: Any, skip_fields: list | None = None) -> Any:
    msg_copy = copy.deepcopy(msg)
    if is_dataclass(msg_copy):
        msg_copy = to_json(msg_copy)
    if isinstance(msg_copy, bytes):
        msg_copy = bytes_to_str(msg_copy)
    if isinstance(msg_copy, dict):
        for key in list(msg_copy):
            if skip_fields and key in skip_fields:
                del msg_copy[key]
                continue
            msg_copy[key] = pp_cast(msg_copy[key], skip_fields=skip_fields)
    if isinstance(msg_copy, list):
        for i, el in enumerate(msg_copy):
            msg_copy[i] = pp_cast(el, skip_fields=skip_fields)
    if isinstance(msg_copy, np.ndarray):
        msg_copy = pp_cast(msg_copy.tolist(), skip_fields=skip_fields)
    if isinstance(msg_copy, float):
        msg_copy = ff(msg)
    if isinstance(msg_copy, datetime.datetime):
        msg_copy = datetime_to_str(msg_copy)
    if isinstance(msg_copy, object) and type(msg_copy).__module__ != "builtins":
        msg_copy = json.loads(json.dumps(msg_copy, default=vars))
    return msg_copy


def pp(msg: str | dict | float | int | list, skip_fields: list | None = None, print_msg: bool = True) -> str:
    msg_ = pp_cast(msg, skip_fields=skip_fields)
    if isinstance(msg_, dict):
        msg_ = json.dumps(msg_, indent=4, ensure_ascii=False)
    if isinstance(msg_, list):
        if len(msg_) > 0 and isinstance(msg_[0], dict):
            msg_ = json.dumps(msg_, indent=4, ensure_ascii=False)
    if isinstance(msg_, int) or isinstance(msg_, float):
        msg_ = ff(msg_)
    if print_msg:
        print(msg_)
    return msg_



def ff(fval: float | list[float] | int | list[int] | dict | str | np.float32, precision: int = 2):
    if isinstance(fval, float):
        return "%0.*f" % (precision, fval)
    if isinstance(fval, int):
        return str(fval)
    if isinstance(fval, list):
        return [ff(v, precision=precision) for v in fval]
    if isinstance(fval, dict):
        return {k: ff(v, precision=precision) for k, v in fval.items()}
    if isinstance(fval, str):
        return ff(float(fval), precision)
    if isinstance(fval, np.float32):
        return ff(fval.item(), precision)
    raise ValueError(f"Unsupported type: {type(fval)}")


def to_json(dataclass_container, key_prefix: str = None, exclude: list[str] | None = None) -> dict:
    if exclude is None:
        exclude = []
    data_dict = asdict(dataclass_container)

    parsed_dict = {}
    for k, v in data_dict.items():
        if k in exclude:
            continue
        new_key = k
        if key_prefix is not None:
            new_key = f"{key_prefix}_{k}"
        if v is None:
            v = ""
        if isinstance(v, bool):
            v = int(v)
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], str):
            v = ', '.join(v)
        if is_dataclass(v):
            v = to_json(v, key_prefix=key_prefix, exclude=exclude)
        parsed_dict[new_key] = v

    return parsed_dict


def hash_dict(input_dict: dict) -> str:
    json_str = json.dumps(input_dict, sort_keys=True)
    json_bytes = json_str.encode('utf-8')
    hash_bytes = hashlib.sha256(json_bytes).digest()
    base64_hash = base64.b64encode(hash_bytes).decode('utf-8')

    return base64_hash


def normalize_path_str(path_: str) -> str:
    return path_.\
        replace("./", "_").\
        replace("~/", ""). \
        replace("~", ""). \
        replace("/", "_").\
        replace("-", "").\
        replace(":", "_").\
        replace("?", "_").\
        replace("&", "_")


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    num_chunks = (len(text) - chunk_overlap) // (chunk_size - chunk_overlap)

    chunks = []
    for i in range(num_chunks):
        start = i * (chunk_size - chunk_overlap)
        end = start + chunk_size
        chunks.append(text[start:end])

    chunks.append(text[num_chunks * (chunk_size - chunk_overlap):])

    return chunks


def truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return f"{s[:limit]} (...{str(len(s) - limit)} more chars)"


def _caller_module_name(offset: int = 2) -> str:
    frame = inspect.stack()[offset]
    module = inspect.getmodule(frame.frame)
    namespace = module.__name__ if module and hasattr(module, '__name__') else '__main__'
    return namespace
