import logging
import uuid
from tqdm import tqdm
from collections import defaultdict
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any
from tinydb import TinyDB, Query

from toolchemy.utils.logger import get_logger
from toolchemy.utils.utils import hash_dict, pp
from toolchemy.utils.cacher import Cacher, ICacher


class NotFoundError(Exception):
    pass


class FilterOp(Enum):
    GREATER = 1
    GREATER_OR_EQUAL = 2
    LESS = 3
    LESS_OR_EQUAL = 4
    EQUAL = 5


@dataclass
class Filter:
    key: str
    value: Any
    op: FilterOp = FilterOp.EQUAL

    def __str__(self) -> str:
        return f"Filter(key={self.key}, op={self.op.name}, value={self.value})"


class ILightDB(ABC):
    ID_FIELD = "id_"
    HASH_FIELD = "hash_"

    @abstractmethod
    def insert(self, doc: dict) -> str:
        pass

    @abstractmethod
    def insert_batch(self, docs: list[dict]) -> list[str]:
        pass

    @abstractmethod
    def update(self, doc: dict) -> str:
        pass

    @abstractmethod
    def upsert(self, doc: dict) -> str:
        pass

    @abstractmethod
    def retrieve(self, doc_id: str) -> dict | None:
        pass

    @abstractmethod
    def search(self, query_filter: Filter) -> list[dict]:
        pass

    @abstractmethod
    def all(self) -> list[dict]:
        pass

    @abstractmethod
    def remove(self, ids: list[str]) -> int:
        pass


class LightTinyDB(ILightDB):
    def __init__(self, db_file_path: str, indexes: list[str] | None = None, log_level: int = logging.INFO, cacher: ICacher | None = None):
        self._logger = get_logger(level=log_level)
        self._db = TinyDB(db_file_path)

        self._cacher = cacher or Cacher()

        self._indexes = {}

        for index_name in (indexes or []):
            self._create_and_rebuild_index(index_name)

    def all(self) -> list[dict]:
        return [dict(document) for document in self._db.all()]

    def insert(self, doc: dict) -> str:
        doc_prepared = self._prepare_doc_for_store(doc)
        cache_key = self._cacher.create_cache_key("exists", [doc])
        if not self._cacher.exists(cache_key):
            self._logger.debug(f"Inserting doc: {doc_prepared}")
            self._db.insert(doc_prepared)
            self._handle_index(doc_prepared)
            self._cacher.set(cache_key, doc_prepared[self.ID_FIELD])
        return doc_prepared[self.ID_FIELD]

    def insert_batch(self, docs: list[dict]) -> list[str]:
        docs_prepared = [self._prepare_doc_for_store(doc) for doc in docs]
        docs_prepared = [doc for doc in docs_prepared if not self._cacher.exists(self._cacher.create_cache_key("exists", [doc]))]
        self._logger.debug(f"Inserting doc batch: {docs_prepared}")
        self._db.insert_multiple(docs_prepared)
        for doc_prepared in docs_prepared:
            self._handle_index(doc_prepared)
        for doc in docs_prepared:
            self._cacher.set(self._cacher.create_cache_key("exists", [doc]), doc[self.ID_FIELD])
        return [doc[self.ID_FIELD] for doc in docs_prepared]

    def update(self, doc: dict) -> str:
        doc = self._prepare_doc_for_store(doc)
        q = Query()
        self._db.update(doc, q.id_ == doc[self.ID_FIELD])
        self._handle_index(doc)
        return doc[self.ID_FIELD]

    def upsert(self, doc: dict) -> str:
        doc = self._prepare_doc_for_store(doc)
        q = Query()
        self._db.upsert(doc, q.id_ == doc[self.ID_FIELD])
        self._handle_index(doc)
        return doc[self.ID_FIELD]

    def retrieve(self, doc_id: str) -> dict | None:
        docs = self.search(Filter(self.ID_FIELD, doc_id))
        if len(docs) < 1:
            return None
        return docs[0]

    def search(self, query_filter: Filter) -> list[dict]:
        self._logger.debug(f"Searching with filter: {str(query_filter)}")
        if query_filter.op == FilterOp.EQUAL and query_filter.value is not None:
            if self._has_index(query_filter.key):
                return self._search_index(query_filter.key, query_filter.value)
        tinydb_query = Query()
        documents = self._db.search(tinydb_query[query_filter.key].test(self._filter_to_test_fn(query_filter)))
        self._logger.debug(f"> documents found: {len(documents)}")
        return [self._prepare_doc_for_return(dict(document)) for document in documents]

    def _create_and_rebuild_index(self, field_name: str) -> None:
        self._logger.debug(f"Creating index for: '{field_name}'...")
        self._indexes[field_name] = defaultdict(list)
        docs = self._db.all()
        for doc in tqdm(docs, desc=f"recreating index '{field_name}':'"):
            self._add_to_index(field_name, doc)
        self._logger.debug(f"> indexed {len(docs)} documents")

    def _add_to_index(self, field_name: str, doc: dict) -> None:
        # self._logger.debug(f"Adding to index '{field_name}': {doc.get(field_name, 'MISSING')}")
        if field_name not in doc:
            return
        self._indexes[field_name][doc[field_name]].append(doc)

    def _remove_from_index(self, field_name: str, doc: dict) -> None:
        if field_name not in doc:
            return
        for indexed_doc in (self._indexes[field_name][doc[field_name]] or []):
            if indexed_doc[self.ID_FIELD] == doc[self.ID_FIELD]:
                self._indexes[field_name][doc[field_name]].remove(indexed_doc)

    def _handle_index(self, doc: dict, remove: bool = False) -> None:
        if remove:
            self._handle_index_remove(doc)
        else:
            self._handle_index_add(doc)

    def _handle_index_add(self, doc: dict) -> None:
        for field_name in doc.keys():
            if field_name in self._indexes:
                self._add_to_index(field_name, doc)

    def _handle_index_remove(self, doc: dict):
        self._logger.debug(f"_handler_index_remove| doc: {doc}")
        if len(doc.keys()) == 1 and list(doc.keys())[0] == self.ID_FIELD:
            self._logger.debug(f"> the doc has a single key, trying to get the full document")
            doc = self.retrieve(doc[self.ID_FIELD])
            if doc is None:
                return
            self._logger.debug(f"> the full document: {doc}")
        for doc_field in doc.keys():
            if doc_field in self._indexes:
                self._remove_from_index(doc_field, doc)

    def _has_index(self, key: str) -> bool:
        if key not in self._indexes:
            self._logger.debug(f"> there is no index: '{key}'")
            return False
        return True

    def _search_index(self, key: str, value: Any) -> list[dict]:
        if value not in self._indexes[key]:
            return []
        return self._indexes[key][value]

    def _filter_to_test_fn(self, query_filter: Filter):
        def test_func(val):
            if not isinstance(val, str) and not isinstance(val, int) and not isinstance(val, float):
                return False
            if query_filter.op == FilterOp.GREATER:
                return val > query_filter.value
            if query_filter.op == FilterOp.GREATER_OR_EQUAL:
                return val >= query_filter.value
            if query_filter.op == FilterOp.LESS:
                return val < query_filter.value
            if query_filter.op == FilterOp.LESS_OR_EQUAL:
                return val <= query_filter.value
            if query_filter.op == FilterOp.EQUAL:
                return val == query_filter.value
            raise ValueError(f"Unknown operator {query_filter.op}")
        return test_func

    def remove(self, ids: list[str] | str) -> int:
        if isinstance(ids, str):
            ids = [ids]
        removed_elements_ids = []
        for doc_id in ids:
            self._handle_index({self.ID_FIELD: doc_id}, remove=True)
            q = Query()
            removed_elements = self._db.remove(q.id_ == doc_id)
            removed_elements_ids.extend(removed_elements)
        return len(list(set(removed_elements_ids)))

    def _prepare_doc_for_store(self, doc: dict) -> dict:
        self._ensure_created_at(doc)
        return self._ensure_hash(self._ensure_id(doc))

    def _prepare_doc_for_return(self, doc: dict) -> dict:
        doc_copy = doc.copy()
        self._ensure_created_at(doc_copy)
        return doc_copy

    def _ensure_created_at(self, doc: dict):
        if "created_at" not in doc:
            err_msg = f"There is no created_at in doc!\n{pp(doc, print_msg=False)}"
            self._logger.error(err_msg)
            raise ValueError(err_msg)

    def _ensure_id(self, doc: dict) -> dict:
        if self.ID_FIELD not in doc:
            doc[self.ID_FIELD] = self._generate_id()
        return doc

    def _generate_id(self) -> str:
        return str(uuid.uuid4())

    def _ensure_hash(self, doc: dict) -> dict:
        if self.HASH_FIELD in doc and doc[self.HASH_FIELD]:
            self._logger.debug(f"the doc already has the hash property, keeping it as is: {doc[self.HASH_FIELD]}")
            return doc
        doc[self.HASH_FIELD] = self._generate_hash(doc)
        return doc

    def _generate_hash(self, doc: dict) -> str:
        doc_copy = doc.copy()
        if self.ID_FIELD in doc_copy:
            del doc_copy[self.ID_FIELD]
        if self.HASH_FIELD in doc_copy:
            del doc_copy[self.HASH_FIELD]
        return hash_dict(doc_copy)
