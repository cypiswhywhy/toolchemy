from pyfakefs.fake_filesystem_unittest import TestCase
from deepdiff import DeepDiff

from toolchemy.db.lightdb import LightTinyDB, Filter, FilterOp
from toolchemy.utils.datestimes import current_datetime_str
from toolchemy.utils.cacher import DummyCacher, ICacher


class LightTinyDBTest(TestCase):
    def setUp(self):
        self.setUpPyfakefs()
        db_dir = "/some/path"
        db_name = "db.json"
        self._db_path = f"{db_dir}/{db_name}"

        self.fs.create_dir(db_dir)

        self._db = LightTinyDB(db_file_path=self._db_path, cacher=self._create_cacher())

        self._doc = {
            "foo": "bar",
            "value": 123,
            "created_at": current_datetime_str(),
        }

        self._batch_docs = [
            {
                "foo": "bar1",
                "value": 100,
                "created_at": current_datetime_str(),
            },
            {
                "foo": "bar2",
                "value": 200,
                "created_at": current_datetime_str(),
            },
            {
                "foo": "bar3",
                "value": 300,
                "created_at": current_datetime_str(),
            }
        ]

    def _create_cacher(self) -> ICacher:
        return DummyCacher(with_memory_store=True)

    def test_insert_and_retrieve(self):
        doc_id = self._db.insert(self._doc)

        retrieved_doc = self._db.retrieve(doc_id)

        diff = DeepDiff(self._doc, retrieved_doc)
        assert diff == {}, diff

    def test_insert_batch_and_search_eq(self):
        self._db.insert_batch(self._batch_docs)

        retrieved_docs = self._db.search(Filter("foo", "bar2"))

        diff = DeepDiff([self._batch_docs[1]], retrieved_docs)
        assert diff == {}, diff

    def test_insert_batch_and_search_eq_with_index(self):
        self._db = LightTinyDB(db_file_path=self._db_path, indexes=["foo"], cacher=self._create_cacher())
        self._db.insert_batch(self._batch_docs)

        retrieved_docs = self._db._search_index("foo", "bar2")

        diff = DeepDiff([self._batch_docs[1]], retrieved_docs)
        assert diff == {}, diff

    def test_insert_batch_and_search_l(self):
        self._db.insert_batch(self._batch_docs)

        retrieved_docs = self._db.search(Filter("value", 200, FilterOp.LESS))

        diff = DeepDiff([self._batch_docs[0]], retrieved_docs)
        assert diff == {}, diff

    def test_insert_batch_and_search_le(self):
        self._db.insert_batch(self._batch_docs)

        retrieved_docs = self._db.search(Filter("value", 200, FilterOp.LESS_OR_EQUAL))

        diff = DeepDiff([self._batch_docs[0], self._batch_docs[1]], retrieved_docs)
        assert diff == {}, diff

    def test_insert_batch_and_search_g(self):
        self._db.insert_batch(self._batch_docs)

        retrieved_docs = self._db.search(Filter("value", 200, FilterOp.GREATER))

        diff = DeepDiff([self._batch_docs[2]], retrieved_docs)
        assert diff == {}, diff

    def test_insert_batch_and_search_ge(self):
        self._db.insert_batch(self._batch_docs)

        retrieved_docs = self._db.search(Filter("value", 200, FilterOp.GREATER_OR_EQUAL))

        diff = DeepDiff([self._batch_docs[1], self._batch_docs[2]], retrieved_docs)
        assert diff == {}, diff

    def test_insert_batch_and_search_ge_when_missing_field(self):
        self._db.insert_batch(self._batch_docs)

        retrieved_docs = self._db.search(Filter("non-existing-value", 200, FilterOp.GREATER_OR_EQUAL))

        assert len(retrieved_docs) == 0

    def test_insert_batch_and_search_ge_when_none_value(self):
        another_doc = {
            "foo": "bar1",
            "value": None,
            "created_at": current_datetime_str(),
        }
        self._db.insert_batch(self._batch_docs)
        self._db.insert(another_doc)

        retrieved_docs = self._db.search(Filter("value", 200, FilterOp.GREATER_OR_EQUAL))

        diff = DeepDiff([self._batch_docs[1], self._batch_docs[2]], retrieved_docs)
        assert diff == {}, diff

    def test_insert_batch(self):
        self._test_insert_batch(self._db)

    def test_insert_batch_with_index(self):
        self._db = LightTinyDB(db_file_path=self._db_path, indexes=["foo"], cacher=self._create_cacher())
        self._test_insert_batch(self._db)

        assert "foo" in self._db._indexes
        assert len(self._db._indexes["foo"]) == 3
        assert self._db._indexes["foo"]["bar1"] == [self._batch_docs[0]]
        assert self._db._indexes["foo"]["bar2"] == [self._batch_docs[1]]
        assert self._db._indexes["foo"]["bar3"] == [self._batch_docs[2]]

    def _test_insert_batch(self, db: LightTinyDB):
        db.insert_batch(self._batch_docs)

        retrieved_docs = db.all()

        diff = DeepDiff(self._batch_docs, retrieved_docs)
        assert diff == {}, diff

    def test_insert_with_index(self):
        self._db = LightTinyDB(db_file_path=self._db_path, indexes=["foo"], cacher=self._create_cacher())
        doc_id = self._db.insert(self._doc)

        retrieved_doc = self._db.retrieve(doc_id)

        diff = DeepDiff(self._doc, retrieved_doc)
        assert diff == {}, diff

        assert "foo" in self._db._indexes
        assert len(self._db._indexes["foo"]) == 1
        assert self._db._indexes["foo"]["bar"] == [self._doc]

    def test_update(self):
        self._test_update(self._db)

    def test_update_with_index(self):
        self._test_update(LightTinyDB(db_file_path=self._db_path, indexes=["foo"], cacher=self._create_cacher()))

    def _test_update(self, db: LightTinyDB):
        doc_id = db.insert(self._doc)
        doc = db.retrieve(doc_id)

        doc["value"] = 321

        db.update(doc)
        retrieved_doc = db.retrieve(doc_id)

        diff = DeepDiff(doc, retrieved_doc)
        assert diff == {}, diff

    def test_remove(self):
        doc_id = self._db.insert(self._doc)
        doc = self._db.retrieve(doc_id)

        assert doc[LightTinyDB.ID_FIELD] == doc_id, f"Expected {doc_id}, got {doc[LightTinyDB.ID_FIELD]}"

        removed_elements = self._db.remove([doc_id])
        assert removed_elements == 1

        doc = self._db.retrieve(doc_id)
        assert doc is None

    def test_remove_with_str_id(self):
        doc_id = self._db.insert(self._doc)
        doc = self._db.retrieve(doc_id)

        assert doc[LightTinyDB.ID_FIELD] == doc_id, f"Expected {doc_id}, got {doc[LightTinyDB.ID_FIELD]}"

        removed_elements = self._db.remove(doc_id)
        assert removed_elements == 1

        doc = self._db.retrieve(doc_id)
        assert doc is None

    def test_remove_with_index(self):
        self._db = LightTinyDB(db_file_path=self._db_path, indexes=["foo"], cacher=self._create_cacher())
        doc_id = self._db.insert(self._doc)
        doc = self._db.retrieve(doc_id)

        assert doc[LightTinyDB.ID_FIELD] == doc_id, f"Expected {doc_id}, got {doc[LightTinyDB.ID_FIELD]}"
        assert "foo" in self._db._indexes
        assert len(self._db._indexes["foo"]) == 1
        assert self._db._indexes["foo"]["bar"] == [self._doc]

        removed_elements = self._db.remove([doc_id])
        assert removed_elements == 1
        assert "foo" in self._db._indexes
        assert len(self._db._indexes["foo"]) == 1
        assert self._db._indexes["foo"]["bar"] == []

        doc = self._db.retrieve(doc_id)
        assert doc is None

    def test_create_and_rebuild_indexes(self):
        self._db = LightTinyDB(db_file_path=self._db_path, indexes=["foo"], cacher=self._create_cacher())
        self._db.insert_batch(self._batch_docs)

        self._db = LightTinyDB(db_file_path=self._db_path, indexes=["foo"], cacher=self._create_cacher())
        assert len(self._db._indexes) == 1
        assert "foo" in self._db._indexes
        assert len(self._db._indexes["foo"]) == 3


    def test_create_and_rebuild_indexes_with_missing_field(self):
        non_existing_field_name = "non_existing_in_docs_to_be_added"
        self._db = LightTinyDB(db_file_path=self._db_path, indexes=[non_existing_field_name], cacher=self._create_cacher())
        self._db.insert_batch(self._batch_docs)

        self._db = LightTinyDB(db_file_path=self._db_path, indexes=[non_existing_field_name], cacher=self._create_cacher())
        assert len(self._db._indexes) == 1
        assert non_existing_field_name in self._db._indexes
        assert len(self._db._indexes[non_existing_field_name]) == 0

    def test_ensure_hash(self):
        sample_doc1 = {
            LightTinyDB.ID_FIELD: 123,
            "content": "foo",
            "another_property": 1.2,
        }
        sample_doc1 = self._db._ensure_hash(sample_doc1)
        sample_doc2 = sample_doc1.copy()
        sample_doc2[LightTinyDB.ID_FIELD] = 124
        sample_doc2 = self._db._ensure_hash(sample_doc2)

        assert sample_doc1[LightTinyDB.HASH_FIELD] == sample_doc2[LightTinyDB.HASH_FIELD]
