import functools
import os
import threading
import unittest
from uuid import uuid4

from python.btree.b_tree_on_disk import PageIndex, Page


class PageIndexTest(unittest.TestCase):
    def setUp(self):
        self.filename = "page_index_test"
        try:
            # prevent stale file from previous test
            os.remove(self.filename)
        except FileNotFoundError:
            pass

    def tearDown(self):
        os.remove(self.filename)

    def test_persistence(self):
        page_index = PageIndex(file_name=self.filename)

        # append some data
        test_dict = {i: i for i in range(10_000)}
        for page_id, offset in test_dict.items():
            page_index.append(page_id, offset)

        page_index.close()

        # reload the index
        page_index = PageIndex(file_name=self.filename)
        self.assertDictEqual(page_index._index, test_dict)

        test_dict = {i: i for i in range(10_000)}
        for page_id, offset in test_dict.items():
            page_index.remove(page_id)

        page_index.close()

        # reload the index
        page_index = PageIndex(file_name=self.filename)
        self.assertDictEqual(page_index._index, {})

    def test_threadsafe(self):
        def worker(page_index, input_array):
            for i in input_array:
                page_index.append(i, i)

        inputs = [
            [i for i in range(1000)],
            [i for i in range(1000, 2000)],
            [i for i in range(2000, 3000)],
        ]

        page_index = PageIndex(file_name=self.filename)

        threads = [
            threading.Thread(target=functools.partial(worker, page_index, inputs[i]))
            for i in range(3)
        ]
        for t in threads:
            t.start()

        for t in threads:
            t.join()

        page_index.close()

        # reload the index
        page_index = PageIndex(file_name=self.filename)
        self.assertDictEqual(
            page_index._index,
            {i: i for i in inputs[0] + inputs[1] + inputs[2]},
        )

        # random delete test


class PageTest(unittest.TestCase):
    def setUp(self):
        self.filename = "page_test"
        try:
            # prevent stale file from previous test
            os.remove(self.filename)
        except FileNotFoundError:
            pass

    def tearDown(self):
        ...
        # os.remove(self.filename)

    def test_persistence(self):
        file = open(self.filename, "w+b")

        page = Page.create(file, page_id=uuid4().int, is_leaf=True)

    def test_threadsafe(self): ...
