import functools
import os
import threading
import unittest
from random import shuffle
from unittest import TestCase
from uuid import uuid4

from python.btree.b_tree_on_disk import PageManager, Page, BPlusTreeOnDisk


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
        page_index = PageManager(file_name=self.filename)

        # append some data
        test_dict = {i: (i, i) for i in range(10_000)}
        for page_id, offset in test_dict.items():
            page_index._append(page_id, *offset)

        page_index.close()

        # reload the index
        page_index = PageManager(file_name=self.filename)
        self.assertDictEqual(page_index._index, test_dict)

        test_dict = {i: i for i in range(10_000)}
        for page_id, offset in test_dict.items():
            page_index._remove(page_id)

        page_index.close()

        # reload the index
        page_index = PageManager(file_name=self.filename)
        self.assertDictEqual(page_index._index, {})

    def test_reserve_page(self):
        test_cases = 1_000
        page_index = PageManager(file_name=self.filename, page_size=128)
        # reserve 10 pages
        for i in range(test_cases):
            page_index._reserve_page()
        page_index.close()

        # reload the index
        page_index = PageManager(file_name=self.filename, page_size=128)
        self.assertEqual(len(page_index._index), test_cases)
        for i in range(test_cases):
            self.assertIn(i, page_index._index)

        intervals = list(page_index._index.values())
        intervals.sort(key=lambda x: x[0])

        for i in range(1, len(intervals)):
            # Check if the end_offset of the previous interval
            # is greater than the start_offset of the current interval
            if intervals[i - 1][1] > intervals[i][0]:
                self.fail(
                    f"Overlapping intervals {intervals[i - 1]} and {intervals[i]}"
                )

    def test_threadsafe(self):
        def worker(page_index, input_array):
            for i in input_array:
                page_index._append(i, i, i)

        inputs = [
            [i for i in range(1000)],
            [i for i in range(1000, 2000)],
            [i for i in range(2000, 3000)],
        ]

        page_index = PageManager(file_name=self.filename)

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
        page_index = PageManager(file_name=self.filename)
        self.assertDictEqual(
            page_index._index,
            {i: (i, i) for i in inputs[0] + inputs[1] + inputs[2]},
        )

        # random delete test


class PageTest(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.filename = "page_test"
        self.page_index_name = "page_test.index"
        try:
            # prevent stale file from previous test
            os.remove(self.filename)
            os.remove(self.page_index_name)
        except FileNotFoundError:
            pass

        self.page_index = PageManager(file_name=self.page_index_name)

    def tearDown(self):
        os.remove(self.filename)
        os.remove(self.page_index_name)
        self.page_index.close()

    def test_persistence(self):
        key_sizes = [4, 8, 16, 32]
        value_sizes = [4, 8, 16, 32]

        for key_size in key_sizes:
            for value_size in value_sizes:
                page_id = self.page_index._reserve_page()
                page = Page(
                    open(self.filename, "w+b"),
                    self.page_index,
                    page_id=page_id,
                    page_size=128,
                    parent_page_id=-1,
                    is_leaf=True,
                    key_size=key_size,
                    value_size=value_size,
                )

                test_data = [i for i in range(1, 101)]
                for i in test_data:
                    k = i.to_bytes(key_size, "big")
                    v = i.to_bytes(value_size, "big")
                    page.insert(k, v)

                page.close()

                # reload the page
                page = Page.from_page_id(
                    db_file=open(self.filename, "r+b"),
                    page_index=self.page_index,
                    page_id=page_id,
                )
                page.load()

                self.assertListEqual(
                    [key for key in test_data],
                    [int.from_bytes(k) for k in page.keys],
                )
                self.assertListEqual(
                    [value for value in test_data],
                    [int.from_bytes(k, "big") for k in page.values],
                )
                page.close()

    def test_threadsafe(self):
        def worker(page, input_array):
            for i in input_array:
                i = i.to_bytes(4, "big")
                page.insert(i, i)

        inputs = [
            [i for i in range(300)],
            [i for i in range(301, 600)],
            [i for i in range(601, 1000)],
        ]

        page_id = self.page_index._reserve_page()
        page = Page(
            open(self.filename, "w+b"),
            self.page_index,
            page_id=page_id,
            page_size=128,
            parent_page_id=-1,
            is_leaf=True,
            key_size=4,
            value_size=4,
        )

        threads = [
            threading.Thread(target=functools.partial(worker, page, inputs[i]))
            for i in range(3)
        ]
        for t in threads:
            t.start()

        for t in threads:
            t.join()

        page.close()

        # reload the page
        page = Page(
            open(self.filename, "r+b"),
            self.page_index,
            page_id=page_id,
            page_size=1024,
            parent_page_id=-1,
            is_leaf=True,
            key_size=4,
            value_size=4,
        )
        page.load()
        page.close()

        self.assertListEqual(
            [int.from_bytes(k, "big") for k in page.keys],
            sorted(inputs[0] + inputs[1] + inputs[2]),
        )
        self.assertListEqual(
            [int.from_bytes(k, "big") for k in page.values],
            sorted(inputs[0] + inputs[1] + inputs[2]),
        )

    def test_split(self):
        key_sizes = [4, 8, 16, 32]

        for key_size in key_sizes:
            page_id = self.page_index._reserve_page()
            page = Page(
                open(self.filename, "w+b"),
                self.page_index,
                page_id=page_id,
                page_size=1024,
                parent_page_id=-1,
                is_leaf=True,
                key_size=key_size,
                value_size=key_size,
            )
            # insert 100 elements
            test_data = [i.to_bytes(key_size, "big") for i in range(1, 101)]
            for k in test_data:
                page.insert(k, k)

            sibling_page_id, mid_key = page.split()
            page.close()

            # reload the page
            page = Page.from_page_id(
                db_file=open(self.filename, "r+b"),
                page_index=self.page_index,
                page_id=page_id,
            )
            page.load()
            page.close()
            sibling_page = Page.from_page_id(
                db_file=open(self.filename, "r+b"),
                page_index=self.page_index,
                page_id=sibling_page_id,
            )
            sibling_page.load()
            sibling_page.close()

            # after split
            # mid key = 51
            # page 1-51 (50 elements)
            expected_page = [i for i in range(1, 52)]
            # sibling page 52-100 (49 elements)
            expected_sibling_page = [i for i in range(52, 101)]

            # check the mid-key
            self.assertEqual(int.from_bytes(mid_key, "big"), 51)

            # check the keys and values
            self.assertEqual(len(page.keys), 51)
            self.assertEqual(len(sibling_page.keys), 49)

            self.assertListEqual(
                expected_page, [int.from_bytes(k, "big") for k in page.keys]
            )
            self.assertListEqual(
                expected_page, [int.from_bytes(k, "big") for k in page.values]
            )
            self.assertListEqual(
                expected_sibling_page,
                [int.from_bytes(k, "big") for k in sibling_page.keys],
            )
            self.assertListEqual(
                expected_sibling_page,
                [int.from_bytes(k, "big") for k in sibling_page.values],
            )


class BPlusTreeOnDiskTest(TestCase):
    def setUp(self):
        # make dir if not exist
        if not os.path.exists("./test_data"):
            os.mkdir("./test_data")

        self.filename = "./test_data/b_plus_tree_test" + str(uuid4())
        try:
            # prevent stale file from previous test
            os.remove(self.filename)
            os.remove(self.filename + ".index")
        except FileNotFoundError:
            pass

    def tearDown(self):
        os.remove(self.filename)
        os.remove(self.filename + ".index")

    def test_insert(self):
        btree = BPlusTreeOnDisk(
            file_name=self.filename,
            page_size=56,
            key_size=4,
            value_size=4,
        )
        # test_data = [i for i in range(20)]
        # shuffle(test_data)
        # fmt: off
        graph = []
        test_data = [12, 0, 14, 10, 19, 8, 2, 13, 17, 9, 3, 16, 11, 5, 18, 6, 4, 15, 1, 7]
        for i in test_data:
            k = i.to_bytes(4, "big")
            try:
                btree.insert(k, k)
                graph.append(btree.graph(i))
            except Exception as e:
                graph.append(btree.graph(i))
                print(e)
                break
                # self.fail(f"Failed to insert {i}\n error: {e}\n array: {test_data}, ")

        btree.merge_pdfs(graph)
        btree.close()
        exit(1)

        # reload the tree
        btree_reload = BPlusTreeOnDisk(
            file_name=self.filename,
            page_size=56,
            key_size=4,
            value_size=4,
        )
        assert btree._file.name == btree_reload._file.name

        for k in test_data:
            value = btree_reload.search(k.to_bytes(4, "big"))
            if value is None:
                btree_reload.graph(k)
            self.assertEqual(k, int.from_bytes(value, "big"))


# trigger duplicate parent key
# [12, 0, 14, 10, 19, 8, 2, 13, 17, 9, 3, 16, 11, 5, 18, 6, 4, 15, 1, 7]
