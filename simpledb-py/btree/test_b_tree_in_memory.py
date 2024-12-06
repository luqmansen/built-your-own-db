import random
from random import shuffle
from unittest import TestCase
from uuid import uuid4

from python.btree.b_tree_in_memory import BPlusTree


class BTreeTest(TestCase):
    def test_insert_and_delete(self):
        for order in range(5, 1000, 100):
            for input_range in [list(range(0, _max)) for _max in [100, 1_000, 10_000]]:
                with self.subTest(
                    msg=f"test orders {order} with input range {input_range[0]} to {input_range[-1]}"
                ):
                    btree = BPlusTree(order=order)
                    shuffle(input_range)

                    for i in input_range:
                        btree.insert(i)

                    for i in input_range:
                        key = btree.find(key=i)
                        if key is None:
                            self.fail(f"cannot find key {i}")

                    shuffle(input_range)
                    to_partial_delete = input_range[: len(input_range) // 2]

                    for i in to_partial_delete:
                        try:
                            affected = btree.delete(i)
                        except Exception:
                            self.fail(f"fail to delete key {i}, input {input_range}")
                        if affected == 0:
                            self.fail(f"fail to delete key {i}, input {input_range}")

                    # find the rest of undeleted key
                    leftover_keys = input_range[len(input_range) // 2 :]
                    for i in leftover_keys:
                        key = btree.find(key=i)
                        if key is None:
                            self.fail(f"key {i} is missing")

                    # delete the rest of the keys
                    for i in leftover_keys:
                        affected = btree.delete(key=i)
                        if affected == 0:
                            self.fail(f"fail to delete key {i}, input {input_range}")

                    # tree should be empty
                    for i in input_range:
                        key = btree.find(key=i)
                        if key is not None:
                            self.fail(f"key {key} is still exists")

    def test_insert(self):
        btree = BPlusTree(order=4)
        test_data = [i for i in range(15)]
        for i in test_data:
            k = i
            btree.insert(k)
            btree.graph(i)

        btree.merge_pdfs()
