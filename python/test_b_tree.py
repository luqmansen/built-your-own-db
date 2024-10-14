import random
from unittest import TestCase
from uuid import uuid4

from python.b_tree_in_memory import BPlusTree


class BTreeTest(TestCase):
    def test_small_inputs(self):
        test_num_of_keys = 100
        btree = BPlusTree(order=5)
        inputs = [i for i in range(test_num_of_keys)]

        for i in inputs:
            try:
                btree.insert(
                    i,
                )
            except Exception:
                btree.graph()
                self.fail(f"cannot insert key {i}")
        for i in inputs:
            try:
                key = btree.find(key=i)
                if key is None:
                    btree.graph()
                    self.fail(f"cannot find key {i}")
            except:
                self.fail(f"cannot find key {i}")

        btree.graph()

    def test_insert_and_retrieve(self):
        test_num_of_keys = 1000
        test_orders = 100
        test_array = [
            list(range(1, test_num_of_keys + 1)),
            list([str(uuid4()) for _ in range(test_num_of_keys)]),
        ]
        for inputs in test_array:
            for order in range(3, test_orders):
                with self.subTest(
                    msg=f"test orders {order} with input type {type(inputs[0])}"
                ):
                    btree = BPlusTree(order=order)
                    random.shuffle(list(inputs))

                    for i in inputs:
                        btree.insert(i)
                    for i in inputs:
                        key = btree.find(key=i)
                        if key is None:
                            self.fail(f"cannot find key {i}")

    def test_delete_without_underflow(self):
        btree = BPlusTree(order=4)
        for i in range(8):
            btree.insert(i)

        btree.graph()

        btree.delete(2)
        btree.graph()
