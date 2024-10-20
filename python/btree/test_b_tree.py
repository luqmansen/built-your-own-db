import random
from random import shuffle
from unittest import TestCase
from uuid import uuid4

from python.btree.b_tree_in_memory import BPlusTree


class BTreeTest(TestCase):
    def test_small_inputs(self):
        test_num_of_keys = 10
        btree = BPlusTree(order=3)
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

    def test_delete_small(self):
        btree = BPlusTree(order=4)
        insert = [i for i in range(25)]
        for i in insert:
            btree.insert(i)
        for i in insert:
            key = btree.find(key=i)
            if key is None:
                self.fail(f"cannot find key {i}")
        delete = insert[:5]
        shuffle(delete)
        for i in insert[:9]:
            affected = btree.delete(i)
            # btree.graph()
            if affected == 0:
                btree.graph()
                self.fail(f"fail to delete key {i}")

        # for i in insert[5:]:
        #     key = btree.find(key=i)
        #     if key is None:
        #         btree.graph()
        #         self.fail(f"cannot find key {i}")

    def test_delete(self):
        btree = BPlusTree(order=5)

        insert = [i for i in range(200)]

        for i in insert:
            btree.insert(i)

        for i in insert:
            key = btree.find(key=i)
            if key is None:
                self.fail(f"cannot find key {i}")

        delete = insert[:20]
        # shuffle(delete)
        for i in delete:
            if i == 16:
                print(i)
            try:
                affected = btree.delete(i)
            except Exception as e:
                btree.graph()
                self.fail(f"fail to delete key {i}")
                raise e

            if affected == 0:
                btree.graph()
                self.fail(f"fail to delete key {i}")

        for i in insert[21:]:
            key = btree.find(key=i)
            if key is None:
                self.fail(f"cannot find key {i}")

    def test_delete_small(self):
        btree = BPlusTree(order=6)
        # turnoff black formatting
        # fmt: off
        input_test_cases = [i for i in range(40)]
        # dangling node case
        input_test_cases = [85, 22, 38, 84, 89, 96, 90, 14, 79, 8, 80, 98, 56, 57, 2, 6, 45, 92, 37, 41, 83, 72, 17, 63, 11, 76, 10, 20, 23, 71, 29, 51, 3, 99, 86, 43, 21, 42, 18, 44, 0, 81, 4, 26, 64, 53, 27, 93, 95, 12, 32, 59, 60, 34, 65, 40, 7, 25, 36, 74, 75, 15, 73, 52, 78, 35, 5, 88, 9, 91, 61, 54, 66, 50, 16, 69, 94, 19, 24, 48, 47, 46, 68, 97, 77, 31, 28, 55, 82, 1, 67, 49, 13, 39, 33, 87, 70, 62, 58, 30]

        # input_test_cases = [31, 12, 19, 24, 33, 18, 20, 35, 7, 15, 10, 4, 22, 17, 8, 26, 1, 38, 5, 0, 37, 25, 11, 2, 13, 39, 36, 30, 34, 32, 3, 29, 9, 21, 16, 14, 23, 28, 6, 27]
        # fmt: on
        # shuffle(input_test_cases)

        for i in input_test_cases:
            btree.insert(i)

        for i in input_test_cases:
            key = btree.find(key=i)
            if key is None:
                self.fail(f"cannot find key {i}")

        delete = input_test_cases
        # shuffle(delete)
        for step, key_to_delete in enumerate(delete):
            try:
                affected = btree.delete(key_to_delete)
            except Exception as e:
                btree.graph(step=step)
                btree.merge_pdfs()

                self.fail(f"fail to delete key {key_to_delete}, array {delete}")
            if affected == 0:
                self.fail(f"fail to delete key {key_to_delete}, array {delete}")
            # btree.graph(step)

        # btree.merge_pdfs()

        for i in input_test_cases:
            key = btree.find(key=i)
            if key is not None:
                self.fail(f"key {i} is still in the tree")

    def test_delete_all(self):
        for order in range(6, 100, 10):
            for input_range in [
                list(range(0, _max)) for _max in [100, 1_000, 10_000, 100_000]
            ]:
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

                    delete = input_range
                    shuffle(delete)
                    for i in delete:
                        try:
                            affected = btree.delete(i)
                        except Exception as e:
                            self.fail(f"fail to delete key {i}, input {input_range}")
                            raise e
                        if affected == 0:
                            self.fail(f"fail to delete key {i}, input {input_range}")

                    for i in input_range:
                        key = btree.find(key=i)
                        if key is not None:
                            self.fail(f"key {i} is still in the tree")
