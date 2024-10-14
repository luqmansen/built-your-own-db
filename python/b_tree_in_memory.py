"""
B+Tree implementation in python using disk persistence


"""

import bisect
from math import ceil
from typing import TypeVar, Optional, List

import graphviz

T = TypeVar("T")


class TreeNode:
    def __init__(self, is_leaf=False, order=3):
        self.order: int = order  # max number of keys in a node
        self.is_leaf: bool = is_leaf
        self.keys: T = []
        self.children: List["TreeNode"] = []
        self.parent: Optional[TreeNode] = None

        self.next: Optional[TreeNode] = None
        self.previous: Optional[TreeNode] = None

    def __str__(self):
        return str([key for key in self.keys])

    @property
    def max_num_of_keys(self):
        return self.order - 1

    @property
    def max_num_of_child(self):
        return self.order

    @property
    def min_num_of_key(self):
        return ceil(self.order / 2) - 1

    def insert(self, key):
        bisect.insort(self.keys, key)

    def split(self):
        mid_idx = len(self.keys) // 2
        mid_key = self.keys[mid_idx]

        sibling_node = TreeNode(is_leaf=self.is_leaf, order=self.order)
        # splitting separator key to sibling node
        sibling_node.keys = self.keys[mid_idx + 1 :]
        self.keys = self.keys[: mid_idx + 1]

        # only leaf that needs next reference, will help on deletion
        if self.is_leaf:
            self.next = sibling_node
            sibling_node.previous = self

        # only non-leaf that has children, it needs to be migrated to new node.
        if not self.is_leaf:
            # move the children of origin node to the sibling node
            sibling_node.children = self.children[mid_idx + 1 :]
            self.children = self.children[: mid_idx + 1]

            for child in sibling_node.children:
                child.parent = sibling_node

        return mid_key, sibling_node


class BPlusTree:
    def __init__(self, order=4):
        self.root = TreeNode(is_leaf=True, order=order)
        self.order = order

    def _find_leaf_node(self, node, key) -> TreeNode:
        if node.is_leaf:
            return node

        # todo: binary search
        for i, separator_key in enumerate(node.keys):
            if key <= separator_key:
                return self._find_leaf_node(node.children[i], key)

        return self._find_leaf_node(node.children[-1], key)

    def _split_and_promote(self, node: TreeNode):
        if len(node.keys) <= node.max_num_of_keys:
            return

        mid_key, sibling = node.split()

        if node.parent is None:
            new_root = TreeNode(is_leaf=False, order=self.order)
            new_root.keys = [mid_key]
            new_root.children = [node, sibling]
            node.parent = new_root
            sibling.parent = new_root
            self.root = new_root
        else:
            parent = node.parent
            sibling.parent = parent

            bisect.insort(parent.keys, mid_key)
            parent.children.insert(parent.keys.index(mid_key) + 1, sibling)

            if len(parent.keys) > node.max_num_of_keys:
                self._split_and_promote(parent)

    def insert(self, key):
        leaf = self._find_leaf_node(self.root, key)
        leaf.insert(key)

        if len(leaf.keys) > leaf.max_num_of_keys:
            self._split_and_promote(leaf)

    def find(self, key):
        node = self._find_leaf_node(self.root, key)
        if key in node.keys:
            return key
        return None

    def graph(self):
        dot = graphviz.Digraph()
        dot.attr("node", shape="square")

        edges = set()

        from queue import Queue

        queue = Queue()
        queue.put(self.root)

        while queue.empty() is False:
            node = queue.get()
            dot.node(str(node), str(node.keys))

            if node.parent:
                edge = f"{str(node.parent)}-{str(node)}"
                if edge not in edges:
                    dot.edge(str(node.parent), str(node))
                    edges.add(edge)

            for child in node.children:
                queue.put(child)

        dot.render("graph", view=True)
