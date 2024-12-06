"""
B+Tree implementation in python with in-memory storage
"""

import bisect
from logging import getLogger
from math import ceil
from typing import TypeVar, Optional, List

import graphviz

T = TypeVar("T")

logger = getLogger(__name__)


class TreeNode:
    def __init__(self, is_leaf=False, order=3):
        self.order: int = order  # max number of keys in a node
        self.is_leaf: bool = is_leaf
        self.keys: List[T] = []
        self.children: List["TreeNode"] = []
        self.parent: Optional[TreeNode] = None

        self.next: Optional[TreeNode] = None
        self.previous: Optional[TreeNode] = None

    def __repr__(self):
        return str([key for key in self.keys])

    @property
    def max_num_of_keys(self):
        return self.order - 1

    @property
    def max_num_of_child(self):
        return self.order

    @property
    def min_num_of_keys(self):
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

        # todo: create next and previous reference
        #  and properly update them on split and merge on random insertion

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

        self.graph_record = []

    def find(self, key):
        node = self._find_leaf_node(self.root, key)
        if key in node.keys:
            return key
        return None

    def insert(self, key):
        leaf = self._find_leaf_node(self.root, key)
        leaf.insert(key)

        if len(leaf.keys) > leaf.max_num_of_keys:
            self._split_and_promote(leaf)

    def delete(self, key: T):
        """
        Delete a key from the tree
        :return:
        """
        leaf = self._find_leaf_node(self.root, key)

        # for debugging
        self._deleted_key = key

        return self._delete_key(key, leaf)

    def _find_leaf_node(self, node, key) -> TreeNode:
        if node.is_leaf:
            return node

        # todo: binary search
        for i, separator_key in enumerate(node.keys):
            if key <= separator_key:
                return self._find_leaf_node(node.children[i], key)

        return self._find_leaf_node(node.children[-1], key)

    def _split_and_promote(self, node: TreeNode):
        if len(node.keys) < node.max_num_of_keys:
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

    def _delete_key(self, key: T, node: TreeNode) -> int:
        """
        Delete from the entire tree
        """
        if key not in node.keys:
            return 0

        node.keys.remove(key)
        # if key is separator key in parent
        # put the new separator key in the parent
        if node.parent and key in node.parent.keys:
            current_separator_index = node.parent.keys.index(key)
            node.parent.keys[current_separator_index] = node.keys[-1]

        # if not a root node, check for underflow
        # (root node cannot underflow)
        if node.parent is not None:
            self._handle_underflow(node)

        # check if key is separator key in parent
        # propagate the deletion to the parent
        if node.parent and key in node.parent.keys:
            self._delete_key(key, node.parent)

        return 1

    def _retrieve_right_sibling(self, node: TreeNode) -> Optional[TreeNode]:
        # root node
        if node.parent is None:
            return None

        # leaf node has direct reference to sibling
        if node.is_leaf and node.next:
            return node.next

        # internal node
        parent = node.parent
        index = parent.children.index(node)
        if index + 1 < len(parent.children):
            return parent.children[index + 1]

        return None

    def _retrieve_left_sibling(self, node: TreeNode) -> Optional[TreeNode]:
        # root node
        if node.parent is None:
            return None

        # leaf node has direct reference to sibling
        if node.is_leaf and node.previous:
            return node.previous

        # internal node
        parent = node.parent
        index = parent.children.index(node)
        if index - 1 >= 0:
            return parent.children[index - 1]

        return None

    def _is_underflow(self, node: TreeNode) -> bool:
        is_root = node == self.root
        is_leaf = node.is_leaf

        if is_root and is_leaf:
            # leaf root node can have 0 keys
            return False

        if is_root and not is_leaf:
            """
            As the root node must have at least two children,
            and thus at least one key to separate them.
            """
            return len(node.keys) < 1

        # internal nodes
        return len(node.keys) < node.min_num_of_keys

    def _handle_underflow(self, node: TreeNode):
        if not self._is_underflow(node):
            return

        # If the node is the root, and it has no keys,
        # make its first child the new root
        if node.parent is None:
            assert len(node.keys) == 0
            # 0 children means that the tree is empty
            # otherwise, the root has only one child
            assert len(node.children) in [0, 1]
            if len(node.children) > 0:
                self.root = node.children[0]
                self.root.parent = None
            return

        left_sibling = self._retrieve_left_sibling(node)
        if left_sibling and len(left_sibling.keys) > node.min_num_of_keys:
            self._borrow_from_left(node)

        right_sibling = self._retrieve_right_sibling(node)
        if right_sibling and len(right_sibling.keys) > node.min_num_of_keys:
            self._borrow_from_right(node)
            return

        self._merge_with_sibling(node)

    def _borrow_from_left(self, node: TreeNode):
        left_sibling = self._retrieve_left_sibling(node)
        assert left_sibling is not None

        # find the separator key of current node in the parent's keys
        # eg:
        # parent_keys: [ 1 |  3 | ]
        # child      : [0,1 | 2,3| 4]
        # to delete: 3
        # index: 0
        # separator_key: 1
        parent = node.parent
        separator_key_index = parent.children.index(node) - 1
        borrowed_key = left_sibling.keys.pop()

        # Insert the borrowed key at
        # the beginning of the current node's keys
        node.keys.insert(0, borrowed_key)

        # copy the last key from the left sibling to the parent as new separator key
        parent.keys[separator_key_index] = left_sibling.keys[-1]

        # If the current node is not a leaf (i.e. has a child),
        # we need to migrate the right-most child from the left
        # sibling to the current node
        # eg:
        # parent_keys:  [ 1 |  3  ]       [8 | 10 ]
        # child      : [0,1 | 2,3| 4]  [5, 8 | 9, 10 ]
        #  we move key=4 to before the key=5
        if not node.is_leaf:
            child = left_sibling.children.pop()
            node.children.insert(0, child)
            child.parent = node

    def _borrow_from_right(self, node: TreeNode):
        right_sibling = self._retrieve_right_sibling(node)
        assert right_sibling is not None
        parent = node.parent

        separator_key_index = parent.children.index(node)
        borrowed_key = right_sibling.keys.pop(0)

        parent.keys[separator_key_index] = borrowed_key
        node.keys.append(borrowed_key)

        if not node.is_leaf:
            child = right_sibling.children.pop(0)
            node.children.append(child)
            child.parent = node

    def _merge_with_sibling(self, node: TreeNode):
        # Find the index of the current node in the parent's children list
        parent = node.parent
        assert parent is not None

        node_index = parent.children.index(node)

        if left_sibling := self._retrieve_left_sibling(node):
            # *** merging current node TO left sibling***
            # current node = origin node
            # left sibling = destination node

            # remove the destination node separator key from the parent
            parent.keys.pop(node_index - 1)

            # move all keys from the current node to left sibling
            left_sibling.keys.extend(node.keys)

            # update the next pointer of the left sibling
            if node.is_leaf:
                left_sibling.next = node.next

            # if current node is an internal node, move all children to the left sibling
            if not node.is_leaf:
                left_sibling.children.extend(node.children)
                for child in node.children:
                    child.parent = left_sibling

            # remove current node from parent's children list
            parent.children.pop(node_index)

            # if parent is underflow, repeat the process
            if len(parent.keys) < parent.min_num_of_keys:
                self._handle_underflow(parent)

        else:
            # *** merging right sibling TO current node ***

            right_sibling = self._retrieve_right_sibling(node)
            assert right_sibling is not None
            assert right_sibling is not None

            parent.keys.pop(node_index)

            # move all keys form right sibling to current node
            node.keys.extend(right_sibling.keys)

            # update the next pointer of the left sibling
            if node.is_leaf:
                node.next = right_sibling.next

            if not node.is_leaf:
                node.children.extend(right_sibling.children)
                for child in right_sibling.children:
                    child.parent = node

            # Remove the right sibling from the parent's children list
            # instead of removing the current node (hence the +1)
            parent.children.pop(node_index + 1)

            if len(parent.keys) < parent.min_num_of_keys:
                self._handle_underflow(parent)

    def graph(self, step, key=""):
        from queue import Queue

        dot = graphviz.Digraph()
        dot.attr("node", shape="square")

        edges = set()
        queue = Queue()
        queue.put(self.root)

        # dot.node(f"order: {step}\n deleted_key: {self._deleted_key}")

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

            # graph the leaf nodes references
            if node.is_leaf:
                if node.next:
                    edge = f"{str(node)}-{str(node.next)}-next"
                    if edge not in edges:
                        dot.edge(
                            str(node),
                            str(node.next),
                            tailport="e",
                            headport="w",
                            constraint="false",
                        )
                        edges.add(edge)

                if node.previous:
                    edge = f"{str(node)}-{str(node.previous)}-previous"
                    if edge not in edges:
                        dot.edge(
                            str(node),
                            str(node.previous),
                            tailport="we",
                            headport="s",
                            constraint="false",
                        )
                        edges.add(edge)

        dot.render(f"{self}-graph-{step}", format="pdf")

    def merge_pdfs(self):
        from pypdf import PdfWriter
        import os

        # read curent directory for all pdfs
        pdfs = list(
            [
                f
                for f in os.listdir()
                if f.endswith(".pdf")
                and f.startswith(
                    f"{'<python.btree.b_tree_in_memory.BPlusTree object at 0x10edcfb90>'}-graph"
                )
            ]
        )

        merger = PdfWriter()

        for pdf in sorted(pdfs, key=lambda x: int(x[-5:].strip(".pdf"))):
            merger.append(pdf)

        merger.write(f"{self}-result.pdf")
        merger.close()

        for file in [f for f in os.listdir() if f.startswith(f"{self}-graph")]:
            try:
                os.remove(file)
            except Exception:
                continue
