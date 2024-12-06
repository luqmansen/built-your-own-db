"""
B+Tree implementation in python with in-memory storage
"""

import bisect
import functools
import io
import os
import threading
from datetime import datetime
from io import DEFAULT_BUFFER_SIZE
from logging import getLogger
from random import shuffle
from threading import Thread
from typing import Optional, List, BinaryIO, Dict, Tuple
from uuid import uuid4

import graphviz

logger = getLogger(__name__)


BUFFER_SIZE = DEFAULT_BUFFER_SIZE  # 8192 bytes/8KB

TOMBSTONE = 0xFFFFFFFF


class Items(list):

    def __repr__(self):
        # decode bytes to string
        content = [int.from_bytes(item, "big") for item in self]
        return f"{self.__class__.__name__}({content})"


"""

Refactoring process

1. PageManager
    Class to handle
    - Page creation
    - Page retrieval
    - Page split
    
    Interface
    
    public
    - create_page()
    _ get_page(id:int)
    
    
    private
    
    
"""


class Page:
    """
    Page is abstraction of disk block, representing a node in B+Tree
    Page is stored in disk, and it is loaded into memory when needed.

    Layout
    ------ internal node
    | page_id | parent_exists | parent_id | is_leaf | key_count | key_size | key1 | page_id_ptr1 | ... | keyN | page_id_ptrN |
    | 4 bytes |  1 byte     | 4 bytes   | 1 byte  | 4 byte   | 4 bytes  | 4 bytes  |   4 bytes | ... | 4 bytes | 4 bytes  |
    ------

    ------ leaf node
    | page_id | parent_exists| parent_id | is_leaf | key_count | key_size | value_size | key1 | value1 | ... | keyN | valueN |
    | 4 bytes | 1 byte       | 4 bytes   | 1 byte  | 4 byte   | 4 bytes  | 4 bytes    |4 bytes | 4 bytes | ... | 4 bytes | 4 bytes  |
    ------

    `page_id_ptr` is the pointer to the child node in the B+Tree. To locate the child node, we need to find the actual
    file offset in the `PageIndex` file.
    """

    _lock = threading.Lock()

    PAGE_ID_SIZE = 4
    PARENT_EXISTS_SIZE = 1
    PARENT_ID_SIZE = 4
    IS_LEAF_SIZE = 1
    KEY_COUNT_SIZE = 4
    # The number of child might be different depending on the internal node position
    # Rightmost node will have one more child than the number of keys
    # The other  internal nodes will have the same number of children as the number of keys
    CHILD_COUNT_SIZE = 4
    CHILD_PTR_SIZE = 4
    KEY_LENGTH_SIZE = 4
    VALUE_LENGTH_SIZE = 4

    @staticmethod
    def with_lock(lock):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with lock:
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def __init__(
        self,
        db_file: BinaryIO,
        page_size: int,
        page_id: int,
        start_offset: int,
        end_offset: int,
        is_leaf: bool = False,
        parent_page_id: int = -1,  # -1 means no parent
        key_size: Optional[int] = 4,
        value_size: Optional[int] = 4,
    ):
        assert start_offset < end_offset, "Invalid page offset"

        self._db_file: BinaryIO = db_file
        self._page_size: int = page_size
        self._start_offset: int = start_offset
        self._end_offset: int = end_offset

        # will either be populated from the file or from the constructor
        self._page_id: int = page_id
        self._parent_id: int = parent_page_id
        self.is_leaf: bool = is_leaf

        self._key_size: int = key_size
        self._value_size: int = value_size

        # public attributes
        self.keys: List[bytes] = Items()
        self.children: List[bytes] = Items()
        self.values: List[bytes] = Items()

        self.is_dirty = False

    # automatically to update the _children_count when children array is modified
    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, value):
        self._children = value

    @property
    def keys(self):
        return self._keys

    @keys.setter
    def keys(self, value):
        assert isinstance(value, list), f"Keys must be a list, got {type(value)}"
        self._keys = value

    def __repr__(self):
        page_id = f"<Page {self._page_id}>"

        if self.is_leaf:
            page_type = "Leaf"
        elif self._parent_id == -1:
            page_type = "Root"
        else:
            page_type = "Internal"

        keys = f"Keys: {[int.from_bytes(k, "big") for k in self.keys]}"
        children = f"Children {[int.from_bytes(c, "big") for c in self.children]}"

        return f"{page_id}| parent: {self.parent_id} | {page_type} | {keys}| {children}"

    @property
    def id(self):
        return self._page_id

    @property
    def parent_id(self):
        return self._parent_id

    @property
    def key_size(self):
        return self._key_size

    @property
    def value_size(self):
        return self._value_size

    @property
    def _header_size(self):
        """
        Header size of the page
        | page_id | parent_exists | parent_id | is_leaf | key_count | child_count | key_size | value_size | ... |
        | 4 bytes | 1 byte        | 4 bytes   | 1 byte  | 4 bytes  | 4 bytes     | 4 bytes  | 4 bytes    | ... |
        """
        return (
            self.PAGE_ID_SIZE
            + self.PARENT_EXISTS_SIZE
            + self.PARENT_ID_SIZE
            + self.IS_LEAF_SIZE
            + self.KEY_COUNT_SIZE
            + self.CHILD_COUNT_SIZE
            + self.KEY_LENGTH_SIZE
            + self.VALUE_LENGTH_SIZE
        )

    @property
    def max_num_of_keys(self):
        # if it's a leaf, we can just count size of the key value pairs,
        # otherwise we need to consider the children pointers + 1
        body_size = self._page_size - self._header_size
        if self.is_leaf:
            return body_size // (self._key_size + self._value_size)
        return body_size // (
            self._key_size + self._value_size + self.CHILD_PTR_SIZE
        )  # 4 bytes for page_id_ptr to last child

    def load(self):
        fd = self._db_file.fileno()
        data = os.pread(
            fd,
            self._page_size,
            self._start_offset,
        )
        assert data != b"", f"Page {self._page_id} is empty"
        self._deserialize(data)

    def _deserialize(self, data: bytes):
        self._page_id = int.from_bytes(data[0:4], "big")
        parent_id_exists = bool.from_bytes(data[4:5], "big")
        if parent_id_exists:
            self._parent_id = int.from_bytes(data[5:9], "big", signed=True)
        self.is_leaf = bool.from_bytes(data[9:10], "big")
        key_count = int.from_bytes(data[10:14], "big")
        children_count = int.from_bytes(data[14:18], "big")
        self._key_size = int.from_bytes(data[18:22], "big")
        self._value_size = int.from_bytes(data[22:26], "big")

        assert (
            self._key_size > 0
        ), f"Key size must be greater than 0, got {self._key_size}"
        assert (
            self._value_size > 0
        ), f"Value size must be greater than 0, got {self._value_size}"

        data = data[self._header_size :]  # skip the header

        for i in range(0, key_count * self.key_size, self._key_size):
            key = data[i : i + self._key_size]
            self.keys.append(key)

        data = data[len(self.keys) * self._key_size :]  # skip the keys
        if self.is_leaf:
            for i in range(0, key_count * self.value_size, self._value_size):
                value = data[i : i + self._value_size]
                self.values.append(value)
        else:
            for i in range(
                0, children_count * self.CHILD_PTR_SIZE, self.CHILD_PTR_SIZE
            ):
                child = data[i : i + self.CHILD_PTR_SIZE]
                self.children._append(child)

        assert (
            len(self.keys) == key_count
        ), f"Key count mismatch, expected {key_count}, got {len(self.keys)}"

        assert (
            len(self.children) == children_count
        ), f"Children count mismatch, expected {children_count}, got {len(self.children)}"

    def _flush(self) -> int:
        # dump entire page content to disk
        data = (
            self._page_id.to_bytes(4, "big")
            + int(self._parent_id != -1).to_bytes(1, "big")
            + (
                self._parent_id.to_bytes(4, "big")
                if self._parent_id != -1
                else b"\x00\x00\x00\x00"
            )
            + int(self.is_leaf).to_bytes(1, "big")
            + len(self.keys).to_bytes(4, "big")
            + len(self.children).to_bytes(4, "big")
            + self._key_size.to_bytes(4, "big")
            + self._value_size.to_bytes(4, "big")
        )
        assert (
            len(data) == self._header_size
        ), f"Data size mismatch, expected {self._header_size}, got {len(data)}"

        for key in self.keys:
            data += key

        if self.is_leaf:
            for value in self.values:
                data += value
        else:
            for child in self.children:
                data += child

        assert (
            self._start_offset + len(data) <= self._end_offset
        ), f"Data size is larger than the page size {len(data)} > {self._end_offset - self._start_offset}"

        fd = self._db_file.fileno()
        written = os.pwrite(fd, data, self._start_offset)
        assert written == len(data)

        return written

    @with_lock(_lock)
    def insert(self, key: bytes, value: bytes):
        # assert key size and value size does not exceed the limit
        assert (
            len(key) == self._key_size
        ), f"Key size mismatch, expected {self._key_size}, got {len(key)}"
        assert (
            len(value) == self._value_size
        ), f"Value size mismatch, expected {self._value_size}, got {len(value)}"

        bisect.insort(self.keys, key)
        # insert the value at the same index as the key
        assert self.is_leaf, "Only leaf node can be inserted with value"

        self.values.insert(bisect.bisect_left(self.keys, key), value)

        # self.is_dirty = True

        self.is_dirty = False

    @with_lock(_lock)
    def split(self):
        mid_idx = len(self.keys) // 2
        mid_key = self.keys[mid_idx]

        sibling_page = Page(
            db_file=self._db_file,
            page_size=self._page_size,
            is_leaf=self.is_leaf,
            parent_page_id=self._parent_id,
            key_size=self._key_size,
            value_size=self._value_size,
        )

        initial_key_count = len(self.keys)
        # splitting separator key to sibling node
        sibling_page.keys = self.keys[mid_idx + 1 :]
        sibling_page.values = self.values[mid_idx + 1 :]

        self.keys = self.keys[: mid_idx + 1]
        self.values = self.values[: mid_idx + 1]

        key_count_after_split = len(self.keys) + len(sibling_page.keys)

        assert (
            initial_key_count == key_count_after_split
        ), f"Key count mismatch, expected {initial_key_count}, got {key_count_after_split}"

        if not self.is_leaf:
            sibling_page.children = self.children[mid_idx + 1 :]
            self.children = self.children[: mid_idx + 1]

            for child_id_bytes in sibling_page.children:
                # update the parent id of the children
                child_page_id = int.from_bytes(child_id_bytes, "big")
                child_page = Page.from_page_id(
                    page_id=child_page_id,
                    db_file=self._db_file,
                    page_index=self._page_index,
                )
                child_page._parent_id = sibling_page.id
                child_page._flush()

        sibling_page._flush()
        self._flush()

        return sibling_page._page_id, mid_key

    def close(self):
        self._flush()
        self._db_file.close()


class AtomicPageID:
    """
    Shared singleton class to generate unique page ID
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._page_id = 0

    def get_and_increment_page_id(self):
        with self._lock:
            page_id = self._page_id
            self._page_id += 1
        return page_id


AtomicPageID.get_and_increment_page_id()


class PageManager:
    """
    PageIndex is a mapping of `page_id` and its corresponding location in the main file.
    This file is stored as separate file than the b+tree.

    When you delete a page, the corresponding entry in the index file is also removed.

    The removal is not in-place, but it is marked as deleted.
    The index file is compacted when the file is loaded, closed or when explicitly called `compact()`

    The tombstone represented as `page_id = -1` and `offset = -1`


    Layout
    ------
    | page_id | start offset | end offset | ... |
      4 bytes |4 bytes.      | 4 bytes    | ...  |
    ------
    """

    def __init__(
        self,
        db_file: BinaryIO,
        page_size=BUFFER_SIZE,
    ):
        self._db_file: BinaryIO = db_file
        page_mapping = f"{db_file.name}.index"
        self._page_mapping_file, created = self._get_or_create_index_file(page_mapping)
        self._index_lock = threading.Lock()

        self.page_size = page_size

        # loaded the index into memory
        # page_id -> (start offset, end offset)
        self._index: Dict[int, Tuple[int, int]] = {}

        # track which page id is located in the current index file
        # page_id -> local offset in the index file
        self._index_local: Dict[int, int] = {}

        # track file pointer for append
        self._page_mapping_file.seek(
            0, io.SEEK_SET
        )  # move to the beginning of the file
        # the last offset of main file, not the index file
        self._last_offset = 0

        if not created:
            self._load()

        self._page_id_lock = threading.Lock()

        latest_page_id = max(self._index.keys(), default=0)
        AtomicPageID()._page_id = latest_page_id

        self._current_page_id = latest_page_id

    def __getitem__(self, item) -> Tuple[int, int]:
        with self._index_lock:
            return self._index[item]

    def create_page(self) -> Page:
        page_id = self._reserve_page()
        start, end = self._index[page_id]
        return Page(
            db_file=self._page_mapping_file,
            start_offset=start,
            end_offset=end,
            page_size=self.page_size,
            page_id=page_id,
        )

    def get_page(self, page_id: int) -> Page:
        start, end = self._index[page_id]
        page = Page(
            db_file=self._db_file,
            start_offset=start,
            end_offset=end,
            page_id=page_id,
            page_size=self.page_size,
        )
        page.load()
        return page

    def split_page(self, page: Page) -> Tuple[Page, Page]:

        page.split()

    def _reserve_page(self) -> int:
        with self._page_id_lock:
            page_id = self._current_page_id
            self._current_page_id += 1
            start_offset = self._last_offset
            self._append(
                page_id=page_id,
                start=start_offset,
                end=start_offset + self.page_size,
            )
            self._last_offset += self.page_size

        return page_id

    def _get_or_create_index_file(self, file_name: str) -> Tuple[BinaryIO, bool]:
        try:
            file = open(file_name, mode="r+b", buffering=0)
            return file, False
        except FileNotFoundError:
            file = open(file_name, "w+b", buffering=0)  # always flush to disk
            return file, True

    def close(self):
        self._page_mapping_file.flush()
        self._page_mapping_file.close()

    def _load(self):
        with self._index_lock:
            while True:
                data = self._page_mapping_file.read(12)
                if data == b"":
                    break

                page_id = int.from_bytes(data[0:4], "big")
                start_offset = int.from_bytes(data[4:8], "big")
                end_offset = int.from_bytes(data[8:12], "big")

                if page_id == TOMBSTONE:
                    continue

                self._index[page_id] = (start_offset, end_offset)
                self._index_local[page_id] = len(data)

                self._last_offset += self.page_size

    def _append(self, page_id: int, start: int, end: int):
        assert page_id not in self._index, f"Page {page_id} already exists in the index"
        assert isinstance(page_id, int), f"Page ID must be integer, got {type(page_id)}"

        self._index_lock.acquire()
        self._index[page_id] = (start, end)

        self._page_mapping_file.write(
            page_id.to_bytes(4, "big")
            + start.to_bytes(4, "big")
            + end.to_bytes(4, "big"),
        )
        self._index_local[page_id] = (
            self._page_mapping_file.tell() - 12  # start of the current entry's offset
        )

        self._index_lock.release()

    def _remove(self, page_id: int):
        with self._index_lock:
            assert page_id in self._index, f"Page {page_id} does not exist in the index"

            del self._index[page_id]

            tombstone = TOMBSTONE.to_bytes(4, "big") * 3

            os.pwrite(
                self._page_mapping_file.fileno(),
                tombstone,
                self._index_local[page_id],
            )
            self._index_local[page_id] = -1


class BPlusTreeOnDisk:
    """
    On Disk B+Tree

    Layout
    ------
    | page_size | root_page_id| ... all pages ... |
    | 4 bytes   | 4 bytes     | (page_size) bytes |
    """

    def __init__(
        self,
        file_name: str,
        key_size: int,
        value_size: int,
        page_size: int = BUFFER_SIZE,
    ):
        self._page_size = page_size
        self._file, created = self._get_or_create_db_file(file_name)
        # assert created is True, "Only support creating new db file"

        self._page_index = PageManager(f"{file_name}.index", page_size=page_size)
        self._root_page_id = None

        if not created:
            self._parse_header()
            assert (
                self._page_size > 0
            ), f"Failed to read db file, got page size {self._page_size}"
            assert (
                self._root_page_id is not None
            ), "Failed to read db file, root page ID is not set"
        else:
            self._page_size = page_size
            self._root_page_id = self._page_index._reserve_page()

            page = Page(
                db_file=self._file,
                page_index=self._page_index,
                page_size=self._page_size,
                page_id=self._root_page_id,
                is_leaf=True,
                key_size=key_size,
                value_size=value_size,
            )
            page._flush()

    def _find_leaf_page(self, page_id: int, key) -> Page:
        if key == int(14).to_bytes(4, "big"):
            print(key)
        page = self._load_page(page_id)
        if page.is_leaf:
            return page

            # todo: binary search
        for i, k in enumerate(page.keys):
            if key <= k:
                child_page_id = int.from_bytes(page.children[i])
                return self._find_leaf_page(child_page_id, key)

        return self._find_leaf_page(int.from_bytes(page.children[-1]), key)

    def insert(self, key: bytes, value: bytes):
        leaf_page = self._find_leaf_page(self._root_page_id, key)
        leaf_page.insert(key, value)

        # overflow
        if len(leaf_page.keys) > leaf_page.max_num_of_keys:
            self._split_and_promote(leaf_page)
        else:
            leaf_page._flush()

    def _split_and_promote(self, page: Page):
        sibling_page_id, mid_key = page.split()

        if page.id == self._root_page_id:
            # create new root
            new_root_page_id = self._page_index._reserve_page()
            new_root_page = Page(
                page_id=new_root_page_id,
                db_file=self._file,
                page_size=self._page_size,
                page_index=self._page_index,
                is_leaf=False,
                parent_page_id=-1,
                key_size=page.key_size,
                value_size=page.value_size,
            )
            new_root_page.keys = [mid_key]
            new_root_page.children = [
                page.id.to_bytes(4, "big"),
                sibling_page_id.to_bytes(4, "big"),
            ]

            self._root_page_id = new_root_page_id
            new_root_page._flush()

            page._parent_id = new_root_page_id
            sibling_page = self._load_page(sibling_page_id)
            sibling_page._parent_id = new_root_page_id

            page._flush()
            sibling_page._flush()

            assert len(set(page.children)) == len(
                page.children
            ), f"Duplicate children in current page {page.children}"
            assert len(set(sibling_page.children)) == len(
                sibling_page.children
            ), f"Duplicate children in sibling_page page {sibling_page.children}"

        else:
            # promote the mid key to the parent

            assert page.parent_id != -1, "This leaf page must have a parent"

            parent_page = self._load_page(page.parent_id)
            sibling_page = self._load_page(sibling_page_id)

            sibling_page._parent_id = parent_page.id

            bisect.insort(parent_page.keys, mid_key)

            idx = parent_page.keys.index(mid_key)

            child_count_before = len(parent_page.children)

            parent_page.children.insert(idx + 1, sibling_page_id.to_bytes(4, "big"))

            assert (
                len(parent_page.children) == child_count_before + 1
            ), f"Child count mismatch, expected {child_count_before + 1}, got {len(parent_page.children)}"

            sibling_page._flush()

            assert len(set(parent_page.keys)) == len(
                parent_page.keys
            ), f"Duplicate keys in parent page {parent_page.keys}"

            assert len(set(parent_page.children)) == len(
                parent_page.children
            ), f"Duplicate children in parent page {parent_page.children}"

            if len(parent_page.keys) > parent_page.max_num_of_keys:
                self._split_and_promote(parent_page)

            parent_page._flush()

    def search(self, key: bytes) -> Optional[bytes]:
        leaf_page = self._find_leaf_page(self._root_page_id, key)
        for i, k in enumerate(leaf_page.keys):
            if key == k:
                return leaf_page.values[i]
        return None
        # idx = bisect.bisect_left(leaf_page.keys, key)
        # if idx <= len(leaf_page.keys) and leaf_page.keys[idx] == key:
        #     return leaf_page.values[idx]
        # return None

    def _get_or_create_db_file(self, file_name: str) -> Tuple[BinaryIO, bool]:
        try:
            file = open(file_name, mode="r+b", buffering=self._page_size)
            return file, False
        except FileNotFoundError:
            file = open(file_name, "w+b", buffering=self._page_size)
            return file, True

    def _parse_header(self):
        fd = self._file.fileno()
        data = os.pread(
            fd,
            8,  # page_size + root_page_id
            0,
        )
        self._page_size = int.from_bytes(data[0:4], "big")
        self._root_page_id = int.from_bytes(data[4:8], "big")

    def _write_header(self):
        data = self._page_size.to_bytes(4, "big") + self._root_page_id.to_bytes(
            4, "big"
        )
        written = os.pwrite(self._file.fileno(), data, 0)
        assert written == 8

    def _load_page(self, page_id: int) -> Page:
        page = Page.from_page_id(
            page_id=page_id,
            db_file=self._file,
            page_index=self._page_index,
        )
        return page

    def _flush(self):
        self._write_header()

    def close(self):
        self._flush()
        self._file.close()
        self._page_index.close()

    def graph(self, step, key="") -> str:
        from queue import Queue

        dot = graphviz.Digraph()
        dot.attr("node", shape="square")

        edges = set()
        queue = Queue()
        queue.put(self._root_page_id)

        # dot.node(f"order: {step}\n deleted_key: {st}")

        while queue.empty() is False:
            page_id = queue.get()
            page = self._load_page(page_id)
            dot.node(
                name=str(page.id),
                label=f"Page: {page.id}"
                f"\nKeys: {page.keys.__repr__()}"
                f"\nChild: {page.children.__repr__()}"
                f"\nValues: {page.values.__repr__()}",
            )

            if page.parent_id != -1:
                parent = self._load_page(page.parent_id)

                edge = f"{parent.id}-{page.id}"
                if edge not in edges:
                    dot.edge(str(parent.id), str(page.id))
                    edges.add(edge)

            for child_id in page.children:
                queue.put(int.from_bytes(child_id, "big"))

            # graph the leaf nodes references
            # if node.is_leaf:
            #     if node.next:
            #         edge = f"{str(node)}-{str(node.next)}-next"
            #         if edge not in edges:
            #             dot.edge(
            #                 str(node),
            #                 str(node.next),
            #                 tailport="e",
            #                 headport="w",
            #                 constraint="false",
            #             )
            #             edges.add(edge)
            #
            #     if node.previous:
            #         edge = f"{str(node)}-{str(node.previous)}-previous"
            #         if edge not in edges:
            #             dot.edge(
            #                 str(node),
            #                 str(node.previous),
            #                 tailport="we",
            #                 headport="s",
            #                 constraint="false",
            #             )
            #             edges.add(edge)

        filename = f"{self}-graph.{step}"
        dot.render(filename, view=False)
        return filename

    def merge_pdfs(self, items: list):
        from pypdf import PdfWriter
        import os

        merger = PdfWriter()

        for pdf in items:
            merger.append(f"{pdf}.pdf")

        merger.write(f"result-{f"{datetime.now().isoformat()}"}.pdf")
        merger.close()
        # remove all pdfs
        for file in items:
            try:
                os.remove(file)
                os.remove(f"{file}.pdf")
            except Exception:
                continue


if __name__ == "__main__":
    test_db_file = "./test_data/b_plus_tree_test+" + str(uuid4())
    btree = BPlusTreeOnDisk(
        file_name=test_db_file,
        page_size=128,
        key_size=4,
        value_size=4,
    )
    test_data = [i for i in range(400)]
    shuffle(test_data)

    pdf_steps = []
    for i in test_data:
        k = i.to_bytes(4, "big")
        btree.insert(k, k)
        # pdf_steps.append(btree.graph(i))

    # btree.merge_pdfs([btree.graph(1)])
    btree.close()

    # reload the tree
    btree_reload = BPlusTreeOnDisk(
        file_name=test_db_file,
        page_size=128,
        key_size=4,
        value_size=4,
    )
    for k in test_data:
        value = btree_reload.search(k.to_bytes(4, "big"))
        assert value == k.to_bytes(4, "big")
