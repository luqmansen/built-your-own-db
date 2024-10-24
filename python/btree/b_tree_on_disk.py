"""
B+Tree implementation in python with in-memory storage
"""

import io
import os
import threading
from io import DEFAULT_BUFFER_SIZE
from logging import getLogger
from typing import Optional, List, BinaryIO, Dict

logger = getLogger(__name__)


BUFFER_SIZE = DEFAULT_BUFFER_SIZE  # 8192 bytes/8KB

TOMBSTONE = 0xFFFFFFFF


class PageIndex:
    """
    PageIndex is a mapping of `page_id` and its corresponding location in the main file.
    This file is stored as separate file than the b+tree.

    When you delete a page, the corresponding entry in the index file is also removed.

    The removal is not in-place, but it is marked as deleted.
    The index file is compacted when the file is loaded, closed or when explicitly called `compact()`

    The tombstone represented as `page_id = -1` and `offset = -1`


    Layout
    ------
    | page_id | offset |
      4 bytes |4 bytes
    ------
    """

    def __init__(self, file_name):
        self._file: BinaryIO = self._get_or_create_index_file(file_name)
        self._index_lock = threading.Lock()

        # loaded the index into memory
        self._index: Dict[int, int] = {}  # page_id -> main file offset

        # track which page id is located in the current index file
        self._index_local: Dict[int, int] = {}  # page_id -> local file offset
        self._load()

        # track file pointer for append
        self._file.seek(0, io.SEEK_END)
        self._last_offset = 0

    def _get_or_create_index_file(self, file_name: str) -> BinaryIO:
        try:
            file = open(file_name, mode="r+b", buffering=0)
        except FileNotFoundError:
            file = open(file_name, "w+b", buffering=0)  # always flush to disk
        return file

    def close(self):
        self._file.close()

    def _load(self):
        with self._index_lock:
            self._file.seek(0)
            data = self._file.read()
            self._parse_index(data)

    def _parse_index(self, data: bytes):
        for i in range(0, len(data), 8):
            page_id = int.from_bytes(data[i : i + 4], "big", signed=True)
            offset = int.from_bytes(data[i + 4 : i + 8], "big", signed=True)

            if page_id == -1 or offset == -1:
                continue

            self._index[page_id] = offset
            self._index_local[page_id] = i

    def append(self, page_id: int, offset: int):
        assert page_id not in self._index, f"Page {page_id} already exists in the index"
        assert isinstance(page_id, int), f"Page ID must be integer, got {type(page_id)}"

        self._index_lock.acquire()
        self._index[page_id] = offset

        os.pwrite(
            self._file.fileno(),
            page_id.to_bytes(4, "big") + offset.to_bytes(4, "big"),
            self._last_offset,
        )
        self._index_local[page_id] = self._last_offset
        self._last_offset += 8

        self._index_lock.release()

    def remove(self, page_id: int):
        with self._index_lock:
            assert page_id in self._index, f"Page {page_id} does not exist in the index"

            del self._index[page_id]

            os.pwrite(
                self._file.fileno(),
                TOMBSTONE.to_bytes(4, "big") + TOMBSTONE.to_bytes(4, "big"),
                self._index_local[page_id],
            )
            self._index_local[page_id] = -1


class Page:
    """
    Page is abstraction of disk block, representing a node in B+Tree
    Page is stored in disk, and it is loaded into memory when needed.
    Layout
    ------
    | page_id | parent_id | is_leaf | key_count | key1 | page_id_ptr | ... | keyN | page_id_ptr|
    | 4 bytes | 4 bytes   | 1 byte  | 1 byte   | 4 bytes | 4 bytes  | ... | 4 bytes | 4 bytes  |
    ------

    `page_id_ptr` is the pointer to the child node in the B+Tree. To locate the child node, we need to find the actual
    file offset in the `PageIndex` file.
    """

    def __init__(
        self,
        db_file: BinaryIO,
        page_id: int,
        offset: int,
        parent_page_id: Optional[int],
        is_leaf: bool,
    ):
        self.file: BinaryIO = db_file
        self.page_id: int = page_id
        self.offset: int = offset
        self.parent_id: int = parent_page_id
        self.is_leaf: bool = is_leaf

        self._lock = threading.Lock()

        self.keys: List[int] = []
        self.children: List[int] = []

    def _allocate(self):
        self.file.seek(self.offset)
        self.file.write(self.page_id.to_bytes(4, "big"))
        self.file.write(self.parent_id.to_bytes(4, "big"))
        self.file.write(int(self.is_leaf).to_bytes(1, "big"))

    def _load_from_file(self):
        self.file.seek(self.offset)
        data = self.file.read(4096)

        self.parent_id = int.from_bytes(data[4:8], "big")
        self.is_leaf = bool(data[8])
        self.key_count = int(data[9])

        for i in range(10, len(data), 8):
            key = int.from_bytes(data[i : i + 4], "big")
            child = int.from_bytes(data[i + 4 : i + 8], "big")
            self.keys.append(key)
            self.children.append(child)


class BPlusTreeOnDisk:
    """
    On Disk B+Tree

    Layout
    ------
    | page_size | last_offset | root_page_id | ... all pages ... |
    | 4 bytes   | 4 bytes     | 4 bytes      |      ...          |

    """

    def __init__(self, file_name: str, page_size: int = BUFFER_SIZE):
        self.file = self._get_or_create_db_file(file_name)

        self.page_index = PageIndex(f"{file_name}.index")
        self.page_size = page_size

        # counter lock
        self._page_id_lock = threading.Lock()
        self._current_page_id: int = 0

    def _load_root_page(self):
        self.file.seek(8)
        root_page_id = int.from_bytes(self.file.read(4), "big")
        return self._load_page(root_page_id)

    def _load_page(self, page_id: int) -> Page:
        offset = self.page_index._index[page_id]
        self.file.seek(offset)
        data = self.file.read(self.page_size)
        return Page.load_from_offset(self.file, offset)

    def _get_and_increment_page_id(self) -> int:
        with self._page_id_lock:
            page_id = self._current_page_id
            self._current_page_id += 1
        return page_id

    def _get_or_create_db_file(self, file_name: str) -> BinaryIO:
        try:
            file = open(file_name, mode="r+b", buffering=self.page_size)
        except FileNotFoundError:
            file = open(file_name, "w+b", buffering=self.page_size)
        return file
