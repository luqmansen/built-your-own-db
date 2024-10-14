package caskdb

type entry struct {
	header headerEntry
	key    []byte
	value  []byte
}

func newEntry(timestamp int64, key, value []byte) *entry {
	header := headerEntry{
		timestamp: timestamp,
		keySize:   uint64(len(key)),
		valueSize: uint64(len(value)),
	}

	return &entry{
		header: header,
		key:    key,
		value:  value,
	}
}

func (e *entry) encode() (int64, []byte) {
	headerByte := e.header.encode()
	length := len(headerByte) + len(e.key) + len(e.value) // length of this entry
	data := make([]byte, length)

	copy(data[:len(headerByte)], headerByte)
	copy(data[len(headerByte):], e.key)
	copy(data[len(headerByte)+len(e.key):], e.value)

	return int64(length), data
}

func decodeEntry(data []byte) entry {
	header := data[0:defaultHeaderLength]
	h := decodeHeader(header)

	key := data[defaultHeaderLength : defaultHeaderLength+h.keySize]
	val := data[defaultHeaderLength+h.keySize:]

	return entry{
		header: h,
		key:    key,
		value:  val,
	}
}
