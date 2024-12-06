package caskdb

import "encoding/binary"

const (
	// byte length of the header
	defaultHeaderLength = 24
)

// headerEntry will hold header of an entry
type headerEntry struct {
	timestamp int64 // default is using time.UnixNano which produce int64
	keySize   uint64
	valueSize uint64
}

// Encode header of an entry. Each of header item is uint64 which
// takes 8 Byte, so we need to allocate 24 Byte for the header
// ref http://golang.org/ref/spec#Size_and_alignment_guarantees
// | timestamp 8B | keySize 8B | valueSize 8B | -> total allocate 24 Byte
func (h *headerEntry) encode() []byte {
	b := make([]byte, defaultHeaderLength)
	binary.LittleEndian.PutUint64(b[0:], uint64(h.timestamp))
	binary.LittleEndian.PutUint64(b[8:], h.keySize)
	binary.LittleEndian.PutUint64(b[16:], h.valueSize)

	return b
}

func decodeHeader(data []byte) headerEntry {
	return headerEntry{
		timestamp: int64(binary.LittleEndian.Uint64(data[0:])),
		keySize:   binary.LittleEndian.Uint64(data[8:]),
		valueSize: binary.LittleEndian.Uint64(data[16:]),
	}
}
