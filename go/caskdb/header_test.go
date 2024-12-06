package caskdb

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func Test_HeaderDecoding(t *testing.T) {
	t.Parallel()

	type args struct {
		timestamp int64
		keySize   uint64
		valueSize uint64
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "key & value dataLength 1B",
			args: args{
				timestamp: time.Now().Unix(),
				keySize:   1,
				valueSize: 1,
			},
		},
		{
			name: "key & value dataLength 1B + 1 bit",
			args: args{
				timestamp: time.Now().Unix(),
				keySize:   256,
				valueSize: 256,
			},
		},
		{
			name: "key & value dataLength 4B",
			args: args{
				timestamp: time.Now().Unix(),
				keySize:   4294967295,
				valueSize: 4294967295,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			header := headerEntry{
				timestamp: tt.args.timestamp,
				keySize:   tt.args.keySize,
				valueSize: tt.args.valueSize,
			}
			b := header.encode()
			headerRes := decodeHeader(b)
			assert.Equal(t, tt.args.timestamp, headerRes.timestamp)
			assert.Equal(t, tt.args.keySize, headerRes.keySize)
			assert.Equal(t, tt.args.valueSize, headerRes.valueSize)
			assert.Equal(t, defaultHeaderLength, len(b)) // encoded header should exactly 12 Byte in length
		})
	}
}
