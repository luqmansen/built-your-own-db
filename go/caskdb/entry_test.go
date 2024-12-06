package caskdb

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func Test_encodeKV(t *testing.T) {
	t.Parallel()

	type args struct {
		timestamp int64
		key       []byte
		value     []byte
	}
	tests := []struct {
		name string
		args args
		want int64
	}{
		{
			name: "normal kv",
			args: args{
				timestamp: time.Now().UnixNano(),
				key:       []byte("hello"),
				value:     []byte("world"),
			},
			want: defaultHeaderLength + 10,
		},
		{
			name: "empty kv",
			args: args{
				timestamp: time.Now().UnixNano(),
				key:       []byte(""),
				value:     []byte(""),
			},
			want: defaultHeaderLength + 0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := newEntry(tt.args.timestamp, tt.args.key, tt.args.value)

			dataLength, data := e.encode()
			assert.Equal(t, tt.want, dataLength)

			decoded := decodeEntry(data)
			assert.Equal(t, tt.args.timestamp, decoded.header.timestamp)
			assert.Equal(t, tt.args.key, decoded.key)
			assert.Equal(t, tt.args.value, decoded.value)
		})
	}
}
