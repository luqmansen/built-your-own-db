package caskdb

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestOptions_AddMaxFileSize(t *testing.T) {
	o := Options{}
	t.Run("KByte", func(t *testing.T) {
		o.SetMaxFileSize("1KB")
		assert.Equal(t, int64(1*1024), o.maxFileSize)
	})
	t.Run("Megabyte", func(t *testing.T) {
		o.SetMaxFileSize("10MB")
		assert.Equal(t, int64(10*1024*1024), o.maxFileSize)
	})
	t.Run("Decimal Megabyte", func(t *testing.T) {
		o.SetMaxFileSize("10.5MB")
		assert.Equal(t, int64(10.5*1024*1024), o.maxFileSize)
	})
	t.Run("GByte", func(t *testing.T) {
		o.SetMaxFileSize("100GB")
		assert.Equal(t, int64(100*1024*1024*1024), o.maxFileSize)
	})
}
