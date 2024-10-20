package caskdb

import (
	"strconv"
)

type Options struct {
	maxFileSize int64
}

func NewOptions() *Options {
	return &Options{}
}

func (o *Options) SetMaxFileSize(size string) *Options {
	unit := size[len(size)-2:]
	actualSize, err := strconv.ParseFloat(size[:len(size)-2], 32)
	if err != nil {
		panic(err)
	}
	switch unit {
	case "KB":
		o.maxFileSize = int64(actualSize * 1024)
	case "MB":
		o.maxFileSize = int64(actualSize * 1024 * 1024)
	case "GB":
		o.maxFileSize = int64(actualSize * 1024 * 1024 * 1024)
	default:
		panic("size unit unknown, please use one of these: KB,MB,GB")
	}

	return o
}
