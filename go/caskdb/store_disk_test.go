package caskdb

import (
	"errors"
	"log"
	"os"
	"path"
	"strconv"
	"sync"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
)

func initStorageHelper(name ...string) (*DiskStorage, string, func()) {
	filename := ""
	baseTestPath := "testdata"                     // location for test file, so we don't clutter root project
	testFolder := name[0] + "_" + uuid.NewString() // each test will get its own folder

	if len(name) >= 1 {
		filename = path.Join(append([]string{baseTestPath, testFolder}, name[1:]...)...)
	} else {
		filename = path.Join(baseTestPath, testFolder, uuid.NewString())
	}

	testfilepath, _ := path.Split(filename)
	if _, err := os.Stat(testfilepath); errors.Is(err, os.ErrNotExist) {
		if err := os.MkdirAll(testfilepath, 0777); err != nil {
			panic(err)
		}
	} else if err != nil {
		panic(err)
	}

	storage := NewDiskStorage(filename)

	cleanup := func() {
		err := storage.Close()
		if err != nil {
			log.Println(err.Error())
		}

		err = os.RemoveAll(path.Join(baseTestPath, testFolder))
		if err != nil {
			log.Println(err.Error())
		}
	}
	return storage, filename, cleanup
}

func Test_initKeyDir_useHintFiles(t *testing.T) {
	t.Parallel()

	store, filename, cleanupFunc := initStorageHelper()
	defer cleanupFunc()

	kv := make(map[string][]byte)
	for i := 0; i <= 10; i++ {
		kv[strconv.Itoa(i)] = []byte(strconv.Itoa(i))
	}
	for k, v := range kv {
		assert.Nil(t, store.Set([]byte(k), v))
	}
	assert.Nil(t, store.Close())

	store = NewDiskStorage(filename)
	for k, v := range kv {
		res, err := store.Get([]byte(k))
		assert.Nil(t, err)
		assert.Equal(t, v, res)
	}
}

func Test_initKeyDir(t *testing.T) {
	t.Parallel()

	store, filename, cleanupFunc := initStorageHelper()
	defer cleanupFunc()

	kv := make(map[string][]byte)
	for i := 0; i <= 10; i++ {
		kv[strconv.Itoa(i)] = []byte(strconv.Itoa(i))
	}
	for k, v := range kv {
		assert.Nil(t, store.Set([]byte(k), v))
	}

	store = NewDiskStorage(filename)
	for k, v := range kv {
		res, err := store.Get([]byte(k))
		assert.Nil(t, err)
		assert.Equal(t, v, res)
	}
}

func TestDiskStorage_singleKey(t *testing.T) {
	t.Parallel()

	store, _, cleanupFunc := initStorageHelper()
	defer cleanupFunc()

	assert.Nil(t, store.Set([]byte("yeet"), []byte("donjon")))
	res, err := store.Get([]byte("yeet"))
	assert.Nil(t, err)
	assert.Equal(t, []byte("donjon"), res)
}

func TestDiskStorage_multiKey(t *testing.T) {
	t.Parallel()

	t.Run("test correct file split", func(t *testing.T) {
		t.Parallel()

		store, filePath, cleanupFunc := initStorageHelper(t.Name(), "test")
		defer cleanupFunc()

		store.WithOptions(NewOptions().SetMaxFileSize("1MB"))

		kv := make(map[string][]byte)
		// this will equal to 3.2 MB of record
		// 1 key consist of 24b header +  2~5 byte of kv pair
		// this should split into 4 files (3x 1MB + 1x ~200KB)
		for i := 0; i <= 100_000; i++ {
			kv[strconv.Itoa(i)] = []byte(strconv.Itoa(i))
		}
		for k, v := range kv {
			assert.Nil(t, store.Set([]byte(k), v))
		}
		for k, v := range kv {
			res, err := store.Get([]byte(k))
			assert.Nil(t, err)
			assert.Equal(t, v, res)
		}
		subPath, _ := path.Split(filePath)
		dirs, err := os.ReadDir(subPath)
		if err != nil {
			panic(err)
		}
		assert.Len(t, dirs, 4) // there should be exactly 4 files in here
	})

	t.Run("test one million key", func(t *testing.T) {
		t.Parallel()

		store, filePath, cleanupFunc := initStorageHelper(t.Name(), "test")
		defer cleanupFunc()

		store.WithOptions(NewOptions().SetMaxFileSize("20MB"))

		kv := make(map[string][]byte)
		for i := 0; i <= 1_000_000; i++ { // this will roughly generate 34MB files
			kv[strconv.Itoa(i)] = []byte(strconv.Itoa(i))
		}
		for k, v := range kv {
			assert.Nil(t, store.Set([]byte(k), v))
		}
		for k, v := range kv {
			res, err := store.Get([]byte(k))
			assert.Nil(t, err)
			assert.Equal(t, v, res)
		}

		subPath, _ := path.Split(filePath)
		dirs, err := os.ReadDir(subPath)
		if err != nil {
			panic(err)
		}
		assert.Len(t, dirs, 2) // there should be exactly 2 files in here
	})

}

func TestDiskStorage_concurrent(t *testing.T) {

	t.Run("concurrent 1K Key, 1KB Filesize", func(t *testing.T) {
		store, filePath, _ := initStorageHelper(t.Name(), "test")
		//defer cleanupFunc()

		store.WithOptions(NewOptions().SetMaxFileSize("1KB"))

		kv := make(map[string][]byte)
		for i := 0; i <= 1_000; i++ { // this will generate ~29KB of data
			kv[strconv.Itoa(i)] = []byte(strconv.Itoa(i))
		}

		limitChan := make(chan struct{}, 8000) // limit maximum number of goroutine on test with race detector

		var wgAdd sync.WaitGroup
		for k, v := range kv {
			wgAdd.Add(1)
			limitChan <- struct{}{}

			go func(k, v []byte) {
				defer wgAdd.Done()

				assert.NoError(t, store.Set(k, v))
				res, err := store.Get(k)
				assert.Nil(t, err)
				equalByte(t, v, res)

				<-limitChan
			}([]byte(k), v)
		}
		wgAdd.Wait()

		var wgGet sync.WaitGroup
		for k, v := range kv {
			wgGet.Add(1)
			limitChan <- struct{}{}

			go func(k string, v []byte) {
				defer wgGet.Done()

				res, err := store.Get([]byte(k))
				assert.Nil(t, err)
				assert.Equal(t, v, res)

				<-limitChan
			}(k, v)
		}
		wgGet.Wait()

		subPath, _ := path.Split(filePath)
		dirs, err := os.ReadDir(subPath)
		if err != nil {
			panic(err)
		}
		assert.Len(t, dirs, 29)
	})

	t.Run("concurrent 10K Key, 1MB Filesize", func(t *testing.T) {
		store, _, cleanupFunc := initStorageHelper(t.Name(), "test")
		defer cleanupFunc()
		store.WithOptions(NewOptions().SetMaxFileSize("1MB"))

		kv := make(map[string][]byte)
		for i := 0; i <= 10_000; i++ {
			kv[strconv.Itoa(i)] = []byte(strconv.Itoa(i))
		}

		limitChan := make(chan struct{}, 8000) // limit maximum number of goroutine on test with race detector

		var wgAdd sync.WaitGroup
		for k, v := range kv {
			wgAdd.Add(1)
			limitChan <- struct{}{}

			go func(k, v []byte) {
				defer wgAdd.Done()

				assert.NoError(t, store.Set(k, v))
				res, err := store.Get(k)
				assert.Nil(t, err)
				equalByte(t, v, res)

				<-limitChan
			}([]byte(k), v)
		}
		wgAdd.Wait()

		var wgGet sync.WaitGroup
		for k, v := range kv {
			wgGet.Add(1)
			limitChan <- struct{}{}

			go func(k string, v []byte) {
				defer wgGet.Done()

				res, err := store.Get([]byte(k))
				assert.Nil(t, err)
				assert.Equal(t, v, res)

				<-limitChan
			}(k, v)
		}
		wgGet.Wait()
	})

	t.Run("concurrent 100K Key, 1MB Filesize", func(t *testing.T) {
		store, _, cleanupFunc := initStorageHelper(t.Name(), "test")
		defer cleanupFunc()
		store.WithOptions(NewOptions().SetMaxFileSize("1MB"))

		kv := make(map[string][]byte)
		for i := 0; i <= 100_000; i++ {
			kv[strconv.Itoa(i)] = []byte(strconv.Itoa(i))
		}

		limitChan := make(chan struct{}, 8000) // limit maximum number of goroutine on test with race detector

		var wgAdd sync.WaitGroup
		for k, v := range kv {
			wgAdd.Add(1)
			limitChan <- struct{}{}

			go func(k, v []byte) {
				defer wgAdd.Done()

				assert.NoError(t, store.Set(k, v))
				res, err := store.Get(k)
				assert.Nil(t, err)
				equalByte(t, v, res)

				<-limitChan
			}([]byte(k), v)
		}
		wgAdd.Wait()

		var wgGet sync.WaitGroup
		for k, v := range kv {
			wgGet.Add(1)
			limitChan <- struct{}{}

			go func(k string, v []byte) {
				defer wgGet.Done()

				res, err := store.Get([]byte(k))
				assert.Nil(t, err)
				assert.Equal(t, v, res)

				<-limitChan
			}(k, v)
		}
		wgGet.Wait()
	})

}

func BenchmarkDiskStorage_Set(b *testing.B) {
	store, _, cleanupFunc := initStorageHelper()
	defer cleanupFunc()

	for i := 0; i < b.N; i++ {
		store.Set([]byte(strconv.Itoa(i)), []byte(strconv.Itoa(i)))
	}
}

func BenchmarkDiskStorage_Get(b *testing.B) {
	store, _, cleanupFunc := initStorageHelper()
	defer cleanupFunc()

	for i := 0; i < 100_000; i++ {
		store.Set([]byte(strconv.Itoa(i)), []byte(strconv.Itoa(i)))
	}
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		store.Get([]byte(strconv.Itoa(i)))
	}
}

// load all key dir value from reading the database files
func BenchmarkNewDiskStorage_from_scratch(b *testing.B) {
	store, filename, cleanupFunc := initStorageHelper()
	defer cleanupFunc()

	for i := 0; i < 100_000; i++ {
		store.Set([]byte(strconv.Itoa(i)), []byte(strconv.Itoa(i)))
	}
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		NewDiskStorage(filename)
	}
}

// load all key dir value from hint files
func BenchmarkNewDiskStorage_from_hintFiles(b *testing.B) {
	store, filename, cleanupFunc := initStorageHelper()
	defer cleanupFunc()

	for i := 0; i < 100_000; i++ {
		store.Set([]byte(strconv.Itoa(i)), []byte(strconv.Itoa(i)))
	}
	store.Close()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		NewDiskStorage(filename)
	}
}

func equalByte(t *testing.T, expected, actual []byte) {
	assert.Equal(t, string(expected), string(actual))
}
