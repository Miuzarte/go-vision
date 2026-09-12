//go:build !windows

package yolo26

import (
	"fmt"
	"unsafe"
)

// 非 Windows 平台暂不支持设备显存直连

func devAlloc(bytes int) (unsafe.Pointer, error) {
	return nil, fmt.Errorf("device buffers are only supported on windows")
}

func devFree(ptr unsafe.Pointer) {}

func devCopy(dst, src unsafe.Pointer, bytes int, kind uintptr) error {
	return fmt.Errorf("device buffers are only supported on windows")
}
