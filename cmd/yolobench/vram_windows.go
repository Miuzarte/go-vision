//go:build windows

package main

import (
	"fmt"
	"syscall"
	"unsafe"
)

// burnVRAM 用 CUDA runtime 分配并占住 mb 兆显存
//
// 真机上只在彩六这种吃显存的游戏里复现过, 所以这里主动制造显存压力
// 走 cudart (cudaMalloc) 而不是 driver API, 不需要手动建 context
func burnVRAM(mb int) (func(), error) {
	if mb <= 0 {
		return func() {}, nil
	}

	dll := syscall.NewLazyDLL("cudart64_13.dll")
	if err := dll.Load(); err != nil {
		return nil, fmt.Errorf("加载 cudart64_13.dll 失败: %w", err)
	}
	malloc := dll.NewProc("cudaMalloc")
	free := dll.NewProc("cudaFree")

	var ptr unsafe.Pointer
	size := uintptr(mb) * 1024 * 1024
	ret, _, _ := malloc.Call(uintptr(unsafe.Pointer(&ptr)), size)
	if ret != 0 {
		// 2 = cudaErrorMemoryAllocation, 3 = cudaErrorInitializationError
		return nil, fmt.Errorf("cudaMalloc(%dMiB) 失败: cudaError %d", mb, int32(ret))
	}

	fmt.Printf("burned %dMiB VRAM at %p\n", mb, ptr)
	return func() { free.Call(uintptr(ptr)) }, nil
}
