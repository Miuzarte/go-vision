//go:build windows

package yolo26

import (
	"fmt"
	"sync"
	"syscall"
	"unsafe"
)

// 设备显存辅助: 零拷贝 I/O 绑定需要我们自己分配设备缓冲并做 H2D / D2H 拷贝
//
// 走 CUDA runtime (cudart), 不引入额外依赖, DLL 名按 CUDA 大版本尝试
var (
	cudartOnce sync.Once
	cudartProc struct {
		malloc uintptr
		free   uintptr
		memcpy uintptr
	}
	cudartErr error
)

const (
	cudaMemcpyHostToDevice = 1
	cudaMemcpyDeviceToHost = 2
)

func cudartLoad() error {
	cudartOnce.Do(func() {
		for _, name := range []string{"cudart64_13.dll", "cudart64_12.dll"} {
			dll := syscall.NewLazyDLL(name)
			if err := dll.Load(); err != nil {
				continue
			}
			cudartProc.malloc = dll.NewProc("cudaMalloc").Addr()
			cudartProc.free = dll.NewProc("cudaFree").Addr()
			cudartProc.memcpy = dll.NewProc("cudaMemcpy").Addr()
			return
		}
		cudartErr = fmt.Errorf("加载 cudart64_13.dll / cudart64_12.dll 失败")
	})
	return cudartErr
}

// devAlloc 分配设备显存
func devAlloc(bytes int) (unsafe.Pointer, error) {
	if err := cudartLoad(); err != nil {
		return nil, err
	}
	var ptr unsafe.Pointer
	ret, _, _ := syscall.SyscallN(cudartProc.malloc, uintptr(unsafe.Pointer(&ptr)), uintptr(bytes))
	if ret != 0 {
		return nil, fmt.Errorf("cudaMalloc(%d) 失败: cudaError %d", bytes, int32(ret))
	}
	return ptr, nil
}

// devFree 释放设备显存
func devFree(ptr unsafe.Pointer) {
	if ptr == nil {
		return
	}
	if err := cudartLoad(); err != nil {
		return
	}
	syscall.SyscallN(cudartProc.free, uintptr(ptr))
}

// devCopy 设备与主机之间拷贝
func devCopy(dst, src unsafe.Pointer, bytes int, kind uintptr) error {
	if err := cudartLoad(); err != nil {
		return err
	}
	ret, _, _ := syscall.SyscallN(cudartProc.memcpy, uintptr(dst), uintptr(src), uintptr(bytes), kind)
	if ret != 0 {
		return fmt.Errorf("cudaMemcpy 失败: cudaError %d", int32(ret))
	}
	return nil
}
