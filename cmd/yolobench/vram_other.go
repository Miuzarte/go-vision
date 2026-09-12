//go:build !windows

package main

// burnVRAM 非 Windows 平台不支持显存占用
func burnVRAM(mb int) (func(), error) {
	return func() {}, nil
}
