package yolo26

import (
	"fmt"
	"image"
	"sync"
	"unsafe"

	"github.com/getcharzp/go-vision"
	ort "github.com/getcharzp/onnxruntime_purego"
	"github.com/up-zero/gotool/convertutil"
)

// DetEngine YOLO26-det Engine
type DetEngine struct {
	session     *ort.Session
	config      Config
	cudaMemInfo ort.MemoryInfoHandle

	// 常驻推理资源: 输入 / 输出张量与 IoBinding 只创建一次并复用
	//
	// 早期实现每帧新建 IoBinding 并绑定新张量, 每帧都会在设备侧分配 / 释放一块输入缓冲
	// 在显存紧张时 (游戏占用大量显存) 这种抖动正是触发 CUDA 粘性错误 (error 700) 的温床,
	// 复用同一 binding 后每帧只剩一次 H2D 拷贝
	mu      sync.Mutex
	input   []float32
	inVal   *ort.Value
	outData []float32
	outVal  *ort.Value
	binding *ort.IoBinding

	// UseDeviceBuffers 时: 张量直接建在自己的显存上, 每帧只做 H2D / D2H 拷贝
	inDev  unsafe.Pointer
	outDev unsafe.Pointer
}

// NewDetEngine 初始化检测引擎
func NewDetEngine(cfg Config) (*DetEngine, error) {
	oc := new(vision.OnnxConfig)
	_ = convertutil.CopyProperties(cfg, oc)
	oc.TensorRTOptions = cfg.TensorRTOptions

	if err := oc.New(); err != nil {
		return nil, fmt.Errorf("初始化失败: %w", err)
	}

	session, err := oc.OnnxEngine.NewSession(cfg.ModelPath, oc.SessionOptions)
	if err != nil {
		return nil, fmt.Errorf("创建 ONNX 会话失败: %w", err)
	}

	cudaMemInfo, err := ort.CreateCudaMemoryInfo(0)
	if err != nil {
		return nil, fmt.Errorf("创建 CUDA 内存信息失败: %w", err)
	}

	engine := &DetEngine{
		session:     session,
		config:      cfg,
		cudaMemInfo: cudaMemInfo,
	}

	// 零拷贝路径提前建好设备缓冲, 让 cudart 缺失 / 显存不足在初始化阶段就报错,
	// 而不是每帧在 Predict 里失败重试
	if cfg.UseDeviceBuffers {
		if err := engine.ensureBound(); err != nil {
			engine.Destroy()
			return nil, fmt.Errorf("初始化设备张量失败: %w", err)
		}
	}

	return engine, nil
}

// OrtVersion 返回底层 ONNX Runtime 版本
func (e *DetEngine) OrtVersion() string {
	if e.session == nil {
		return ""
	}
	return e.session.Version()
}

// Destroy 释放相关资源
//
// CUDA context 已损坏 (粘性错误) 时不要调用, 否则 EP 析构里会再次触发非法访问并崩溃
func (e *DetEngine) Destroy() {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.binding != nil {
		e.binding.Release()
		e.binding = nil
	}
	if e.outVal != nil {
		e.outVal.Destroy()
		e.outVal = nil
	}
	if e.inVal != nil {
		e.inVal.Destroy()
		e.inVal = nil
	}
	e.outData = nil
	e.input = nil

	if e.inDev != nil {
		devFree(e.inDev)
		e.inDev = nil
	}
	if e.outDev != nil {
		devFree(e.outDev)
		e.outDev = nil
	}

	if e.cudaMemInfo != 0 {
		ort.ReleaseMemoryInfo(e.cudaMemInfo)
		e.cudaMemInfo = 0
	}
	if e.session != nil {
		e.session.Destroy()
		e.session = nil
	}
}

// ensureBound 懒初始化常驻张量与 IoBinding
func (e *DetEngine) ensureBound() error {
	if e.binding != nil {
		return nil
	}

	size := e.config.InputSize
	if size <= 0 {
		size = 640
	}

	e.input = make([]float32, 3*size*size)
	e.outData = make([]float32, 1*300*6)

	var inVal, outVal *ort.Value
	var err error
	if e.config.UseDeviceBuffers {
		inVal, outVal, err = e.bindDeviceTensors(size)
	} else {
		inVal, outVal, err = e.bindHostTensors(size)
	}
	if err != nil {
		return err
	}

	binding, err := e.session.CreateIoBinding()
	if err != nil {
		inVal.Destroy()
		outVal.Destroy()
		return fmt.Errorf("创建 IoBinding 失败: %w", err)
	}
	if err := binding.BindInput("images", inVal); err != nil {
		binding.Release()
		inVal.Destroy()
		outVal.Destroy()
		return fmt.Errorf("绑定输入失败: %w", err)
	}
	if err := binding.BindOutput("output0", outVal); err != nil {
		binding.Release()
		inVal.Destroy()
		outVal.Destroy()
		return fmt.Errorf("绑定输出失败: %w", err)
	}

	e.inVal = inVal
	e.outVal = outVal
	e.binding = binding
	return nil
}

// bindHostTensors 张量建在 host 内存上, 每帧由 EP 负责 H2D / D2H
func (e *DetEngine) bindHostTensors(size int) (*ort.Value, *ort.Value, error) {
	inVal, err := ort.NewTensor([]int64{1, 3, int64(size), int64(size)}, e.input)
	if err != nil {
		return nil, nil, fmt.Errorf("创建输入张量失败: %w", err)
	}

	outVal, err := ort.NewTensor([]int64{1, 300, 6}, e.outData)
	if err != nil {
		inVal.Destroy()
		return nil, nil, fmt.Errorf("创建输出张量失败: %w", err)
	}
	return inVal, outVal, nil
}

// bindDeviceTensors 张量建在自分配的显存上 (零拷贝 I/O 绑定)
//
// EP 不再为每次推理分配 / 拷贝设备缓冲, 我们只做一次 cudaMemcpy
func (e *DetEngine) bindDeviceTensors(size int) (*ort.Value, *ort.Value, error) {
	inBytes := len(e.input) * 4
	inDev, err := devAlloc(inBytes)
	if err != nil {
		return nil, nil, fmt.Errorf("分配输入显存失败: %w", err)
	}

	outBytes := len(e.outData) * 4
	outDev, err := devAlloc(outBytes)
	if err != nil {
		devFree(inDev)
		return nil, nil, fmt.Errorf("分配输出显存失败: %w", err)
	}

	inVal, err := ort.NewTensorFromPtr(
		[]int64{1, 3, int64(size), int64(size)},
		ort.TensorElementDataTypeFloat, inDev, len(e.input), e.cudaMemInfo,
	)
	if err != nil {
		devFree(inDev)
		devFree(outDev)
		return nil, nil, fmt.Errorf("创建输入张量失败: %w", err)
	}

	outVal, err := ort.NewTensorFromPtr(
		[]int64{1, 300, 6},
		ort.TensorElementDataTypeFloat, outDev, len(e.outData), e.cudaMemInfo,
	)
	if err != nil {
		inVal.Destroy()
		devFree(inDev)
		devFree(outDev)
		return nil, nil, fmt.Errorf("创建输出张量失败: %w", err)
	}

	e.inDev = inDev
	e.outDev = outDev
	return inVal, outVal, nil
}

// Predict 执行检测推理
func (e *DetEngine) Predict(img image.Image) ([]DetResult, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if err := e.ensureBound(); err != nil {
		return nil, err
	}

	// 预处理写入常驻输入缓冲区
	params := preprocessInto(img, e.config.InputSize, e.input)

	if e.config.UseDeviceBuffers {
		// 零拷贝 I/O: 绑定不用动, 只需要把 host 输入拷到自己的输入显存
		if err := devCopy(e.inDev, unsafe.Pointer(&e.input[0]), len(e.input)*4, cudaMemcpyHostToDevice); err != nil {
			return nil, fmt.Errorf("拷贝输入到显存失败: %w", err)
		}
	} else {
		// 注意: 实测 (yolobench -probe) 只有 BindInput 才会把 host 输入真的拷到设备侧,
		// 复用绑定再调 SynchronizeBoundInputs 不会刷新设备数据, 所以每帧重新绑定
		if err := e.binding.BindInput("images", e.inVal); err != nil {
			return nil, fmt.Errorf("绑定输入失败: %w", err)
		}
		if err := e.binding.BindOutput("output0", e.outVal); err != nil {
			return nil, fmt.Errorf("绑定输出失败: %w", err)
		}

		if err := e.binding.SynchronizeBoundInputs(); err != nil {
			return nil, fmt.Errorf("同步输入失败: %w", err)
		}
	}

	if err := e.binding.RunWithBinding(0); err != nil {
		return nil, fmt.Errorf("推理失败: %w", err)
	}

	if err := e.binding.SynchronizeBoundOutputs(); err != nil {
		return nil, fmt.Errorf("同步输出失败: %w", err)
	}

	if e.config.UseDeviceBuffers {
		if err := devCopy(unsafe.Pointer(&e.outData[0]), e.outDev, len(e.outData)*4, cudaMemcpyDeviceToHost); err != nil {
			return nil, fmt.Errorf("拷贝输出回主机失败: %w", err)
		}
	}

	return e.postprocess(e.outData, params), nil
}

// LastOutput 返回最近一次推理的原始输出缓冲区 (shape [1, 300, 6])
//
// 只读, 且会被下一次 Predict 覆盖, 仅用于诊断: 例如校验常驻 IoBinding 复用时
// 输入是否真的被重新拷到了设备侧 (输入变了输出必须变)
func (e *DetEngine) LastOutput() []float32 {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.outData
}

// postprocess 后处理，输出结果解析
func (e *DetEngine) postprocess(data []float32, params imageParams) []DetResult {
	results := make([]DetResult, 0)

	const stride = 6
	numDetections := len(data) / stride

	for i := 0; i < numDetections; i++ {
		offset := i * stride

		// [x1, y1, x2, y2, score, class_id]
		x1 := data[offset+0]
		y1 := data[offset+1]
		x2 := data[offset+2]
		y2 := data[offset+3]
		score := data[offset+4]
		classID := int(data[offset+5])

		if score < e.config.ConfThreshold {
			continue
		}

		// 转换回原图坐标
		origX1 := max(0, int(x1/params.scale))
		origY1 := max(0, int(y1/params.scale))
		origX2 := min(params.origW, int(x2/params.scale))
		origY2 := min(params.origH, int(y2/params.scale))

		results = append(results, DetResult{
			ClassID: classID,
			Score:   score,
			Box:     image.Rect(origX1, origY1, origX2, origY2),
		})
	}

	return results
}
