// yolobench 推理复现 / 对照 harness
//
// 固定输入满速循环推理, 用来快速判定 CUDA 非法访问 (error 700) 出在哪一层:
//   - -ep trt / cuda / cpu 切换执行提供者
//   - -opts "k=v,k=v" 改 EP provider options (例如关掉 nv_use_sync_gpu_allocator 做对照)
//   - -burn-vram 占住显存, 模拟游戏占用大量显存的场景 (真机上只在彩六这种吃显存的游戏里复现)
//   - -anim 每帧改动输入, 校验输入确实被重新拷到设备侧 (复用常驻 IoBinding 时最容易出错的地方)
//
// 用法示例:
//
//	go run ./cmd/yolobench -model B:\Git\go-vision\_weights\yolo26_weights\yolo26n.onnx \
//	    -ort B:\Git\GoCVStreamer\libs\onnxruntime-win-x64-gpu_cuda13-1.28.0\lib\onnxruntime.dll
package main

import (
	"flag"
	"fmt"
	"hash/fnv"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/getcharzp/go-vision/yolo26"
	ort "github.com/getcharzp/onnxruntime_purego"
)

var (
	modelPath  = flag.String("model", "", "ONNX 模型路径 (默认 $YOLO_MODEL_PATH 或 ./yolo26_weights/yolo26n.onnx)")
	ortLib     = flag.String("ort", "", "onnxruntime.dll 路径 (默认 $ORT_LIB_PATH 或 ./lib/onnxruntime.dll)")
	pluginPath = flag.String("plugin", "", "NvTensorRTRTX EP 插件 DLL 路径 (默认 $TENSOR_RT_EP_ABI_PATH\\onnxruntime_providers_nv_tensorrt_rtx.dll)")
	ep         = flag.String("ep", "trt", "执行提供者: trt, cuda, cpu")
	optsFlag   = flag.String("opts", "", "额外的 EP provider options, 形如 k=v,k=v")
	syncAlloc  = flag.Bool("sync-alloc", true, "trt 时设置 nv_use_sync_gpu_allocator=1 (关掉 cudaMallocAsync 异步显存池)")

	size       = flag.Int("size", 640, "模型输入边长")
	conf       = flag.Float64("conf", 0.45, "置信度阈值")
	deviceIO   = flag.Bool("device-io", false, "输入/输出张量用自分配显存做零拷贝 I/O 绑定")
	imgSize    = flag.Int("img-size", 1280, "合成图边长 (真实场景是屏幕中心裁剪)")
	imgPath    = flag.String("img", "", "可选输入图片 (png/jpeg), 缺省用合成图")
	anim       = flag.Bool("anim", true, "每帧改动输入图, 用于校验输入真的被拷到设备侧")
	iterations = flag.Int("iter", 0, "总迭代次数 (0 = 一直跑)")
	fps        = flag.Int("fps", 0, "限速 (0 = 满速)")
	warmup     = flag.Int("warmup", 20, "预热次数")
	report     = flag.Int("report", 200, "每多少帧打印一次进度")
	burnVram   = flag.Int("burn-vram", 0, "预先占住的显存 MiB (模拟游戏占用)")
	verbose    = flag.Bool("v", false, "打印每帧耗时")
	probe      = flag.Bool("probe", false, "只做输入传播自检: 用几张差异极大的图各跑一次并对比原始输出")
)

type stats struct {
	latencies []time.Duration
	errors    int
	firstErr  error
	firstErrN int
	fatal     bool
}

func (s *stats) observe(d time.Duration) {
	s.latencies = append(s.latencies, d)
}

func (s *stats) summary() string {
	if len(s.latencies) == 0 {
		return "no samples"
	}
	sorted := append([]time.Duration(nil), s.latencies...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	var sum time.Duration
	for _, d := range sorted {
		sum += d
	}
	avg := sum / time.Duration(len(sorted))
	p50 := sorted[len(sorted)*50/100]
	p99 := sorted[min(len(sorted)-1, len(sorted)*99/100)]
	return fmt.Sprintf("avg=%.2fms p50=%.2fms p99=%.2fms min=%.2fms max=%.2fms",
		ms(avg), ms(p50), ms(p99), ms(sorted[0]), ms(sorted[len(sorted)-1]))
}

func ms(d time.Duration) float64 {
	return float64(d.Microseconds()) / 1000
}

func main() {
	flag.Parse()

	cfg := yolo26.Config{
		ModelPath:          firstNonEmpty(*modelPath, os.Getenv("YOLO_MODEL_PATH"), filepath.FromSlash("./yolo26_weights/yolo26n.onnx")),
		OnnxRuntimeLibPath: firstNonEmpty(*ortLib, os.Getenv("ORT_LIB_PATH"), filepath.FromSlash("./lib/onnxruntime.dll")),
		ConfThreshold:      float32(*conf),
		InputSize:          *size,
		UseDeviceBuffers:   *deviceIO,
	}
	var optDesc []string
	switch *ep {
	case "trt":
		cfg.UseTensorRT = true
		cfg.TensorRTPluginPath = firstNonEmpty(*pluginPath, pluginFromEnv())
		if cfg.TensorRTPluginPath == "" {
			fatal("请用 -plugin 指定 NvTensorRTRTX EP 插件 DLL 路径 (或设置 TENSOR_RT_EP_ABI_PATH)")
		}
		opts := map[string]string{}
		if *syncAlloc {
			opts["nv_use_sync_gpu_allocator"] = "1"
		}
		for k, v := range parseOpts(*optsFlag) {
			opts[k] = v
		}
		// 引擎缓存默认丢到临时目录, 避免在仓库里留 trt_cache
		if _, ok := opts["nv_runtime_cache_path"]; !ok {
			opts["nv_runtime_cache_path"] = filepath.Join(os.TempDir(), "yolobench_trt_cache")
		}
		cfg.TensorRTOptions = opts
		for k, v := range opts {
			optDesc = append(optDesc, k+"="+v)
		}
		sort.Strings(optDesc)
	case "cuda":
		cfg.UseCuda = true
	case "cpu":
	default:
		fatal("未知 -ep: %s (可选 trt / cuda / cpu)", *ep)
	}

	img, err := loadImage(*imgPath, *imgSize)
	if err != nil {
		fatal("加载图片失败: %v", err)
	}

	release, err := burnVRAM(*burnVram)
	if err != nil {
		fatal("占显存失败: %v", err)
	}
	defer release()

	fmt.Printf("model=%s\nep=%s opts=%s input=%d image=%v anim=%v deviceIO=%v burnVram=%dMiB limit=%dfps\n",
		cfg.ModelPath, *ep, strings.Join(optDesc, ","), cfg.InputSize,
		img.Bounds().Size(), *anim, *deviceIO, *burnVram, *fps)

	eng, err := yolo26.NewDetEngine(cfg)
	if err != nil {
		fatal("初始化失败: %v", err)
	}
	defer eng.Destroy()

	fmt.Printf("ortVersion=%s\n", eng.OrtVersion())

	if *probe {
		runProbe(eng)
		return
	}

	// 预热
	for i := range *warmup {
		t0 := time.Now()
		if _, err := eng.Predict(img); err != nil {
			fatal("预热第 %d 次失败: %v", i, err)
		}
		if i == 0 {
			fmt.Printf("warmup first=%.2fms\n", ms(time.Since(t0)))
		}
	}

	st := &stats{}
	hashes := map[uint64]struct{}{}
	dets := 0
	start := time.Now()
	var frames int

	interval := time.Duration(0)
	if *fps > 0 {
		interval = time.Second / time.Duration(*fps)
	}

	for n := 1; *iterations == 0 || n <= *iterations; n++ {
		frames = n
		if *anim {
			animate(img, n)
		}

		iterStart := time.Now()
		results, err := eng.Predict(img)
		st.observe(time.Since(iterStart))
		if err != nil {
			st.errors++
			if st.firstErr == nil {
				st.firstErr, st.firstErrN = err, n
			}
			fmt.Printf("[%d] ERROR: %v\n", n, err)
			if ort.IsFatalError(err) {
				st.fatal = true
				break
			}
			if st.errors > 20 {
				fmt.Printf("连续错误过多, 停止\n")
				break
			}
			time.Sleep(time.Millisecond * 100)
			continue
		}
		dets = len(results)
		hashes[hashFloats(eng.LastOutput())] = struct{}{}

		if *verbose {
			fmt.Printf("[%d] %.2fms dets=%d\n", n, ms(st.latencies[len(st.latencies)-1]), dets)
		}
		if *report > 0 && n%*report == 0 {
			elapsed := time.Since(start)
			fmt.Printf("[%d] fps=%.1f %s errors=%d distinctOut=%d dets=%d\n",
				n, float64(n)/elapsed.Seconds(), st.summary(), st.errors, len(hashes), dets)
		}
		if interval > 0 {
			time.Sleep(interval)
		}
	}

	elapsed := time.Since(start)
	fmt.Printf("\ndone: iter=%d elapsed=%.1fs fps=%.1f %s errors=%d\n",
		frames, elapsed.Seconds(), float64(frames)/elapsed.Seconds(), st.summary(), st.errors)
	if st.firstErr != nil {
		fmt.Printf("first error at iter=%d: %v\n", st.firstErrN, st.firstErr)
	}
	if *anim {
		// 输入每帧都在变, 输出就那么几种说明输入根本没进到设备侧 (常驻 IoBinding 复用写错了)
		if len(hashes) <= 1 {
			fmt.Printf("WARN: 输入每帧都变但输出只有 %d 种, 疑似输入没有被拷到设备侧\n", len(hashes))
		} else {
			fmt.Printf("input propagation ok: distinctOut=%d\n", len(hashes))
		}
	}
	if st.fatal {
		fmt.Printf("FATAL: CUDA 粘性错误, context 已报废\n")
		os.Exit(2)
	}
}

// runProbe 输入传播自检
//
// 用差异极大的几张图各推理一次, 对比原始输出: 输出必须随输入变化
// 复用常驻 IoBinding 时, 如果 SynchronizeBoundInputs 没把 host 输入推到设备侧,
// 这里会看到所有图输出完全一致
func runProbe(eng *yolo26.DetEngine) {
	check := func(name string, img *image.RGBA) {
		if _, err := eng.Predict(img); err != nil {
			fmt.Printf("probe %-8s ERROR: %v\n", name, err)
			return
		}
		out := eng.LastOutput()
		nonzero := 0
		for _, v := range out {
			if v != 0 {
				nonzero++
			}
		}
		head := out
		if len(head) > 6 {
			head = head[:6]
		}
		fmt.Printf("probe %-8s hash=%016x nonzero=%d/%d head=%v\n",
			name, hashFloats(out), nonzero, len(out), head)
	}

	size := *imgSize
	black := image.NewRGBA(image.Rect(0, 0, size, size))
	white := image.NewRGBA(image.Rect(0, 0, size, size))
	draw.Draw(white, white.Bounds(), &image.Uniform{C: color.RGBA{R: 255, G: 255, B: 255, A: 255}}, image.Point{}, draw.Src)

	check("black", black)
	check("white", white)
	check("gradient", syntheticImage(size))
	check("black2", black)
}

func pluginFromEnv() string {
	dir := os.Getenv("TENSOR_RT_EP_ABI_PATH")
	if dir == "" {
		return ""
	}
	return filepath.Join(dir, "onnxruntime_providers_nv_tensorrt_rtx.dll")
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

func parseOpts(s string) map[string]string {
	out := map[string]string{}
	for _, kv := range strings.Split(s, ",") {
		kv = strings.TrimSpace(kv)
		if kv == "" {
			continue
		}
		k, v, ok := strings.Cut(kv, "=")
		if !ok {
			fatal("选项格式错误: %q (应为 k=v)", kv)
		}
		out[strings.TrimSpace(k)] = strings.TrimSpace(v)
	}
	return out
}

// loadImage 读取图片, 缺省合成一张带渐变的方图
func loadImage(path string, size int) (*image.RGBA, error) {
	if path == "" {
		return syntheticImage(size), nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var src image.Image
	if strings.EqualFold(filepath.Ext(path), ".png") {
		src, err = png.Decode(f)
	} else {
		src, err = jpeg.Decode(f)
	}
	if err != nil {
		return nil, err
	}

	dst := image.NewRGBA(image.Rect(0, 0, src.Bounds().Dx(), src.Bounds().Dy()))
	draw.Draw(dst, dst.Bounds(), src, src.Bounds().Min, draw.Src)
	return dst, nil
}

func syntheticImage(size int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, size, size))
	for y := range size {
		for x := range size {
			img.SetRGBA(x, y, color.RGBA{
				R: uint8(x * 255 / size),
				G: uint8(y * 255 / size),
				B: uint8((x + y) * 255 / (2 * size)),
				A: 255,
			})
		}
	}
	// 几个固定色块, 让画面有点结构
	draw.Draw(img, image.Rect(size/8, size/8, size/3, size/3), &image.Uniform{C: color.RGBA{R: 220, A: 255}}, image.Point{}, draw.Src)
	draw.Draw(img, image.Rect(size/2, size/3, size*3/4, size*2/3), &image.Uniform{C: color.RGBA{G: 220, A: 255}}, image.Point{}, draw.Src)
	return img
}

// animate 把输入图改一改, 保证每帧输入都不同
func animate(img *image.RGBA, frame int) {
	b := img.Bounds()
	w := b.Dx() / 8
	h := b.Dy() / 8
	x := (frame * 17) % (b.Dx() - w)
	y := (frame * 29) % (b.Dy() - h)
	c := color.RGBA{R: uint8(frame * 7), G: uint8(frame * 13), B: uint8(frame * 3), A: 255}
	draw.Draw(img, image.Rect(x, y, x+w, y+h), &image.Uniform{C: c}, image.Point{}, draw.Src)
}

func hashFloats(data []float32) uint64 {
	h := fnv.New64a()
	var buf [4]byte
	for _, v := range data {
		bits := uint32(int32(v * 1000))
		buf[0] = byte(bits)
		buf[1] = byte(bits >> 8)
		buf[2] = byte(bits >> 16)
		buf[3] = byte(bits >> 24)
		h.Write(buf[:])
	}
	return h.Sum64()
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "yolobench: "+format+"\n", args...)
	os.Exit(1)
}
