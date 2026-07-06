//go:build windows

package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"runtime"
	"strings"
	"syscall"
	"time"
	"unsafe"
)

type endpointInfo struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	Default bool   `json:"default"`
}

type guid struct {
	Data1 uint32
	Data2 uint16
	Data3 uint16
	Data4 [8]byte
}

var (
	ole32                  = syscall.NewLazyDLL("ole32.dll")
	procCoInitializeEx     = ole32.NewProc("CoInitializeEx")
	procCoUninitialize     = ole32.NewProc("CoUninitialize")
	procCoCreateInstance   = ole32.NewProc("CoCreateInstance")
	procCoTaskMemFree      = ole32.NewProc("CoTaskMemFree")
	procPropVariantClear   = ole32.NewProc("PropVariantClear")
	clsidMMDeviceEnum      = guid{0xBCDE0395, 0xE52F, 0x467C, [8]byte{0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E}}
	iidIMMDeviceEnum       = guid{0xA95664D2, 0x9614, 0x4F35, [8]byte{0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6}}
	iidIAudioClient        = guid{0x1CB9AD4C, 0xDBFA, 0x4c32, [8]byte{0xB1, 0x78, 0xC2, 0xF5, 0x68, 0xA7, 0x03, 0xB2}}
	iidIAudioCaptureClient = guid{0xC8ADBD64, 0xE71E, 0x48a0, [8]byte{0xA4, 0xDE, 0x18, 0x5C, 0x39, 0x5C, 0xD3, 0x17}}
	pkeyDeviceFriendlyName = propertyKey{FMTID: guid{0xA45C254E, 0xDF1C, 0x4EFD, [8]byte{0x80, 0x20, 0x67, 0xD1, 0x46, 0xA8, 0x50, 0xE0}}, PID: 14}
	subtypePCM             = guid{0x00000001, 0x0000, 0x0010, [8]byte{0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}}
	subtypeIEEEFloat       = guid{0x00000003, 0x0000, 0x0010, [8]byte{0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}}
)

const (
	coinitMultithreaded        = 0x0
	rpcEChangedMode            = 0x80010106
	clsctxAll                  = 0x17
	eRender                    = 0
	eConsole                   = 0
	deviceStateActive          = 0x00000001
	stgmRead                   = 0x00000000
	vtLPWSTR                   = 31
	waveFormatPCM              = 0x0001
	waveFormatIEEEFloat        = 0x0003
	waveFormatTagExtensible    = 0xFFFE
	audclntShareModeShared     = 0
	audclntStreamFlagsLoopback = 0x00020000
	audclntBufferFlagsSilent   = 0x00000002
	audclntEBufferEmpty        = 0x88890001
	defaultBufferDuration100NS = 10000000
	pollInterval               = 5 * time.Millisecond
)

type propertyKey struct {
	FMTID guid
	PID   uint32
}

type propVariant struct {
	VT        uint16
	Reserved1 uint16
	Reserved2 uint16
	Reserved3 uint16
	Val       *uint16
	Val2      uintptr
}

type waveFormatEx struct {
	FormatTag      uint16
	Channels       uint16
	SamplesPerSec  uint32
	AvgBytesPerSec uint32
	BlockAlign     uint16
	BitsPerSample  uint16
	CbSize         uint16
}

type hresultError struct {
	hr uintptr
}

func (e hresultError) Error() string {
	switch uint32(e.hr) {
	case audclntEBufferEmpty:
		return "AUDCLNT_E_BUFFER_EMPTY"
	default:
		return fmt.Sprintf("HRESULT 0x%08X", uint32(e.hr))
	}
}

func failedHRESULT(hr uintptr) bool {
	return int32(hr) < 0
}

func recordLoopback(options recordOptions) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	uninitialize, err := initializeCOM()
	if err != nil {
		return err
	}
	defer uninitialize()

	enumerator, err := newMMDeviceEnumerator()
	if err != nil {
		return err
	}
	defer enumerator.Close()

	device, selected, err := enumerator.FindRenderDevice(options.DeviceID, options.DeviceName)
	if err != nil {
		return err
	}
	defer device.Close()

	client, mixFormatPointer, mixFormat, err := device.ActivateAudioClient()
	if err != nil {
		return fmt.Errorf("activate %q: %w", selected.Name, err)
	}
	defer client.Close()
	defer coTaskMemFree(mixFormatPointer)

	if err := client.InitializeLoopback(mixFormatPointer); err != nil {
		return fmt.Errorf("initialize loopback client for %q (%s): %w", selected.Name, formatDescription(mixFormat), err)
	}
	captureClient, err := client.CaptureClient()
	if err != nil {
		return err
	}
	defer captureClient.Close()

	converter, err := newPCM16Converter(mixFormat, options.TargetSampleRate, options.TargetChannels)
	if err != nil {
		return fmt.Errorf("prepare conversion from %s: %w", formatDescription(mixFormat), err)
	}
	writer, err := newPCM16WAVWriter(options.OutputFile, options.TargetSampleRate, options.TargetChannels)
	if err != nil {
		return fmt.Errorf("create WAV: %w", err)
	}
	writerClosed := false
	defer func() {
		if !writerClosed {
			_ = writer.Close()
		}
	}()

	if err := client.Start(); err != nil {
		return fmt.Errorf("start loopback capture: %w", err)
	}
	defer func() { _ = client.Stop() }()

	deadline := time.Now().Add(options.Duration)
	for time.Now().Before(deadline) {
		packetFrames, err := captureClient.NextPacketSize()
		if err != nil {
			return err
		}
		if packetFrames == 0 {
			sleepUntilNextPacket(deadline)
			continue
		}
		for packetFrames > 0 {
			data, frames, flags, err := captureClient.GetBuffer(mixFormat)
			if err != nil {
				if isHRESULT(err, audclntEBufferEmpty) {
					break
				}
				return err
			}
			samples, convertErr := converter.ConvertPacket(data, frames, flags&audclntBufferFlagsSilent != 0)
			releaseErr := captureClient.ReleaseBuffer(frames)
			if convertErr != nil {
				return convertErr
			}
			if releaseErr != nil {
				return releaseErr
			}
			if err := writer.WriteSamples(samples); err != nil {
				return err
			}
			packetFrames, err = captureClient.NextPacketSize()
			if err != nil {
				return err
			}
		}
	}
	if err := writer.WriteSamples(converter.Flush()); err != nil {
		return err
	}
	expectedFrames := int(options.Duration.Seconds() * float64(options.TargetSampleRate))
	if expectedFrames <= 0 {
		expectedFrames = options.TargetSampleRate / 10
	}
	if missingFrames := expectedFrames - writer.FramesWritten(); missingFrames > 0 {
		if err := writer.WriteSilence(missingFrames); err != nil {
			return err
		}
	}
	if err := writer.Close(); err != nil {
		return err
	}
	writerClosed = true
	return nil
}

func listRenderEndpoints() ([]endpointInfo, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	uninitialize, err := initializeCOM()
	if err != nil {
		return nil, err
	}
	defer uninitialize()
	enumerator, err := newMMDeviceEnumerator()
	if err != nil {
		return nil, err
	}
	defer enumerator.Close()
	return enumerator.RenderEndpoints()
}

func initializeCOM() (func(), error) {
	hr, _, _ := procCoInitializeEx.Call(0, coinitMultithreaded)
	if hr == rpcEChangedMode {
		return func() {}, nil
	}
	if failedHRESULT(hr) {
		return nil, fmt.Errorf("initialize COM: %w", hresultError{hr: hr})
	}
	return func() { procCoUninitialize.Call() }, nil
}

type mmDeviceEnumerator struct {
	ptr unsafe.Pointer
}

func newMMDeviceEnumerator() (*mmDeviceEnumerator, error) {
	var ptr unsafe.Pointer
	hr, _, _ := procCoCreateInstance.Call(
		uintptr(unsafe.Pointer(&clsidMMDeviceEnum)),
		0,
		clsctxAll,
		uintptr(unsafe.Pointer(&iidIMMDeviceEnum)),
		uintptr(unsafe.Pointer(&ptr)),
	)
	if failedHRESULT(hr) {
		return nil, fmt.Errorf("create MMDeviceEnumerator: %w", hresultError{hr: hr})
	}
	if ptr == nil {
		return nil, errors.New("create MMDeviceEnumerator: nil COM pointer")
	}
	return &mmDeviceEnumerator{ptr: ptr}, nil
}

func (e *mmDeviceEnumerator) Close() {
	releaseCOM(e.ptr)
	e.ptr = nil
}

func (e *mmDeviceEnumerator) FindRenderDevice(deviceID, deviceName string) (*mmDevice, endpointInfo, error) {
	selectors := deviceSelectors(deviceID, deviceName)
	if len(selectors) == 0 || isDefaultDeviceSelector(selectors[0]) {
		device, err := e.DefaultRenderDevice()
		if err != nil {
			return nil, endpointInfo{}, err
		}
		info, err := device.Info()
		if err != nil {
			device.Close()
			return nil, endpointInfo{}, err
		}
		info.Default = true
		return device, info, nil
	}

	if device, err := e.DeviceByID(selectors[0]); err == nil {
		info, infoErr := device.Info()
		if infoErr != nil {
			device.Close()
			return nil, endpointInfo{}, infoErr
		}
		return device, info, nil
	}

	collection, err := e.EnumRenderEndpoints()
	if err != nil {
		return nil, endpointInfo{}, err
	}
	defer collection.Close()
	count, err := collection.Count()
	if err != nil {
		return nil, endpointInfo{}, err
	}
	for i := uint32(0); i < count; i++ {
		device, err := collection.Item(i)
		if err != nil {
			return nil, endpointInfo{}, err
		}
		info, err := device.Info()
		if err != nil {
			device.Close()
			return nil, endpointInfo{}, err
		}
		if endpointMatches(info, selectors) {
			return device, info, nil
		}
		device.Close()
	}
	return nil, endpointInfo{}, fmt.Errorf("render endpoint not found for %q; run `%s list --json` to see active WASAPI render endpoints", strings.Join(selectors, " / "), helperExecutableName())
}

func (e *mmDeviceEnumerator) RenderEndpoints() ([]endpointInfo, error) {
	collection, err := e.EnumRenderEndpoints()
	if err != nil {
		return nil, err
	}
	defer collection.Close()
	count, err := collection.Count()
	if err != nil {
		return nil, err
	}
	defaultID := ""
	if defaultDevice, err := e.DefaultRenderDevice(); err == nil {
		if info, infoErr := defaultDevice.Info(); infoErr == nil {
			defaultID = info.ID
		}
		defaultDevice.Close()
	}
	endpoints := make([]endpointInfo, 0, count)
	for i := uint32(0); i < count; i++ {
		device, err := collection.Item(i)
		if err != nil {
			return nil, err
		}
		info, err := device.Info()
		device.Close()
		if err != nil {
			return nil, err
		}
		info.Default = info.ID != "" && info.ID == defaultID
		endpoints = append(endpoints, info)
	}
	return endpoints, nil
}

func (e *mmDeviceEnumerator) DefaultRenderDevice() (*mmDevice, error) {
	var ptr unsafe.Pointer
	if _, err := comCall(e.ptr, 4, eRender, eConsole, uintptr(unsafe.Pointer(&ptr))); err != nil {
		return nil, fmt.Errorf("get default render endpoint: %w", err)
	}
	return &mmDevice{ptr: ptr}, nil
}

func (e *mmDeviceEnumerator) DeviceByID(id string) (*mmDevice, error) {
	wide, err := syscall.UTF16PtrFromString(id)
	if err != nil {
		return nil, err
	}
	var ptr unsafe.Pointer
	if _, err := comCall(e.ptr, 5, uintptr(unsafe.Pointer(wide)), uintptr(unsafe.Pointer(&ptr))); err != nil {
		return nil, err
	}
	return &mmDevice{ptr: ptr}, nil
}

func (e *mmDeviceEnumerator) EnumRenderEndpoints() (*mmDeviceCollection, error) {
	var ptr unsafe.Pointer
	if _, err := comCall(e.ptr, 3, eRender, deviceStateActive, uintptr(unsafe.Pointer(&ptr))); err != nil {
		return nil, fmt.Errorf("enumerate render endpoints: %w", err)
	}
	return &mmDeviceCollection{ptr: ptr}, nil
}

type mmDeviceCollection struct {
	ptr unsafe.Pointer
}

func (c *mmDeviceCollection) Close() {
	releaseCOM(c.ptr)
	c.ptr = nil
}

func (c *mmDeviceCollection) Count() (uint32, error) {
	var count uint32
	if _, err := comCall(c.ptr, 3, uintptr(unsafe.Pointer(&count))); err != nil {
		return 0, err
	}
	return count, nil
}

func (c *mmDeviceCollection) Item(index uint32) (*mmDevice, error) {
	var ptr unsafe.Pointer
	if _, err := comCall(c.ptr, 4, uintptr(index), uintptr(unsafe.Pointer(&ptr))); err != nil {
		return nil, err
	}
	return &mmDevice{ptr: ptr}, nil
}

type mmDevice struct {
	ptr unsafe.Pointer
}

func (d *mmDevice) Close() {
	releaseCOM(d.ptr)
	d.ptr = nil
}

func (d *mmDevice) Info() (endpointInfo, error) {
	id, err := d.ID()
	if err != nil {
		return endpointInfo{}, err
	}
	name, err := d.FriendlyName()
	if err != nil || strings.TrimSpace(name) == "" {
		name = id
	}
	return endpointInfo{ID: id, Name: name}, nil
}

func (d *mmDevice) ID() (string, error) {
	var ptr *uint16
	if _, err := comCall(d.ptr, 5, uintptr(unsafe.Pointer(&ptr))); err != nil {
		return "", err
	}
	defer coTaskMemFree(ptr)
	if ptr == nil {
		return "", errors.New("endpoint ID pointer was nil")
	}
	return utf16PtrToString(ptr), nil
}

func (d *mmDevice) FriendlyName() (string, error) {
	var storePtr unsafe.Pointer
	if _, err := comCall(d.ptr, 4, stgmRead, uintptr(unsafe.Pointer(&storePtr))); err != nil {
		return "", err
	}
	defer releaseCOM(storePtr)
	var value propVariant
	if _, err := comCall(storePtr, 5, uintptr(unsafe.Pointer(&pkeyDeviceFriendlyName)), uintptr(unsafe.Pointer(&value))); err != nil {
		return "", err
	}
	defer propVariantClear(&value)
	if value.VT != vtLPWSTR || value.Val == nil {
		return "", nil
	}
	return utf16PtrToString(value.Val), nil
}

func (d *mmDevice) ActivateAudioClient() (*audioClient, unsafe.Pointer, captureFormat, error) {
	var clientPtr unsafe.Pointer
	if _, err := comCall(d.ptr, 3, uintptr(unsafe.Pointer(&iidIAudioClient)), clsctxAll, 0, uintptr(unsafe.Pointer(&clientPtr))); err != nil {
		return nil, nil, captureFormat{}, err
	}
	client := &audioClient{ptr: clientPtr}
	var formatPtr unsafe.Pointer
	if _, err := comCall(client.ptr, 8, uintptr(unsafe.Pointer(&formatPtr))); err != nil {
		client.Close()
		return nil, nil, captureFormat{}, fmt.Errorf("get mix format: %w", err)
	}
	format, err := parseMixFormat(formatPtr)
	if err != nil {
		client.Close()
		coTaskMemFree(formatPtr)
		return nil, nil, captureFormat{}, err
	}
	return client, formatPtr, format, nil
}

type audioClient struct {
	ptr unsafe.Pointer
}

func (c *audioClient) Close() {
	releaseCOM(c.ptr)
	c.ptr = nil
}

func (c *audioClient) InitializeLoopback(formatPtr unsafe.Pointer) error {
	_, err := comCall(c.ptr, 3, audclntShareModeShared, audclntStreamFlagsLoopback, defaultBufferDuration100NS, 0, uintptr(formatPtr), 0)
	return err
}

func (c *audioClient) CaptureClient() (*audioCaptureClient, error) {
	var ptr unsafe.Pointer
	if _, err := comCall(c.ptr, 14, uintptr(unsafe.Pointer(&iidIAudioCaptureClient)), uintptr(unsafe.Pointer(&ptr))); err != nil {
		return nil, fmt.Errorf("get IAudioCaptureClient: %w", err)
	}
	return &audioCaptureClient{ptr: ptr}, nil
}

func (c *audioClient) Start() error {
	_, err := comCall(c.ptr, 10)
	return err
}

func (c *audioClient) Stop() error {
	_, err := comCall(c.ptr, 11)
	return err
}

type audioCaptureClient struct {
	ptr unsafe.Pointer
}

func (c *audioCaptureClient) Close() {
	releaseCOM(c.ptr)
	c.ptr = nil
}

func (c *audioCaptureClient) NextPacketSize() (uint32, error) {
	var frames uint32
	if _, err := comCall(c.ptr, 5, uintptr(unsafe.Pointer(&frames))); err != nil {
		return 0, err
	}
	return frames, nil
}

func (c *audioCaptureClient) GetBuffer(format captureFormat) ([]byte, uint32, uint32, error) {
	var dataPtr *byte
	var frames uint32
	var flags uint32
	var devicePosition uint64
	var qpcPosition uint64
	if _, err := comCall(c.ptr, 3, uintptr(unsafe.Pointer(&dataPtr)), uintptr(unsafe.Pointer(&frames)), uintptr(unsafe.Pointer(&flags)), uintptr(unsafe.Pointer(&devicePosition)), uintptr(unsafe.Pointer(&qpcPosition))); err != nil {
		return nil, 0, 0, err
	}
	if flags&audclntBufferFlagsSilent != 0 || frames == 0 {
		return nil, frames, flags, nil
	}
	if dataPtr == nil {
		return nil, frames, flags, errors.New("capture buffer pointer was nil")
	}
	byteCount := int(frames) * format.BlockAlign
	data := unsafe.Slice(dataPtr, byteCount)
	copyData := append([]byte(nil), data...)
	return copyData, frames, flags, nil
}

func (c *audioCaptureClient) ReleaseBuffer(frames uint32) error {
	_, err := comCall(c.ptr, 4, uintptr(frames))
	return err
}

func parseMixFormat(ptr unsafe.Pointer) (captureFormat, error) {
	if ptr == nil {
		return captureFormat{}, errors.New("mix format pointer was nil")
	}
	format := (*waveFormatEx)(unsafe.Pointer(ptr))
	result := captureFormat{
		SampleRate:    int(format.SamplesPerSec),
		Channels:      int(format.Channels),
		BitsPerSample: int(format.BitsPerSample),
		BlockAlign:    int(format.BlockAlign),
	}
	switch format.FormatTag {
	case waveFormatPCM:
		result.SampleKind = captureSamplePCM
	case waveFormatIEEEFloat:
		result.SampleKind = captureSampleFloat
	case waveFormatTagExtensible:
		if format.CbSize < 22 {
			return captureFormat{}, fmt.Errorf("WAVE_FORMAT_EXTENSIBLE fmt chunk is too small: cbSize=%d", format.CbSize)
		}
		formatBytes := unsafe.Slice((*byte)(ptr), 18+int(format.CbSize))
		subFormat := guidFromBytes(formatBytes[24:40])
		switch subFormat {
		case subtypePCM:
			result.SampleKind = captureSamplePCM
		case subtypeIEEEFloat:
			result.SampleKind = captureSampleFloat
		default:
			return captureFormat{}, fmt.Errorf("unsupported WAVE_FORMAT_EXTENSIBLE subformat %+v", subFormat)
		}
	default:
		return captureFormat{}, fmt.Errorf("unsupported WASAPI mix format tag 0x%X", format.FormatTag)
	}
	if result.SampleKind == captureSampleFloat {
		result.BitsPerSample = int(format.BitsPerSample)
	}
	if err := validateCaptureFormat(result); err != nil {
		return captureFormat{}, err
	}
	return result, nil
}

func guidFromBytes(data []byte) guid {
	var value guid
	value.Data1 = binary.LittleEndian.Uint32(data[0:4])
	value.Data2 = binary.LittleEndian.Uint16(data[4:6])
	value.Data3 = binary.LittleEndian.Uint16(data[6:8])
	copy(value.Data4[:], data[8:16])
	return value
}

func formatDescription(format captureFormat) string {
	kind := "PCM"
	if format.SampleKind == captureSampleFloat {
		kind = "float"
	}
	return fmt.Sprintf("%dHz/%dch/%s%d", format.SampleRate, format.Channels, kind, format.BitsPerSample)
}

func deviceSelectors(deviceID, deviceName string) []string {
	candidates := []string{
		strings.TrimSpace(deviceID),
		stripLoopbackDisplaySuffix(deviceID),
		strings.TrimSpace(deviceName),
		stripLoopbackDisplaySuffix(deviceName),
	}
	var selectors []string
	seen := map[string]struct{}{}
	for _, candidate := range candidates {
		candidate = strings.TrimSpace(candidate)
		if candidate == "" {
			continue
		}
		key := strings.ToLower(candidate)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		selectors = append(selectors, candidate)
	}
	return selectors
}

func endpointMatches(info endpointInfo, selectors []string) bool {
	values := []string{info.ID, info.Name, loopbackDisplayName(info.Name)}
	for _, selector := range selectors {
		for _, value := range values {
			if value == selector || strings.EqualFold(value, selector) || normalizeEndpointName(value) == normalizeEndpointName(selector) {
				return true
			}
		}
	}
	return false
}

func normalizeEndpointName(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	replacer := strings.NewReplacer("_", " ", "-", " ", ".", " ", ":", " ", "\t", " ")
	value = replacer.Replace(value)
	return strings.Join(strings.Fields(value), " ")
}

func utf16PtrToString(ptr *uint16) string {
	if ptr == nil {
		return ""
	}
	var values []uint16
	for p := unsafe.Pointer(ptr); ; p = unsafe.Add(p, unsafe.Sizeof(uint16(0))) {
		value := *(*uint16)(p)
		if value == 0 {
			break
		}
		values = append(values, value)
	}
	return syscall.UTF16ToString(values)
}

func helperExecutableName() string {
	return "wasapi-loopback-recorder"
}

func sleepUntilNextPacket(deadline time.Time) {
	remaining := time.Until(deadline)
	if remaining <= 0 {
		return
	}
	if remaining < pollInterval {
		time.Sleep(remaining)
		return
	}
	time.Sleep(pollInterval)
}

func isHRESULT(err error, hr uint32) bool {
	var h hresultError
	return errors.As(err, &h) && uint32(h.hr) == hr
}

func comCall(obj unsafe.Pointer, index int, args ...uintptr) (uintptr, error) {
	r1 := comCallRaw(obj, index, args...)
	if failedHRESULT(r1) {
		return r1, hresultError{hr: r1}
	}
	return r1, nil
}

func comCallRaw(obj unsafe.Pointer, index int, args ...uintptr) uintptr {
	if obj == nil {
		return uintptr(0x80004003) // E_POINTER
	}
	vtbl := *(*unsafe.Pointer)(obj)
	method := *(*uintptr)(unsafe.Add(vtbl, uintptr(index)*unsafe.Sizeof(uintptr(0))))
	callArgs := make([]uintptr, 0, len(args)+1)
	callArgs = append(callArgs, uintptr(obj))
	callArgs = append(callArgs, args...)
	r1, _, _ := syscall.SyscallN(method, callArgs...)
	return r1
}

func releaseCOM(ptr unsafe.Pointer) {
	if ptr != nil {
		_ = comCallRaw(ptr, 2)
	}
}

func coTaskMemFree(ptr any) {
	switch value := ptr.(type) {
	case unsafe.Pointer:
		if value != nil {
			procCoTaskMemFree.Call(uintptr(value))
		}
	case *uint16:
		if value != nil {
			procCoTaskMemFree.Call(uintptr(unsafe.Pointer(value)))
		}
	}
}

func propVariantClear(value *propVariant) {
	if value != nil {
		procPropVariantClear.Call(uintptr(unsafe.Pointer(value)))
	}
}
