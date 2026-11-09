// Record the audio of one macOS process (default: QQ Music) with a Core Audio process tap.
//
// Read-only relative to the source app: it taps the process mixdown and writes raw
// interleaved Float32 PCM to a file; it does not mute the source by default and it does
// not output any audio itself.
//
// Build (once):
//   swiftc -O tools/mac/qqmusic_tap_record.swift -o /tmp/qqmusic_tap_record \
//       -framework CoreAudio -framework AVFoundation -framework AppKit
//
// Run (normally driven by tools/capture_qqmusic.py):
//   OUT=/tmp/qq.raw SECONDS=45 WAIT_AUDIO_SECONDS=180 /tmp/qqmusic_tap_record
//
// Environment:
//   OUT                 raw output file (required)
//   SECONDS             seconds of audio to record after audio is detected (default 45)
//   WAIT_AUDIO_SECONDS  how long to wait for the target process to actually output audio
//                       (default 180; QQ Music does not start a tap aggregate while paused)
//   MUTE                0 = leave the source audible (default), 2 = mutedWhenTapped
//   BUNDLE              bundle id to tap (default com.tencent.QQMusicMac)
//   TARGET_PID          tap a concrete pid instead of a bundle (testing hook)
//
// Exit codes: 0 = ROOT_OK, 2 = target not running, 3 = process object not found,
// 4 = tap create failed, 5 = tap property read failed, 6 = aggregate device failed,
// 7 = tap list failed, 8 = IOProc/start failed, 12 = no audio from target in time,
// 13 = output write failed.
import AppKit
import AVFoundation
import CoreAudio
import Foundation

func fail(_ code: Int32, _ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(code)
}

let environment = ProcessInfo.processInfo.environment
guard let outPath = environment["OUT"], !outPath.isEmpty else {
    fail(13, "OUT is required")
}
let seconds = Double(environment["SECONDS"] ?? "45") ?? 45.0
let waitAudioSeconds = Double(environment["WAIT_AUDIO_SECONDS"] ?? "180") ?? 180.0
let muteRaw = Int(environment["MUTE"] ?? "0") ?? 0
let bundleID = environment["BUNDLE"] ?? "com.tencent.QQMusicMac"

// ---- resolve the target audio process -------------------------------------------------
var targetPID: pid_t
if let pidText = environment["TARGET_PID"], let pid = pid_t(pidText) {
    targetPID = pid
} else {
    guard let app = NSRunningApplication.runningApplications(withBundleIdentifier: bundleID).first else {
        fail(2, "TARGET_NOT_RUNNING bundle=\(bundleID)")
    }
    targetPID = app.processIdentifier
}
print("target_pid=\(targetPID)")

var pidValue = targetPID
var processAddress = AudioObjectPropertyAddress(
    mSelector: kAudioHardwarePropertyTranslatePIDToProcessObject,
    mScope: kAudioObjectPropertyScopeGlobal,
    mElement: kAudioObjectPropertyElementMain)
var processID = AudioObjectID(kAudioObjectUnknown)
var processSize = UInt32(MemoryLayout<AudioObjectID>.size)
var processStatus = OSStatus(kAudioHardwareUnspecifiedError)
let processDeadline = Date().addingTimeInterval(waitAudioSeconds)
repeat {
    processID = AudioObjectID(kAudioObjectUnknown)
    processSize = UInt32(MemoryLayout<AudioObjectID>.size)
    processStatus = AudioObjectGetPropertyData(
        AudioObjectID(kAudioObjectSystemObject), &processAddress,
        UInt32(MemoryLayout<pid_t>.size), &pidValue, &processSize, &processID)
    if processStatus == noErr, processID != kAudioObjectUnknown { break }
    RunLoop.current.run(until: Date().addingTimeInterval(0.1))
} while Date() < processDeadline
guard processStatus == noErr, processID != kAudioObjectUnknown else {
    fail(3, "PROCESS_OBJECT_NOT_FOUND status=\(processStatus) within \(Int(waitAudioSeconds))s")
}
print("audio_process_id=\(processID)")

// ---- create the tap -------------------------------------------------------------------
let description = CATapDescription(stereoMixdownOfProcesses: [processID])
description.name = "qqmusic-recorder"
description.isPrivate = true
description.muteBehavior = CATapMuteBehavior(rawValue: muteRaw)!
var tapID = AudioObjectID(kAudioObjectUnknown)
let tapStatus = AudioHardwareCreateProcessTap(description, &tapID)
guard tapStatus == noErr, tapID != kAudioObjectUnknown else {
    fail(4, "TAP_CREATE_FAILED status=\(tapStatus)")
}
print("tap_id=\(tapID)")

func tapUID(_ tap: AudioObjectID) -> CFString? {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioTapPropertyUID,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var value: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    guard AudioObjectGetPropertyData(tap, &address, 0, nil, &size, &value) == noErr, let value else {
        return nil
    }
    return value.takeRetainedValue()
}

func tapFormat(_ tap: AudioObjectID) -> AudioStreamBasicDescription? {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioTapPropertyFormat,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var format = AudioStreamBasicDescription()
    var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
    guard AudioObjectGetPropertyData(tap, &address, 0, nil, &size, &format) == noErr else {
        return nil
    }
    return format
}

func defaultOutputUID() -> CFString? {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultOutputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var deviceID = AudioDeviceID(kAudioObjectUnknown)
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    guard AudioObjectGetPropertyData(
        AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &deviceID) == noErr,
        deviceID != kAudioObjectUnknown else { return nil }
    var uidAddress = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyDeviceUID,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var uid: Unmanaged<CFString>?
    var uidSize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    guard AudioObjectGetPropertyData(deviceID, &uidAddress, 0, nil, &uidSize, &uid) == noErr,
          let uid else { return nil }
    return uid.takeRetainedValue()
}

guard let uid = tapUID(tapID), let format = tapFormat(tapID) else {
    AudioHardwareDestroyProcessTap(tapID)
    fail(5, "TAP_PROPERTY_READ_FAILED")
}
let outputUID = defaultOutputUID()
var aggregateDescription: [String: Any] = [
    kAudioAggregateDeviceNameKey: "qqmusic-recorder",
    kAudioAggregateDeviceUIDKey: "qqmusic-recorder-\(UUID().uuidString)",
    kAudioAggregateDeviceSubDeviceListKey: [] as CFArray,
    kAudioAggregateDeviceMasterSubDeviceKey: 0,
    kAudioAggregateDeviceIsPrivateKey: true,
    kAudioAggregateDeviceIsStackedKey: false,
]
if let outputUID {
    aggregateDescription[kAudioAggregateDeviceSubDeviceListKey] = [outputUID] as CFArray
    aggregateDescription[kAudioAggregateDeviceMasterSubDeviceKey] = outputUID
}
var aggregateID = AudioDeviceID(kAudioObjectUnknown)
let aggregateStatus = AudioHardwareCreateAggregateDevice(aggregateDescription as CFDictionary, &aggregateID)
guard aggregateStatus == noErr, aggregateID != kAudioObjectUnknown else {
    AudioHardwareDestroyProcessTap(tapID)
    fail(6, "AGGREGATE_CREATE_FAILED status=\(aggregateStatus)")
}

var tapList = [uid] as CFArray
var tapListAddress = AudioObjectPropertyAddress(
    mSelector: kAudioAggregateDevicePropertyTapList,
    mScope: kAudioObjectPropertyScopeGlobal,
    mElement: kAudioObjectPropertyElementMain)
let tapListStatus = withUnsafePointer(to: &tapList) { pointer in
    AudioObjectSetPropertyData(aggregateID, &tapListAddress, 0, nil,
                               UInt32(MemoryLayout<CFArray>.stride), pointer)
}
guard tapListStatus == noErr else {
    AudioHardwareDestroyAggregateDevice(aggregateID)
    AudioHardwareDestroyProcessTap(tapID)
    fail(7, "TAP_LIST_FAILED status=\(tapListStatus)")
}

// ---- record ---------------------------------------------------------------------------
final class Recorder {
    let lock = NSLock()
    var data = Data()
    var files: [(sampleRate: Double, channels: Int)] = []
    var peak: Float = 0
    var framesSeen = 0
    var started = false
}
let recorder = Recorder()

let ioProc: AudioDeviceIOProc = { _, _, inInputData, _, _, _, inClientData -> OSStatus in
    guard let clientData = inClientData else { return noErr }
    let state = Unmanaged<Recorder>.fromOpaque(clientData).takeUnretainedValue()
    let buffers = UnsafeMutableAudioBufferListPointer(
        UnsafeMutablePointer(mutating: inInputData))
    var local = Data()
    var localPeak: Float = 0
    var localFrames = 0
    for buffer in buffers {
        guard let data = buffer.mData, buffer.mDataByteSize > 0 else { continue }
        let count = Int(buffer.mDataByteSize)
        local.append(data.assumingMemoryBound(to: UInt8.self), count: count)
        let samples = data.assumingMemoryBound(to: Float.self)
        for index in 0..<(count / MemoryLayout<Float>.size) {
            let value = abs(samples[index])
            if value > localPeak { localPeak = value }
        }
        localFrames += count
    }
    state.lock.lock()
    state.peak = max(state.peak, localPeak)
    state.framesSeen += localFrames
    if state.started {
        state.data.append(local)
    }
    state.lock.unlock()
    return noErr
}
var ioProcID: AudioDeviceIOProcID?
guard AudioDeviceCreateIOProcID(aggregateID, ioProc,
                                Unmanaged.passUnretained(recorder).toOpaque(), &ioProcID) == noErr,
      let ioProcID else {
    AudioHardwareDestroyAggregateDevice(aggregateID)
    AudioHardwareDestroyProcessTap(tapID)
    fail(8, "IOPROC_CREATE_FAILED")
}
guard AudioDeviceStart(aggregateID, ioProcID) == noErr else {
    AudioDeviceDestroyIOProcID(aggregateID, ioProcID)
    AudioHardwareDestroyAggregateDevice(aggregateID)
    AudioHardwareDestroyProcessTap(tapID)
    fail(8, "DEVICE_START_FAILED")
}

// Wait until the target process actually emits audio (QQ Music's aggregate does not start
// producing PCM while paused; silence here would produce a useless recording).
let waitDeadline = Date().addingTimeInterval(waitAudioSeconds)
while Date() < waitDeadline {
    recorder.lock.lock()
    let peak = recorder.peak
    recorder.lock.unlock()
    if peak > 0.0001 { break }
    RunLoop.current.run(until: Date().addingTimeInterval(0.1))
}
recorder.lock.lock()
let detectedPeak = recorder.peak
if detectedPeak <= 0.0001 {
    recorder.lock.unlock()
    _ = AudioDeviceStop(aggregateID, ioProcID)
    _ = AudioDeviceDestroyIOProcID(aggregateID, ioProcID)
    AudioHardwareDestroyAggregateDevice(aggregateID)
    AudioHardwareDestroyProcessTap(tapID)
    fail(12, "NO_AUDIO_FROM_TARGET within \(Int(waitAudioSeconds))s (is the player actually playing?)")
}
recorder.started = true
recorder.lock.unlock()
print("audio_detected peak=\(detectedPeak)")

let recordUntil = Date().addingTimeInterval(seconds)
while Date() < recordUntil {
    RunLoop.current.run(until: Date().addingTimeInterval(0.05))
}

_ = AudioDeviceStop(aggregateID, ioProcID)
_ = AudioDeviceDestroyIOProcID(aggregateID, ioProcID)
AudioHardwareDestroyAggregateDevice(aggregateID)
AudioHardwareDestroyProcessTap(tapID)

recorder.lock.lock()
let payload = recorder.data
let peak = recorder.peak
recorder.lock.unlock()

let bytesPerFrame = Int(format.mBytesPerFrame)
let frames = bytesPerFrame > 0 ? payload.count / bytesPerFrame : 0
let sidecar: [String: Any] = [
    "sample_rate": format.mSampleRate,
    "channels": Int(format.mChannelsPerFrame),
    "bits_per_channel": 32,
    "format": "float32le-interleaved",
    "bytes_recorded": payload.count,
    "frames": frames,
    "seconds": format.mSampleRate > 0 ? Double(frames) / format.mSampleRate : 0,
    "peak": peak,
    "target_pid": Int(targetPID),
    "mute_behavior": muteRaw,
]
do {
    try payload.write(to: URL(fileURLWithPath: outPath))
    let sidecarPath = outPath + ".json"
    let data = try JSONSerialization.data(withJSONObject: sidecar, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: URL(fileURLWithPath: sidecarPath))
} catch {
    fail(13, "WRITE_FAILED \(error)")
}
print("ROOT_OK frames=\(frames) seconds=\(sidecar["seconds"]!) sample_rate=\(format.mSampleRate)")
