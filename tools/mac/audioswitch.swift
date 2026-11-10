// audioswitch — tiny CoreAudio helper for the Khadas A2DP sink test.
// Usage:
//   audioswitch list                # device id | name | kind | is default output
//   audioswitch set <substring>     # set default output (+ system output) to first match
//   audioswitch get                 # print current default output
import CoreAudio
import Foundation

func getAddr(_ sel: AudioObjectPropertySelector, scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: sel, mScope: scope, mElement: kAudioObjectPropertyElementMain)
}

func cfStr(_ device: AudioDeviceID, _ sel: AudioObjectPropertySelector) -> String? {
    var addr = getAddr(sel)
    var size = UInt32(MemoryLayout<CFString?>.size)
    var value: CFString?
    let err = AudioObjectGetPropertyData(device, &addr, 0, nil, &size, &value)
    if err != noErr { return nil }
    return value as String?
}

func deviceName(_ d: AudioDeviceID) -> String {
    cfStr(d, kAudioObjectPropertyName) ?? "?"
}

func outputChannels(_ d: AudioDeviceID) -> Int {
    var addr = getAddr(kAudioDevicePropertyStreamConfiguration, scope: kAudioObjectPropertyScopeOutput)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(d, &addr, 0, nil, &size) == noErr, size > 0 else { return 0 }
    let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: 16)
    defer { raw.deallocate() }
    guard AudioObjectGetPropertyData(d, &addr, 0, nil, &size, raw) == noErr else { return 0 }
    let list = raw.assumingMemoryBound(to: AudioBufferList.self)
    let bufs = UnsafeMutableAudioBufferListPointer(list)
    return bufs.reduce(0) { $0 + Int($1.mNumberChannels) }
}

func allDevices() -> [AudioDeviceID] {
    var addr = getAddr(kAudioHardwarePropertyDevices)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size) == noErr else { return [] }
    let count = Int(size) / MemoryLayout<AudioDeviceID>.size
    var ids = [AudioDeviceID](repeating: 0, count: count)
    guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &ids) == noErr else { return [] }
    return ids
}

func defaultOutput() -> AudioDeviceID {
    var addr = getAddr(kAudioHardwarePropertyDefaultOutputDevice)
    var dev = AudioDeviceID(0)
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    _ = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &dev)
    return dev
}

func setDefaultOutput(_ dev: AudioDeviceID) -> OSStatus {
    var addr = getAddr(kAudioHardwarePropertyDefaultOutputDevice)
    var d = dev
    var err = AudioObjectSetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, UInt32(MemoryLayout<AudioDeviceID>.size), &d)
    if err != noErr { return err }
    addr = getAddr(kAudioHardwarePropertyDefaultSystemOutputDevice)
    err = AudioObjectSetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, UInt32(MemoryLayout<AudioDeviceID>.size), &d)
    return err
}

let args = CommandLine.arguments
let cmd = args.count > 1 ? args[1] : "list"
let cur = defaultOutput()

switch cmd {
case "list":
    for d in allDevices() {
        let out = outputChannels(d)
        let mark = d == cur ? "  [DEFAULT]" : ""
        print("\(d)\t\(deviceName(d))\toutch=\(out)\(mark)")
    }
case "get":
    print(deviceName(cur))
case "set":
    guard args.count > 2 else { print("usage: audioswitch set <name substring>"); exit(2) }
    let needle = args[2].lowercased()
    let match = allDevices().first { outputChannels($0) > 0 && deviceName($0).lowercased().contains(needle) }
    guard let dev = match else { print("no output device matching '\(args[2])'"); exit(1) }
    let err = setDefaultOutput(dev)
    print(err == noErr ? "default output -> \(deviceName(dev)) [\(dev)]" : "set failed: \(err)")
    exit(err == noErr ? 0 : 1)
default:
    print("usage: audioswitch [list|get|set <substring>]")
    exit(2)
}
