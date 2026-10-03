import AVFoundation
import Carbon.HIToolbox
import Cocoa
import Metal
import QuartzCore
import Vision

// MARK: - Tunables

struct Settings {
    var maxSpawnPerFrame = 16
    var perBlob = 24             // max windows one movement can keep alive; more motion = more windows
    var faceBase = 2             // windows each eye/mouth keeps alive while a face is visible
    var faceBurst = 10           // extra windows when that eye/mouth moves (blink, talk)
    var maxWindows = 180
    var bigChance = 0.03         // occasional large window
}

let GW = 64, GH = 36   // motion grid

// MARK: - Geometry

/// Maps between screen points (bottom-left origin) and raw camera pixels (top-left origin,
/// unmirrored). The camera is aspect-filled to the screen and shown mirrored.
struct Mapping {
    let screen: CGSize
    let cam: CGSize
    let scale: CGFloat   // screen points per camera pixel
    let off: CGPoint     // camera px cropped off left/top by aspect fill

    init(screen: CGSize, cam: CGSize) {
        self.screen = screen
        self.cam = cam
        scale = max(screen.width / cam.width, screen.height / cam.height)
        off = CGPoint(x: (cam.width - screen.width / scale) / 2,
                      y: (cam.height - screen.height / scale) / 2)
    }

    /// Where the whole camera image lands on screen.
    var cameraRectOnScreen: CGRect {
        CGRect(x: -off.x * scale, y: -off.y * scale, width: cam.width * scale, height: cam.height * scale)
    }

    func screenPoint(raw p: CGPoint) -> CGPoint {
        CGPoint(x: (cam.width - p.x - off.x) * scale, y: screen.height - (p.y - off.y) * scale)
    }

    func screenRect(raw r: CGRect) -> CGRect {
        let a = screenPoint(raw: CGPoint(x: r.minX, y: r.minY))
        let b = screenPoint(raw: CGPoint(x: r.maxX, y: r.maxY))
        return CGRect(x: min(a.x, b.x), y: min(a.y, b.y), width: abs(a.x - b.x), height: abs(a.y - b.y))
    }
}

/// One connected patch of motion, in screen coordinates.
struct Blob {
    let rect: CGRect
    let cells: Int
}

// MARK: - Camera, motion and face detection

final class Camera: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    let queue = DispatchQueue(label: "segcam.capture", qos: .userInteractive)
    var threshold = 24          // per-cell luma change (0-255) that counts as motion; `queue` only
    var faceTracking = true     // `queue` only
    var onFrame: ((CVPixelBuffer, Mapping, [Blob]) -> Void)?
    var onFace: (([CGRect]) -> Void)?   // [leftEye, rightEye, mouth] in screen coords, or []

    private let screenSize: CGSize
    private var mapping: Mapping?
    private var prev = [UInt8](repeating: 0, count: GW * GH)
    private var mask = [Bool](repeating: false, count: GW * GH)
    private var visited = [Bool](repeating: false, count: GW * GH)
    private var hasPrev = false
    private var busy = false     // main thread hasn't applied the previous frame yet
    private var lastFace: CFTimeInterval = 0
    private var faceROI: CGRect?     // last face, normalized; Vision only searches near it

    private let visionQueue = DispatchQueue(label: "segcam.vision", qos: .userInitiated)
    private let faceRequest = VNDetectFaceLandmarksRequest()
    private var visionBusy = false   // `queue` only

    init(screenSize: CGSize) {
        self.screenSize = screenSize
        super.init()
    }

    func configure() -> Bool {
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .unspecified)
        guard let device = discovery.devices.first(where: { $0.localizedName.contains("FaceTime") })
                ?? discovery.devices.first
                ?? AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: device) else { return false }

        session.beginConfiguration()
        session.sessionPreset = session.canSetSessionPreset(.hd1280x720) ? .hd1280x720 : .high
        guard session.canAddInput(input) else { return false }
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(output) else { return false }
        session.addOutput(output)
        useFastestFormat(device)
        session.commitConfiguration()
        return true
    }

    /// Picks the format near 720p with the highest frame rate (up to 60 fps): smoother live feed.
    private func useFastestFormat(_ device: AVCaptureDevice) {
        func fps(_ f: AVCaptureDevice.Format) -> Double {
            min(60, f.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0)
        }
        func score(_ f: AVCaptureDevice.Format) -> Double {
            let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
            return fps(f) * 10_000 - Double(abs(Int(d.width) - 1280))
        }
        let candidates = device.formats.filter {
            let d = CMVideoFormatDescriptionGetDimensions($0.formatDescription)
            return d.width >= 960 && d.width <= 1920
        }
        guard let best = candidates.max(by: { score($0) < score($1) }), fps(best) > 30,
              (try? device.lockForConfiguration()) != nil else { return }
        device.activeFormat = best
        device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: CMTimeScale(fps(best)))
        device.unlockForConfiguration()
    }

    func start() { queue.async { self.hasPrev = false; self.session.startRunning() } }
    func stop() { queue.async { self.session.stopRunning() } }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let camSize = CGSize(width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb))
        if mapping?.cam != camSize { mapping = Mapping(screen: screenSize, cam: camSize); hasPrev = false }
        let map = mapping!

        CVPixelBufferLockBaseAddress(pb, .readOnly)
        let blobs = detectMotion(pb, map: map)
        CVPixelBufferUnlockBaseAddress(pb, .readOnly)

        let now = CACurrentMediaTime()
        if faceTracking && !visionBusy && now - lastFace > 0.066 {
            visionBusy = true
            lastFace = now
            visionQueue.async { self.detectFace(pb, map: map) }
        }

        guard !busy else { return }
        busy = true
        DispatchQueue.main.async {
            self.onFrame?(pb, map, blobs)
            self.queue.async { self.busy = false }
        }
    }

    private func detectMotion(_ pb: CVPixelBuffer, map: Mapping) -> [Blob] {
        guard let base = CVPixelBufferGetBaseAddress(pb)?.assumingMemoryBound(to: UInt8.self) else { return [] }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        let bpr = CVPixelBufferGetBytesPerRow(pb)
        let cellW = w / GW, cellH = h / GH
        var moving = 0

        @inline(__always) func luma(_ p: UnsafePointer<UInt8>) -> Int {
            (Int(p[0]) + 2 * Int(p[1]) + Int(p[2])) >> 2   // BGRA
        }

        // 2x2 samples per cell, compared against a running average of recent frames.
        for gy in 0..<GH {
            let r0 = base + (gy * cellH + cellH / 4) * bpr
            let r1 = base + (gy * cellH + 3 * cellH / 4) * bpr
            for gx in 0..<GW {
                let xa = (gx * cellW + cellW / 4) * 4, xb = (gx * cellW + 3 * cellW / 4) * 4
                let l = (luma(r0 + xa) + luma(r0 + xb) + luma(r1 + xa) + luma(r1 + xb)) >> 2
                let i = gy * GW + gx
                let m = hasPrev && abs(l - Int(prev[i])) > threshold
                mask[i] = m
                if m { moving += 1 }
                prev[i] = UInt8((Int(prev[i]) + l) >> 1)
            }
        }
        hasPrev = true

        // Too little = sensor noise; nearly everything = auto-exposure jump.
        guard moving >= 4, moving < GW * GH * 7 / 10 else { return [] }

        // Connected components (8-neighbour) -> one blob per separate movement.
        for i in 0..<visited.count { visited[i] = false }
        let sx = map.cam.width / CGFloat(GW), sy = map.cam.height / CGFloat(GH)
        let screenBounds = CGRect(origin: .zero, size: screenSize)
        var blobs: [Blob] = []
        var stack: [Int] = []
        for i in 0..<mask.count where mask[i] && !visited[i] {
            var minX = GW, maxX = 0, minY = GH, maxY = 0, count = 0
            visited[i] = true
            stack.append(i)
            while let j = stack.popLast() {
                let x = j % GW, y = j / GW
                count += 1
                minX = min(minX, x); maxX = max(maxX, x); minY = min(minY, y); maxY = max(maxY, y)
                for ny in max(0, y - 1)...min(GH - 1, y + 1) {
                    for nx in max(0, x - 1)...min(GW - 1, x + 1) {
                        let k = ny * GW + nx
                        if mask[k] && !visited[k] { visited[k] = true; stack.append(k) }
                    }
                }
            }
            guard count >= 2 else { continue }
            let raw = CGRect(x: CGFloat(minX) * sx, y: CGFloat(minY) * sy,
                             width: CGFloat(maxX - minX + 1) * sx, height: CGFloat(maxY - minY + 1) * sy)
            let rect = map.screenRect(raw: raw).intersection(screenBounds)
            if !rect.isEmpty { blobs.append(Blob(rect: rect, cells: count)) }
        }
        return blobs
    }

    private func detectFace(_ pb: CVPixelBuffer, map: Mapping) {
        let handler = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up, options: [:])
        faceRequest.regionOfInterest = faceROI ?? CGRect(x: 0, y: 0, width: 1, height: 1)
        try? handler.perform([faceRequest])
        var rects: [CGRect] = []
        let face = (faceRequest.results ?? []).max(by: { $0.boundingBox.width < $1.boundingBox.width })
        // Results are relative to the ROI; convert back to full-image coordinates.
        faceROI = face.map { f in
            let roi = faceRequest.regionOfInterest
            let b = CGRect(x: roi.minX + f.boundingBox.minX * roi.width, y: roi.minY + f.boundingBox.minY * roi.height,
                           width: f.boundingBox.width * roi.width, height: f.boundingBox.height * roi.height)
            return b.insetBy(dx: -b.width * 0.6, dy: -b.height * 0.6)
                .intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
        }
        if let face, let lm = face.landmarks {
            for region in [lm.leftEye, lm.rightEye, lm.outerLips] {
                // Vision points are bottom-left origin; flip to raw top-left.
                guard let pts = region?.pointsInImage(imageSize: map.cam), !pts.isEmpty else { rects = []; break }
                let xs = pts.map(\.x), ys = pts.map { map.cam.height - $0.y }
                let raw = CGRect(x: xs.min()!, y: ys.min()!, width: xs.max()! - xs.min()!, height: ys.max()! - ys.min()!)
                rects.append(map.screenRect(raw: raw))
            }
        }
        DispatchQueue.main.async { self.onFace?(rects) }
        queue.async { self.visionBusy = false }
    }
}

// MARK: - Anchors (things windows follow)

final class Anchor {
    enum Kind { case motion, face }
    let kind: Kind
    var rect: CGRect
    var lastSeen: CFTimeInterval
    var cells = 0        // motion strength this frame
    var windows = 0      // live windows attached

    init(kind: Kind, rect: CGRect, now: CFTimeInterval) {
        self.kind = kind
        self.rect = rect
        lastSeen = now
    }
}

// MARK: - Fake window (just data; the Metal renderer draws it)

final class FakeWindow {
    static let popIn = 0.0, popOut = 0.0     // instant: no open/close animation

    let anchor: Anchor
    let size: CGSize
    let titleH: CGFloat
    let radius: CGFloat
    let offset: CGPoint        // from anchor centre, points
    let stiffness: Double      // spring frequency; varied per window so they trail each other
    let born: CFTimeInterval
    var death: CFTimeInterval
    var center: CGPoint
    var velocity = CGVector.zero
    var scale: CGFloat = popIn > 0 ? 0.85 : 1
    var active = false         // only the frontmost window gets colored traffic lights
    var dying = false

    init(anchor: Anchor, size: CGSize, titleH: CGFloat, offset: CGPoint, stiffness: Double,
         life: Double, now: CFTimeInterval) {
        self.anchor = anchor
        self.size = size
        self.titleH = titleH
        self.offset = offset
        self.stiffness = stiffness
        radius = max(5, min(14, min(size.width, size.height) * 0.08))
        born = now
        death = now + life
        center = CGPoint(x: anchor.rect.midX + offset.x, y: anchor.rect.midY + offset.y)
    }

    /// Spring toward the anchor and pop in/out.
    func step(dt: Double, now: CFTimeInterval) {
        let target = CGPoint(x: anchor.rect.midX + offset.x, y: anchor.rect.midY + offset.y)
        // Slightly under-damped spring: windows accelerate, glide and settle with a tiny sway.
        // Sub-stepped at <= 1/120 s so stiff springs stay stable.
        let w = CGFloat(stiffness), damping = 2 * 0.8 * w
        let steps = max(1, Int((dt * 120).rounded(.up))), h = CGFloat(dt) / CGFloat(steps)
        for _ in 0..<steps {
            velocity.dx += ((target.x - center.x) * w * w - velocity.dx * damping) * h
            velocity.dy += ((target.y - center.y) * w * w - velocity.dy * damping) * h
            center.x += velocity.dx * h
            center.y += velocity.dy * h
        }

        if Self.popIn > 0, now < born + Self.popIn {
            let t = (now - born) / Self.popIn             // easeOutCubic from 85%
            scale = CGFloat(0.85 + 0.15 * (1 - pow(1 - t, 3)))
        } else if Self.popOut > 0, now > death {
            scale = CGFloat(max(0, 1 - (now - death) / Self.popOut))
        } else {
            scale = 1
        }
    }
}

// MARK: - Metal renderer: every window in one instanced draw call

struct Instance {
    var rect: SIMD4<Float>     // x, y, w, h in points, bottom-left origin
    var params: SIMD4<Float>   // titleH, corner radius, pop scale, active
}

struct Uniforms {
    var view: SIMD4<Float>     // screen w, h in points, backing scale, unused
    var camRect: SIMD4<Float>  // where the whole camera image lands on screen
}

final class Renderer {
    static let shader = """
    #include <metal_stdlib>
    using namespace metal;

    struct Instance { float4 rect; float4 params; };
    struct Uniforms { float4 view; float4 camRect; };
    struct VOut {
        float4 position [[position]];
        float2 local;              // point in the window, unscaled, bottom-left origin
        float2 screen;             // point on screen
        float2 size;
        float4 params [[flat]];
    };

    constant float kMargin = 18.0; // room around the window for its shadow
    constant float3 kLights[3] = { float3(1.0, 0.37, 0.34), float3(1.0, 0.74, 0.18), float3(0.16, 0.78, 0.25) };

    vertex VOut windowVertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                             const device Instance* inst [[buffer(0)]], constant Uniforms& u [[buffer(1)]]) {
        Instance w = inst[iid];
        float2 corner = float2(vid & 1, vid >> 1);
        float2 size = w.rect.zw;
        float2 local = mix(float2(-kMargin), size + kMargin, corner);
        float2 screen = w.rect.xy + size * 0.5 + (local - size * 0.5) * w.params.z;
        VOut o;
        o.position = float4(screen / u.view.xy * 2.0 - 1.0, 0.0, 1.0);
        o.local = local;
        o.screen = screen;
        o.size = size;
        o.params = w.params;
        return o;
    }

    float roundedBox(float2 p, float2 halfSize, float r) {
        float2 q = abs(p) - halfSize + r;
        return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
    }

    fragment float4 windowFragment(VOut in [[stage_in]], constant Uniforms& u [[buffer(0)]],
                                   texture2d<float> cam [[texture(0)]]) {
        constexpr sampler smp(filter::linear, address::clamp_to_edge);
        float titleH = in.params.x, radius = in.params.y;
        float px = 1.0 / (u.view.z * max(in.params.z, 0.05));   // one device pixel, in window units
        float2 halfSize = in.size * 0.5;
        float2 p = in.local - halfSize;

        float d = roundedBox(p, halfSize, radius);
        float shadow = 0.45 * (1.0 - smoothstep(-4.0, 14.0, roundedBox(p - float2(0.0, -5.0), halfSize, radius)));
        float cover = saturate(0.5 - d / px);
        if (cover <= 0.0) return float4(0.0, 0.0, 0.0, shadow);

        float3 c;
        if (in.local.y > in.size.y - titleH) {
            c = float3(0.16);
            float dotR = titleH * 0.2, gap = titleH * 0.24;
            for (int i = 0; i < 3; i++) {
                float2 dc = float2(titleH * 0.45 + dotR + i * (2.0 * dotR + gap), in.size.y - titleH * 0.5);
                float a = saturate(0.5 - (length(in.local - dc) - dotR) / px);
                c = mix(c, in.params.w > 0.5 ? kLights[i] : float3(0.4), a);
            }
        } else {
            // The slice of the mirrored camera image behind this spot. Texture row 0 = top.
            float2 uv = (in.screen - u.camRect.xy) / u.camRect.zw;
            c = cam.sample(smp, float2(1.0 - uv.x, 1.0 - uv.y)).rgb;
        }
        c = mix(c, float3(1.0), 0.18 * saturate(1.0 + d / px));   // hairline border
        return float4(c * cover, cover + shadow * (1.0 - cover));
    }
    """

    let layer = CAMetalLayer()
    private let queue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private var textureCache: CVMetalTextureCache?
    private var camTexture: CVMetalTexture?
    private let maxInstances = 512
    private let buffers: [MTLBuffer]
    private var bufferIndex = 0
    private let inFlight = DispatchSemaphore(value: 3)
    private var drewWindows = false
    private let viewSize: CGSize
    private let pxScale: CGFloat

    init?(size: CGSize, scale: CGFloat) {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
              let library = try? device.makeLibrary(source: Self.shader, options: nil) else { return nil }
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = library.makeFunction(name: "windowVertex")
        desc.fragmentFunction = library.makeFunction(name: "windowFragment")
        let att = desc.colorAttachments[0]!
        att.pixelFormat = .bgra8Unorm
        att.isBlendingEnabled = true     // premultiplied alpha "over"
        att.sourceRGBBlendFactor = .one
        att.sourceAlphaBlendFactor = .one
        att.destinationRGBBlendFactor = .oneMinusSourceAlpha
        att.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        guard let pipeline = try? device.makeRenderPipelineState(descriptor: desc) else { return nil }

        self.queue = queue
        self.pipeline = pipeline
        viewSize = size
        pxScale = scale
        buffers = (0..<3).map { _ in
            device.makeBuffer(length: 512 * MemoryLayout<Instance>.stride, options: .storageModeShared)!
        }
        CVMetalTextureCacheCreate(nil, nil, device, nil, &textureCache)

        layer.device = device
        layer.pixelFormat = .bgra8Unorm
        layer.isOpaque = false
        layer.framebufferOnly = true
        layer.contentsScale = scale
        layer.drawableSize = CGSize(width: size.width * scale, height: size.height * scale)
        layer.maximumDrawableCount = 3
    }

    /// Wraps the camera frame as a Metal texture without copying it.
    func setCamera(_ pb: CVPixelBuffer) {
        guard let cache = textureCache else { return }
        var tex: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(nil, cache, pb, nil, .bgra8Unorm,
                                                  CVPixelBufferGetWidth(pb), CVPixelBufferGetHeight(pb), 0, &tex)
        if let tex { camTexture = tex }
    }

    func clearCamera() { camTexture = nil }

    func draw(_ windows: [FakeWindow], cameraRect: CGRect) {
        let tex = camTexture.flatMap(CVMetalTextureGetTexture)
        let n = tex == nil ? 0 : min(windows.count, maxInstances)
        if n == 0 && !drewWindows { return }    // already transparent; let the GPU idle
        guard inFlight.wait(timeout: .now() + .milliseconds(30)) == .success else { return }
        guard let drawable = layer.nextDrawable(), let cb = queue.makeCommandBuffer() else {
            inFlight.signal()
            return
        }

        let buf = buffers[bufferIndex]
        bufferIndex = (bufferIndex + 1) % buffers.count
        let out = buf.contents().bindMemory(to: Instance.self, capacity: maxInstances)
        for (i, w) in windows.suffix(n).enumerated() {   // spawn order: newest drawn on top
            out[i] = Instance(
                rect: SIMD4(Float(w.center.x - w.size.width / 2), Float(w.center.y - w.size.height / 2),
                            Float(w.size.width), Float(w.size.height)),
                params: SIMD4(Float(w.titleH), Float(w.radius), Float(w.scale), w.active ? 1 : 0))
        }
        var u = Uniforms(
            view: SIMD4(Float(viewSize.width), Float(viewSize.height), Float(pxScale), 0),
            camRect: SIMD4(Float(cameraRect.minX), Float(cameraRect.minY), Float(cameraRect.width), Float(cameraRect.height)))

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = drawable.texture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.colorAttachments[0].storeAction = .store
        if let enc = cb.makeRenderCommandEncoder(descriptor: pass) {
            if n > 0 {
                enc.setRenderPipelineState(pipeline)
                enc.setVertexBuffer(buf, offset: 0, index: 0)
                enc.setVertexBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
                enc.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 0)
                enc.setFragmentTexture(tex, index: 0)
                enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: n)
            }
            enc.endEncoding()
        }
        let held = camTexture   // keep the camera buffer out of the capture pool until the GPU is done
        let sem = inFlight
        cb.addCompletedHandler { _ in
            _ = held
            sem.signal()
        }
        cb.present(drawable)
        cb.commit()
        drewWindows = n > 0
    }
}

// MARK: - App

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var overlay: NSWindow!
    private var renderer: Renderer!
    private var camera: Camera!
    private var link: CADisplayLink?
    private var settings = Settings()
    private var windows: [FakeWindow] = []
    private var front: FakeWindow?
    private var motionAnchors: [Anchor] = []
    private var faceAnchors: [Anchor] = []        // leftEye, rightEye, mouth
    private var faceTracking = true
    private var cameraRect = CGRect.zero
    private var lastTick = CACurrentMediaTime()
    private var statusItem: NSStatusItem!
    private var statusLine: NSMenuItem!
    private var toggleItem: NSMenuItem!
    private var hotKey: EventHotKeyRef?
    private var paused = false

    func applicationDidFinishLaunching(_ note: Notification) {
        let screen = NSScreen.main!
        guard let renderer = Renderer(size: screen.frame.size, scale: screen.backingScaleFactor) else {
            fail("SegCam needs Metal", "This Mac's GPU couldn't be set up.")
            return
        }
        self.renderer = renderer

        overlay = NSWindow(contentRect: screen.frame, styleMask: .borderless, backing: .buffered, defer: false)
        overlay.setFrame(screen.frame, display: false)
        overlay.level = .screenSaver
        overlay.isOpaque = false
        overlay.backgroundColor = .clear
        overlay.hasShadow = false
        overlay.ignoresMouseEvents = true
        overlay.collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary, .ignoresCycle]
        let view = NSView(frame: CGRect(origin: .zero, size: screen.frame.size))
        view.layer = renderer.layer   // layer-hosting view
        view.wantsLayer = true
        overlay.contentView = view
        overlay.orderFrontRegardless()

        let link = view.displayLink(target: self, selector: #selector(tick(_:)))
        link.preferredFrameRateRange = CAFrameRateRange(minimum: 50, maximum: 60, preferred: 60)
        link.add(to: .main, forMode: .common)
        self.link = link

        buildMenu()

        camera = Camera(screenSize: screen.frame.size)
        camera.onFrame = { [weak self] pb, map, blobs in self?.cameraFrame(pb, map, blobs) }
        camera.onFace = { [weak self] rects in self?.faceUpdate(rects) }
        AVCaptureDevice.requestAccess(for: .video) { granted in
            DispatchQueue.main.async {
                guard granted, self.camera.configure() else {
                    self.fail("SegCam needs the camera",
                              "Allow camera access in System Settings → Privacy & Security → Camera.")
                    return
                }
                self.camera.start()
            }
        }
    }

    private func fail(_ title: String, _ detail: String) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = detail
        alert.runModal()
        NSApp.terminate(nil)
    }

    // MARK: Per camera frame: track movements, spawn windows, swap in the new image

    private func cameraFrame(_ pb: CVPixelBuffer, _ map: Mapping, _ blobs: [Blob]) {
        guard !paused else { return }
        let now = CACurrentMediaTime()
        cameraRect = map.cameraRectOnScreen
        renderer.setCamera(pb)

        // Match each movement to the nearest existing anchor, or start a new one.
        var matched = Set<ObjectIdentifier>()
        for blob in blobs.sorted(by: { $0.cells > $1.cells }).prefix(12) {
            let c = CGPoint(x: blob.rect.midX, y: blob.rect.midY)
            let reach = max(90, max(blob.rect.width, blob.rect.height))
            let best = motionAnchors
                .filter { !matched.contains(ObjectIdentifier($0)) }
                .min { hypot($0.rect.midX - c.x, $0.rect.midY - c.y) < hypot($1.rect.midX - c.x, $1.rect.midY - c.y) }
            let anchor: Anchor
            if let best, hypot(best.rect.midX - c.x, best.rect.midY - c.y) < reach {
                anchor = best
                anchor.rect = lerp(anchor.rect, blob.rect, 0.75)
                anchor.lastSeen = now
            } else {
                anchor = Anchor(kind: .motion, rect: blob.rect, now: now)
                motionAnchors.append(anchor)
            }
            anchor.cells = blob.cells
            matched.insert(ObjectIdentifier(anchor))
        }
        motionAnchors.removeAll { $0.lastSeen < now - 0.4 && $0.windows == 0 }

        var budget = settings.maxSpawnPerFrame
        // The more a movement moves, the more windows it keeps alive and the faster they spawn.
        for a in motionAnchors.sorted(by: { $0.cells > $1.cells }) where a.lastSeen == now {
            let want = min(settings.perBlob, 1 + a.cells / 3)
            let burst = min(8, 2 + a.cells / 8, want - a.windows, budget)
            for _ in 0..<max(0, burst) { spawn(on: a, now: now) }
            budget -= max(0, burst)
        }

        // Eyes and mouth: always a few windows, a burst when they move.
        if faceTracking, let first = faceAnchors.first, first.lastSeen > now - 0.5 {
            for f in faceAnchors {
                let zone = f.rect.insetBy(dx: -25, dy: -25)
                let activity = blobs.reduce(0) { $0 + ($1.rect.intersects(zone) ? $1.cells : 0) }
                let want = settings.faceBase + min(settings.faceBurst, activity / 2)
                for _ in 0..<max(0, min(4, want - f.windows)) where budget > 0 {
                    spawn(on: f, now: now)
                    budget -= 1
                }
            }
        }

        // Over the cap: retire the oldest.
        var excess = windows.filter { !$0.dying }.count - settings.maxWindows
        for w in windows where excess > 0 && !w.dying {
            retire(w, now: now)
            excess -= 1
        }
    }

    private func faceUpdate(_ rects: [CGRect]) {
        guard faceTracking, rects.count == 3 else { return }
        let now = CACurrentMediaTime()
        if faceAnchors.isEmpty {
            faceAnchors = rects.map { Anchor(kind: .face, rect: $0, now: now) }
        } else {
            for (a, r) in zip(faceAnchors, rects) { a.rect = lerp(a.rect, r, 0.8); a.lastSeen = now }
        }
    }

    /// Random window shape: wide, landscape, square or tall. Returned as width / height.
    private func randomAspect() -> CGFloat {
        switch Double.random(in: 0..<1) {
        case ..<0.25: return .random(in: 1.7...2.8)    // wide strip
        case ..<0.60: return .random(in: 1.2...1.6)    // landscape
        case ..<0.75: return .random(in: 0.9...1.1)    // square
        default:      return .random(in: 0.45...0.75) // tall
        }
    }

    private func spawn(on a: Anchor, now: CFTimeInterval) {
        // `d` is the window's "diameter"; the shape keeps roughly the same area for any aspect.
        let d: CGFloat, offset: CGPoint, stiffness: Double, life: Double
        switch a.kind {
        case .motion:
            // Scales with how big the movement is: a blink-sized twitch vs. a full arm swing.
            let span = sqrt(a.rect.width * a.rect.height)
            let big = Double.random(in: 0..<1) < settings.bigChance
            d = big ? .random(in: 220...340) : min(320, max(50, span * .random(in: 0.3...0.75)))
            offset = CGPoint(x: a.rect.width * .random(in: -0.35...0.35), y: a.rect.height * .random(in: -0.35...0.35))
            stiffness = .random(in: 25...40)
            life = .random(in: 0.15...0.5)
        case .face:
            d = min(130, max(40, a.rect.width * .random(in: 0.5...0.95)))
            offset = CGPoint(x: d * .random(in: -0.6...0.6), y: d * .random(in: -0.6...0.6))
            stiffness = .random(in: 30...45)
            life = .random(in: 0.2...0.6)
        }
        let aspect = randomAspect()
        let size = CGSize(width: d * sqrt(aspect), height: d / sqrt(aspect) + 10)
        let titleH = min(22, max(10, d * 0.14))
        let w = FakeWindow(anchor: a, size: size, titleH: titleH, offset: offset, stiffness: stiffness,
                           life: life, now: now)
        windows.append(w)
        a.windows += 1
        front?.active = false
        w.active = true
        front = w
    }

    private func retire(_ w: FakeWindow, now: CFTimeInterval) {
        guard !w.dying else { return }
        w.dying = true
        w.death = min(w.death, now)
        w.anchor.windows -= 1
    }

    // MARK: Per display frame: fluid motion + one draw call

    @objc private func tick(_ link: CADisplayLink) {
        let now = CACurrentMediaTime()
        let dt = min(1.0 / 30, now - lastTick)   // keeps the stiff springs stable after a hitch
        lastTick = now
        windows.removeAll { w in
            if now >= w.death { retire(w, now: now) }
            return now > w.death + FakeWindow.popOut
        }
        for w in windows { w.step(dt: dt, now: now) }
        renderer.draw(windows, cameraRect: cameraRect)
    }

    private func clearWindows() {
        windows.removeAll()
        motionAnchors.removeAll()
        faceAnchors.removeAll()
        front = nil
        renderer.draw([], cameraRect: cameraRect)   // one transparent frame
        renderer.clearCamera()
    }

    private func lerp(_ a: CGRect, _ b: CGRect, _ t: CGFloat) -> CGRect {
        CGRect(x: a.minX + (b.minX - a.minX) * t, y: a.minY + (b.minY - a.minY) * t,
               width: a.width + (b.width - a.width) * t, height: a.height + (b.height - a.height) * t)
    }

    // MARK: Menu bar

    private func buildMenu() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem.button?.toolTip = "SegCam (⌃⌥⌘S to stop/start)"
        let menu = NSMenu()
        statusLine = NSMenuItem(title: "", action: nil, keyEquivalent: "")
        statusLine.isEnabled = false
        menu.addItem(statusLine)
        toggleItem = item("", #selector(toggleRunning), key: "s")
        toggleItem.keyEquivalentModifierMask = [.control, .option, .command]
        menu.addItem(toggleItem)
        menu.addItem(item("Clear Windows", #selector(clear), key: "k"))
        menu.addItem(.separator())
        menu.addItem(submenu("Sensitivity", ["Low", "Medium", "High"], selected: 1, action: #selector(setSensitivity)))
        menu.addItem(submenu("Density", ["Calm", "Normal", "Chaos"], selected: 1, action: #selector(setDensity)))
        let face = item("Eyes & Mouth Tracking", #selector(toggleFace))
        face.state = .on
        menu.addItem(face)
        menu.addItem(.separator())
        menu.addItem(item("Quit SegCam", #selector(quit), key: "q"))
        statusItem.menu = menu
        updateStatus()
        registerHotKey()
    }

    private func updateStatus() {
        statusItem.button?.image = Self.menuBarIcon(running: !paused)
        statusLine.title = paused ? "SegCam — Stopped" : "SegCam — Running"
        toggleItem.title = paused ? "Start SegCam" : "Stop SegCam"
    }

    /// Two stacked mini windows with traffic-light dots; outlined and slashed when stopped.
    private static func menuBarIcon(running: Bool) -> NSImage {
        let img = NSImage(size: NSSize(width: 20, height: 16), flipped: false) { _ in
            guard let ctx = NSGraphicsContext.current else { return false }
            NSColor.black.set()
            let back = NSBezierPath(roundedRect: NSRect(x: 6.5, y: 5.5, width: 12, height: 9), xRadius: 2, yRadius: 2)
            back.lineWidth = 1.3
            back.stroke()

            let frontRect = NSRect(x: 1.5, y: 1.5, width: 12.5, height: 9.5)
            ctx.compositingOperation = .clear
            NSBezierPath(roundedRect: frontRect.insetBy(dx: -1.3, dy: -1.3), xRadius: 3, yRadius: 3).fill()
            ctx.compositingOperation = .sourceOver
            let front = NSBezierPath(roundedRect: frontRect, xRadius: 2, yRadius: 2)
            front.lineWidth = 1.3
            if running { front.fill() } else { front.stroke() }

            // Traffic lights: knocked out of the filled window, or drawn into the outlined one.
            ctx.compositingOperation = running ? .clear : .sourceOver
            for i in 0..<3 {
                NSBezierPath(ovalIn: NSRect(x: 3.3 + CGFloat(i) * 2.6, y: 7.6, width: 1.7, height: 1.7)).fill()
            }
            ctx.compositingOperation = .sourceOver

            if !running {
                let slash = NSBezierPath()
                slash.move(to: NSPoint(x: 1, y: 15))
                slash.line(to: NSPoint(x: 19, y: 1))
                slash.lineWidth = 1.6
                slash.lineCapStyle = .round
                slash.stroke()
            }
            return true
        }
        img.isTemplate = true   // follows the menu bar's light/dark appearance
        img.accessibilityDescription = running ? "SegCam running" : "SegCam stopped"
        return img
    }

    /// Global ⌃⌥⌘S toggle. Carbon hot keys need no Accessibility permission.
    private func registerHotKey() {
        var spec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        InstallEventHandler(GetApplicationEventTarget(), { _, _, userData in
            let me = Unmanaged<AppDelegate>.fromOpaque(userData!).takeUnretainedValue()
            DispatchQueue.main.async { me.toggleRunning() }
            return noErr
        }, 1, &spec, Unmanaged.passUnretained(self).toOpaque(), nil)
        let id = EventHotKeyID(signature: OSType(0x5347_434D), id: 1)   // 'SGCM'
        RegisterEventHotKey(UInt32(kVK_ANSI_S), UInt32(controlKey | optionKey | cmdKey), id,
                            GetApplicationEventTarget(), 0, &hotKey)
    }

    private func item(_ title: String, _ action: Selector, key: String = "") -> NSMenuItem {
        let it = NSMenuItem(title: title, action: action, keyEquivalent: key)
        it.target = self
        return it
    }

    private func submenu(_ title: String, _ options: [String], selected: Int, action: Selector) -> NSMenuItem {
        let parent = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        let sub = NSMenu()
        for (i, name) in options.enumerated() {
            let it = item(name, action)
            it.tag = i
            it.state = i == selected ? .on : .off
            sub.addItem(it)
        }
        parent.submenu = sub
        return parent
    }

    private func check(_ sender: NSMenuItem) {
        sender.menu?.items.forEach { $0.state = $0 === sender ? .on : .off }
    }

    @objc private func toggleRunning() {
        paused.toggle()
        link?.isPaused = paused
        if paused { camera.stop(); clearWindows() } else { camera.start() }
        updateStatus()
    }

    @objc private func clear() { clearWindows() }

    @objc private func setSensitivity(_ sender: NSMenuItem) {
        check(sender)
        let threshold = [36, 24, 14][sender.tag]
        camera.queue.async { self.camera.threshold = threshold }
    }

    @objc private func setDensity(_ sender: NSMenuItem) {
        check(sender)
        let presets: [(spawn: Int, perBlob: Int, base: Int, burst: Int, max: Int)] = [
            (6, 8, 1, 4, 60), (16, 24, 2, 10, 180), (32, 48, 3, 16, 360),
        ]
        let p = presets[sender.tag]
        settings.maxSpawnPerFrame = p.spawn
        settings.perBlob = p.perBlob
        settings.faceBase = p.base
        settings.faceBurst = p.burst
        settings.maxWindows = p.max
    }

    @objc private func toggleFace(_ sender: NSMenuItem) {
        faceTracking.toggle()
        sender.state = faceTracking ? .on : .off
        let on = faceTracking
        camera.queue.async { self.camera.faceTracking = on }
        if !on { faceAnchors.removeAll() }
    }

    @objc private func quit() { NSApp.terminate(nil) }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
