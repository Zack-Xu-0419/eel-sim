"""Offscreen MP4/JPG export of the runner's already calculated poses.

MuJoCo renders the actual model. Export never advances mechanical dynamics.
Pillow and FFmpeg are needed only for the requested image/video outputs.
"""
from dataclasses import dataclass
import csv
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import mujoco
import numpy as np

from src.hydraulic.hydraulic_model import finite


@dataclass(frozen=True)
class ExportOptions:
    video_path: Path | None = None
    jpg_path: Path | None = None
    frames_dir: Path | None = None
    width: int = 1920
    height: int = 1080
    fps: float = 60.
    video_crf: int = 18
    jpg_quality: int = 95
    snapshot_time_s: float | None = None
    overlay: bool = False

    def __post_init__(self):
        if not isinstance(self.overlay, bool):
            raise ValueError('overlay must be a boolean')
        for name, low, high in [('width', 1, None), ('height', 1, None),
                                ('video_crf', 0, 51), ('jpg_quality', 1, 100)]:
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value < low or (high is not None and value > high)):
                raise ValueError(f'{name} must be an integer in [{low}, {high or "unbounded"}]')
        fps = finite(self.fps, 'export fps')
        if fps <= 0:
            raise ValueError('export fps must be positive')
        object.__setattr__(self, 'fps', fps)
        if self.video_path is not None and (self.width % 2 or self.height % 2):
            raise ValueError('MP4 width and height must be even for H.264/yuv420p')
        for name, extensions in [('video_path', ('.mp4',)), ('jpg_path', ('.jpg', '.jpeg'))]:
            value = getattr(self, name)
            if value is not None:
                value = Path(value).expanduser().resolve()
                if value.suffix.lower() not in extensions:
                    raise ValueError(f'{name} must end in {" or ".join(extensions)}')
                if value.is_dir():
                    raise ValueError(f'Output is a directory: {value}')
                object.__setattr__(self, name, value)
        if self.frames_dir is not None:
            folder = Path(self.frames_dir).expanduser().resolve()
            if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
                raise ValueError(f'JPG frames need a new or empty directory: {folder}')
            object.__setattr__(self, 'frames_dir', folder)
        if self.snapshot_time_s is not None:
            clock = finite(self.snapshot_time_s, 'snapshot time')
            if self.jpg_path is None:
                raise ValueError('--snapshot-time requires --save-jpg')
            object.__setattr__(self, 'snapshot_time_s', clock)


def frame_times(start_s, end_s, fps):
    """Uniform presentation timestamps; video duration rounds up by <1 frame.

    A 12 s, 60 fps run produces 720 frames, at 0 through 11.98333 s. The last
    frame occupies the final interval; no extra endpoint frame slows playback.
    """
    start, end, rate = finite(start_s, 'start time'), finite(end_s, 'end time'), finite(fps, 'fps')
    if end <= start or rate <= 0:
        raise ValueError('End time must exceed start time and fps must be positive')
    count = max(1, math.ceil((end-start)*rate-1e-10))
    return start + np.arange(count)/rate


def trajectory_arrays(samples):
    if not samples:
        raise ValueError('No hydraulic samples to export')
    clocks = np.array([s.time_s for s in samples], dtype=float)
    angles = np.array([s.angle_rad for s in samples], dtype=float)
    if (not np.all(np.isfinite(clocks)) or not np.all(np.isfinite(angles))
            or np.any(np.diff(clocks) <= 0)):
        raise ValueError('Export samples need finite angles and strictly increasing times')
    return clocks, angles


class TrajectoryTelemetry:
    """Values at a rendered time, with the CSV's completed-step convention.

    At a solver timestamp, flow/rate refer to the interval ending there. The
    initial sample uses the first interval. Angle velocity is the slope of the
    same piecewise-linear angle reconstruction used to render the pose.
    """
    def __init__(self, samples):
        self.clocks, self.angles = trajectory_arrays(samples)
        self.columns = {name: np.array([getattr(s, name) for s in samples], dtype=float)
                        for name in ('pump_flow_m3_s', 'leak_flow_m3_s',
                                     'volume_m3', 'pressure_difference_pa')}
        if any(not np.all(np.isfinite(column)) for column in self.columns.values()):
            raise ValueError('Telemetry samples must contain finite flows, volumes and pressures')
        self.rates = np.diff(self.angles)/np.diff(self.clocks)

    def at(self, time_s):
        clock = finite(time_s, 'telemetry time')
        if not self.clocks[0] <= clock <= self.clocks[-1]:
            raise ValueError('Telemetry time lies outside the calculated trajectory')
        interval = int(np.clip(np.searchsorted(self.clocks, clock, side='left')-1,
                               0, max(0, len(self.clocks)-2)))
        end = min(interval+1, len(self.clocks)-1)
        pump = float(self.columns['pump_flow_m3_s'][end])
        result = {name: float(np.interp(clock, self.clocks, self.columns[name]))
                  for name in ('leak_flow_m3_s', 'volume_m3', 'pressure_difference_pa')}
        result.update(time_s=clock, angle_rad=float(np.interp(clock, self.clocks, self.angles)),
                      angle_velocity_rad_s=float(self.rates[interval]) if len(self.rates) else None,
                      pump_flow_m3_s=pump, net_flow_m3_s=pump-result['leak_flow_m3_s'])
        return result


class TelemetryOverlay:
    """Resolution-scaled information panel; Pillow imports only when needed."""
    def __init__(self, width, height):
        from PIL import ImageFont
        self.scale = min(width/1920, height/1080)
        self.fonts = []
        for size, bold in [(28, True), (22, False), (27, False)]:
            candidates = [Path('/System/Library/Fonts/Supplemental')/
                          ('Arial Bold.ttf' if bold else 'Arial.ttf'),
                          Path('/usr/share/fonts/truetype/dejavu')/
                          ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')]
            pixels = max(8, round(size*self.scale))
            path = next((p for p in candidates if p.is_file()), None)
            self.fonts.append(ImageFont.truetype(str(path), pixels) if path else
                              ImageFont.load_default(size=pixels))

    def draw(self, pixels, values):
        from PIL import Image, ImageDraw
        image = Image.fromarray(pixels).convert('RGBA')
        panel = Image.new('RGBA', image.size)
        draw = ImageDraw.Draw(panel)
        scale = self.scale
        x, y = round(28*scale), round(28*scale)
        pad, line = max(6, round(22*scale)), max(12, round(44*scale))
        title, note, font = self.fonts
        rate = values['angle_velocity_rad_s']
        rows = [
            ('Time t', f"{values['time_s']:.3f} s"),
            ('Pump flow Qp', f"{values['pump_flow_m3_s']*1e6:+.3f} mL/s"),
            ('Leak flow', f"{values['leak_flow_m3_s']*1e6:+.3f} mL/s"),
            ('Net flow', f"{values['net_flow_m3_s']*1e6:+.3f} mL/s"),
            ('Total angle θ', f"{math.degrees(values['angle_rad']):+.2f}°"),
            ('Angular velocity θ̇', f'{math.degrees(rate):+.2f}°/s' if rate is not None else 'n/a'),
            ('Transferred volume', f"{values['volume_m3']*1e6:+.3f} mL"),
            ('Pressure Δp', f"{values['pressure_difference_pa']/1000:+.3f} kPa"),
        ]
        label_width = max(draw.textlength(label, font=font) for label, _ in rows)
        value_width = max(draw.textlength(value, font=font) for _, value in rows)
        width = max(round(610*scale), int(label_width+value_width+3*pad))
        top = 2*line
        height = top + len(rows)*line + 2*pad
        draw.rounded_rectangle((x, y, x+width, y+height), radius=max(4, round(12*scale)),
                               fill=(15, 25, 40, 232))
        draw.text((x+pad, y+pad), 'HYDRAULIC SECTION', font=title, fill=(255, 255, 255))
        draw.text((x+pad, y+pad+line), 'MODEL VALUES • UNCALIBRATED', font=note,
                  fill=(183, 198, 215))
        for i, (label, value) in enumerate(rows):
            row_y = y+pad+top+i*line
            draw.text((x+pad, row_y), label, font=font, fill=(231, 237, 244))
            value_x = x+width-pad-draw.textlength(value, font=font)
            draw.text((value_x, row_y), value, font=font, fill=(116, 221, 238))
        return np.asarray(Image.alpha_composite(image, panel).convert('RGB'))


def export_media(section, samples, options):
    """Render frames once; stream video and optionally save JPEG frames.

    A lone JPEG defaults to the largest absolute bend in the trajectory.
    Angles between integration samples are interpolated linearly for uniform
    video timing, independently of the solver timestep. Existing MP4/JPG files
    are replaced only after successful encoding. Frame folders must be empty.
    """
    clocks, angles = trajectory_arrays(samples)
    telemetry = TrajectoryTelemetry(samples)
    snapshot_time = (float(clocks[np.argmax(np.abs(angles))]) if options.snapshot_time_s is None
                     else options.snapshot_time_s)
    if options.jpg_path is not None and not clocks[0] <= snapshot_time <= clocks[-1]:
        raise ValueError(f'Snapshot time must be between {clocks[0]:g} and {clocks[-1]:g} s')
    wants_sequence = options.video_path is not None or options.frames_dir is not None
    times = frame_times(clocks[0], clocks[-1], options.fps) if wants_sequence else np.array([])
    ffmpeg = shutil.which('ffmpeg') if options.video_path is not None else None
    if options.video_path is not None and ffmpeg is None:
        raise RuntimeError('MP4 export needs FFmpeg on PATH; JPG export does not require it')
    if options.jpg_path is not None or options.frames_dir is not None or options.overlay:
        from PIL import Image
    overlay = TelemetryOverlay(options.width, options.height) if options.overlay else None

    record = dict(status='completed', width=options.width, height=options.height,
                  fps=options.fps, video_crf=options.video_crf, jpg_quality=options.jpg_quality,
                  frame_count=len(times), interpolation='Linear interpolation of calculated angles',
                  video_path=str(options.video_path) if options.video_path is not None else None,
                  jpg_path=str(options.jpg_path) if options.jpg_path is not None else None,
                  frames_dir=str(options.frames_dir) if options.frames_dir is not None else None,
                  overlay=options.overlay,
                  angle_velocity_definition='Finite difference over the interval ending at t; first interval at t=0',
                  telemetry_units='SI in CSV/JSON; overlay uses mL/s, degrees, degrees/s, mL and kPa')
    if options.jpg_path is not None:
        record.update(snapshot_time_s=snapshot_time,
                      snapshot_angle_rad=float(np.interp(snapshot_time, clocks, angles)),
                      snapshot_telemetry=telemetry.at(snapshot_time))
    if options.video_path is not None:
        record['video_duration_s'] = len(times)/options.fps

    model, data = section.model, section.data
    original_qpos, original_qvel, original_time = data.qpos.copy(), data.qvel.copy(), data.time
    original_width, original_height = model.vis.global_.offwidth, model.vis.global_.offheight
    original_samples = model.vis.quality.offsamples
    model.vis.global_.offwidth = max(options.width, original_width)
    model.vis.global_.offheight = max(options.height, original_height)
    model.vis.quality.offsamples = max(4, original_samples)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [.06, 0, 0]
    camera.distance, camera.azimuth, camera.elevation = .36, 145, -18
    process, video_temp, jpg_temp, renderer = None, None, None, None
    try:
        renderer = mujoco.Renderer(model, height=options.height, width=options.width)

        def render(clock):
            values = telemetry.at(float(clock))
            section.apply(values['angle_rad'], float(clock), values['angle_velocity_rad_s'] or 0.)
            renderer.update_scene(data, camera=camera)
            pixels = renderer.render()
            if overlay is not None:
                pixels = overlay.draw(pixels, values)
            return pixels, values

        if options.frames_dir is not None:
            options.frames_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile() as encoder_log:
            if options.video_path is not None:
                options.video_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=options.video_path.parent, suffix='.mp4', delete=False) as file:
                    video_temp = Path(file.name)
                command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                           '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                           '-s:v', f'{options.width}x{options.height}', '-r', f'{options.fps:g}',
                           '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
                           '-crf', str(options.video_crf), '-pix_fmt', 'yuv420p',
                           '-movflags', '+faststart', str(video_temp)]
                process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                           stdout=subprocess.DEVNULL, stderr=encoder_log)
            frame_records = []
            try:
                for index, clock in enumerate(times):
                    pixels, values = render(clock)
                    if process is not None:
                        process.stdin.write(np.ascontiguousarray(pixels).tobytes())
                    if options.frames_dir is not None:
                        name = f'frame_{index:06d}.jpg'
                        Image.fromarray(pixels).save(options.frames_dir/name, quality=options.jpg_quality,
                                                     subsampling=0)
                    frame_records.append(dict(frame=index, **values,
                                              file=name if options.frames_dir is not None else ''))
                    if index % max(1, round(options.fps)) == 0:
                        print(f'Export: {index+1}/{len(times)} frames', flush=True)
                if process is not None:
                    process.stdin.close()
                    if process.wait() != 0:
                        raise RuntimeError('FFmpeg failed to encode the video')
            except BrokenPipeError as exc:
                raise RuntimeError('FFmpeg stopped while encoding the video') from exc
            except Exception:
                if process is not None and process.poll() is None:
                    process.terminate()
                    process.wait()
                raise
            finally:
                if process is not None and process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
                if process is not None and process.returncode:
                    encoder_log.seek(0)
                    details = encoder_log.read().decode(errors='replace').strip()
                    if details:
                        print(f'FFmpeg: {details}')
            if options.frames_dir is not None:
                with (options.frames_dir/'frames.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(frame_records[0]))
                    writer.writeheader()
                    writer.writerows(frame_records)

        if options.jpg_path is not None:
            options.jpg_path.parent.mkdir(parents=True, exist_ok=True)
            pixels, _ = render(snapshot_time)
            with tempfile.NamedTemporaryFile(dir=options.jpg_path.parent, suffix='.jpg', delete=False) as file:
                jpg_temp = Path(file.name)
            Image.fromarray(pixels).save(jpg_temp, quality=options.jpg_quality, subsampling=0)
            jpg_temp.replace(options.jpg_path)
        if video_temp is not None:
            video_temp.replace(options.video_path)
            telemetry_path = options.video_path.with_name(options.video_path.stem+'_telemetry.csv')
            with telemetry_path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(frame_records[0]))
                writer.writeheader()
                writer.writerows(frame_records)
            record['video_telemetry_csv'] = str(telemetry_path)
        return record
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        if renderer is not None:
            renderer.close()
        for path in (video_temp, jpg_temp):
            if path is not None:
                path.unlink(missing_ok=True)
        model.vis.global_.offwidth, model.vis.global_.offheight = original_width, original_height
        model.vis.quality.offsamples = original_samples
        data.qpos[:], data.qvel[:], data.time = original_qpos, original_qvel, original_time
        mujoco.mj_forward(model, data)
