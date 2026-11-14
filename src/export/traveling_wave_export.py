"""Render a chain's calculated actuator angles; never integrate MuJoCo forces.

Uses the existing ExportOptions settings. Video telemetry records the exact
interpolated poses shown in each frame, with SI units in column names.
"""
import csv
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import mujoco
import numpy as np

from src.export.calibrated_section_export import ExportOptions, frame_times


class ChainTelemetry:
    """Validate vector samples and reconstruct their display-time poses."""

    def __init__(self, samples, module_count):
        self.count = module_count
        if not samples or module_count < 1:
            raise ValueError('Export requires samples and at least one actuator')
        self.clocks = np.asarray([s.time_s for s in samples], dtype=float)
        self.angles = np.asarray([s.angles_rad for s in samples], dtype=float)
        self.commands = np.asarray([s.commands for s in samples], dtype=float)
        shape = (len(samples), module_count)
        self.columns = {
            name: np.asarray([[getattr(a, name) for a in s.actuator_samples]
                              for s in samples], dtype=float)
            for name in ('pump_flow_m3_s', 'leak_flow_m3_s', 'volume_m3')}
        arrays = [self.angles, self.commands, *self.columns.values()]
        if (not np.all(np.isfinite(self.clocks))
                or np.any(np.diff(self.clocks) <= 0)
                or any(a.shape != shape or not np.all(np.isfinite(a)) for a in arrays)):
            raise ValueError('Chain samples require finite module values and strictly increasing times')
        self.rates = np.diff(self.angles, axis=0)/np.diff(self.clocks)[:, None]

    def at(self, time_s):
        """Interpolate angles; hold input values over their completed interval.

        The sample at t describes the integration interval ending at t. As in
        the single-section exporter, the first interval supplies initial inputs.
        """
        clock = float(time_s)
        if not math.isfinite(clock) or not self.clocks[0] <= clock <= self.clocks[-1]:
            raise ValueError('Frame time must lie inside the calculated trajectory')
        interval = int(np.clip(np.searchsorted(self.clocks, clock, side='left')-1,
                               0, max(0, len(self.clocks)-2)))
        end = min(interval+1, len(self.clocks)-1)

        def interpolate(array):
            return np.array([np.interp(clock, self.clocks, array[:, i])
                             for i in range(self.count)])

        result = {name: interpolate(array) for name, array in self.columns.items()
                  if name != 'pump_flow_m3_s'}
        result.update(time_s=clock, angles_rad=interpolate(self.angles),
                      angle_velocities_rad_s=(self.rates[interval] if len(self.rates)
                                              else np.zeros(self.count)),
                      commands=self.commands[end],
                      pump_flow_m3_s=self.columns['pump_flow_m3_s'][end])
        result['net_flow_m3_s'] = result['pump_flow_m3_s']-result['leak_flow_m3_s']
        return result

    def row(self, frame, values):
        """Flatten one rendered instant into explicitly named SI columns."""
        row = dict(frame=frame, time_s=values['time_s'])
        names = dict(angles_rad='angle_rad', angle_velocities_rad_s='angle_velocity_rad_s',
                     commands='command', pump_flow_m3_s='pump_flow_m3_s',
                     leak_flow_m3_s='leak_flow_m3_s', net_flow_m3_s='net_flow_m3_s',
                     volume_m3='volume_m3')
        for i in range(self.count):
            for source, label in names.items():
                row[f'actuator_{i+1}_{label}'] = float(values[source][i])
        return row


def chain_camera(chain):
    """A whole-chain view: head at left, tail at right in the straight pose."""
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    length = float(chain.total_length_m)
    if not math.isfinite(length) or length <= 0:
        raise ValueError('Chain length must be finite and positive')
    camera.lookat[:] = [.46*length, 0, 0]
    # Leave vertical room for the wave envelope and the telemetry panels.
    # Fitting only the straight length can hide the bent tail behind the title.
    camera.distance = max(.3, 1.6*length)
    camera.azimuth, camera.elevation = 90, -70
    return camera


class ChainOverlay:
    """A header and colored actuator cards leave the center for the geometry."""

    def __init__(self, width, height, colors, label):
        from PIL import ImageFont
        self.width, self.height, self.label = width, height, label
        self.scale = min(width/1920, height/1080)
        self.colors = colors
        self.fonts = []
        for size, bold in ((34, True), (24, False), (25, True), (23, False)):
            choices = [Path('/System/Library/Fonts/Supplemental')/
                       ('Arial Bold.ttf' if bold else 'Arial.ttf'),
                       Path('/usr/share/fonts/truetype/dejavu')/
                       ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')]
            path = next((p for p in choices if p.is_file()), None)
            pixels = max(8, round(size*self.scale))
            self.fonts.append(ImageFont.truetype(str(path), pixels) if path else
                              ImageFont.load_default(size=pixels))

    def draw(self, pixels, values):
        from PIL import Image, ImageDraw
        image = Image.fromarray(pixels).convert('RGBA')
        panel = Image.new('RGBA', image.size)
        draw = ImageDraw.Draw(panel)
        s, pad = self.scale, max(5, round(24*self.scale))
        title, note, card_title, font = self.fonts
        dark, white = (17, 29, 43, 236), (245, 248, 252)
        draw.rounded_rectangle((pad, pad, self.width-pad, round(150*s)),
                               radius=max(3, round(12*s)), fill=dark)
        draw.text((2*pad, round(35*s)), 'TRAVELING WAVE: LINKED ACTUATORS', font=title, fill=white)
        draw.text((2*pad, round(83*s)), 'Head → tail: phase-delayed independent actuators',
                  font=note, fill=(185, 207, 226))
        label = f"{self.label}  |  t = {values['time_s']:.2f} s"
        draw.text((2*pad, round(116*s)), label, font=note, fill=(185, 207, 226))
        count = len(values['angles_rad'])
        gap = max(2, round(10*s))
        card_width = (self.width-2*pad-gap*(count-1))/count
        top, bottom = self.height-round(177*s), self.height-pad
        for i in range(count):
            left, right = round(pad+i*(card_width+gap)), round(pad+i*(card_width+gap)+card_width)
            rgba = np.asarray(self.colors[i % len(self.colors)], dtype=float)[:3]
            color = tuple(np.clip(rgba*255, 0, 255).astype(int))
            draw.rounded_rectangle((left, top, right, bottom), radius=max(3, round(9*s)), fill=dark)
            draw.rectangle((left, top, right, top+max(3, round(6*s))), fill=color)
            title_color = tuple(round(channel*.65+255*.35) for channel in color)
            rows = [(f'Actuator {i+1}', card_title, title_color),
                    (f"PWM  {values['commands'][i]:.1f}", font, white),
                    (f"Qp  {values['pump_flow_m3_s'][i]*1e6:+.2f} mL/s", font, white),
                    (f"Bend  {math.degrees(values['angles_rad'][i]):+.1f}°", font, white)]
            for j, (text, face, ink) in enumerate(rows):
                draw.text((left+round(13*s), top+round((16+32*j)*s)), text, font=face, fill=ink)
        return np.asarray(Image.alpha_composite(image, panel).convert('RGB'))


def _write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_chain_media(chain, samples, options: ExportOptions):
    """Export MP4/JPG/frames with one fixed camera and synchronized telemetry.

    Angles are prescribed, not force-driven. Temporary video/JPG files protect
    previous exports; all pose and framebuffer settings are restored afterward.
    """
    telemetry = ChainTelemetry(samples, chain.module_count)
    clocks, angles = telemetry.clocks, telemetry.angles
    sequence = options.video_path is not None or options.frames_dir is not None
    times = frame_times(clocks[0], clocks[-1], options.fps) if sequence else []
    snapshot = (float(clocks[np.argmax(np.max(np.abs(angles), axis=1))])
                if options.snapshot_time_s is None else options.snapshot_time_s)
    if options.jpg_path is not None:
        telemetry.at(snapshot)
    ffmpeg = shutil.which('ffmpeg') if options.video_path is not None else None
    if options.video_path is not None and ffmpeg is None:
        raise RuntimeError('MP4 export needs FFmpeg on PATH')
    from PIL import Image
    colors = getattr(chain, 'module_colors', None)
    if colors is None or len(colors) == 0:
        colors = [(.2, .65, .85)]*chain.module_count
    overlay = (ChainOverlay(options.width, options.height, colors, options.overlay_label)
               if options.overlay else None)
    model, data = chain.model, chain.data
    saved = {name: getattr(data, name).copy() for name in
             ('qpos', 'qvel', 'ctrl', 'qfrc_applied', 'xfrc_applied')}
    saved_time = data.time
    vis = model.vis.global_
    original = (vis.offwidth, vis.offheight, model.vis.quality.offsamples)
    vis.offwidth, vis.offheight = max(vis.offwidth, options.width), max(vis.offheight, options.height)
    model.vis.quality.offsamples = max(4, model.vis.quality.offsamples)
    renderer, process = None, None
    temporary = []
    record = dict(status='completed', width=options.width, height=options.height,
                  fps=options.fps, frame_count=len(times), overlay=options.overlay,
                  overlay_label=options.overlay_label, interpolation='Linear angle interpolation',
                  telemetry_units='SI; raw ESC commands; display uses degrees and mL/s',
                  prescribed_motion=True)

    def temp_for(destination, suffix):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=suffix, delete=False) as f:
            path = Path(f.name)
        temporary.append(path)
        return path

    try:
        renderer = mujoco.Renderer(model, height=options.height, width=options.width)
        camera = chain_camera(chain)

        def render(clock):
            values = telemetry.at(float(clock))
            chain.apply(values['angles_rad'], values['time_s'], values['angle_velocities_rad_s'])
            renderer.update_scene(data, camera=camera)
            pixels = renderer.render()
            return (overlay.draw(pixels, values) if overlay else pixels), values

        if options.frames_dir is not None:
            options.frames_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile() as encoder_log:
            video_temp = None
            if options.video_path is not None:
                video_temp = temp_for(options.video_path, '.mp4')
                command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s:v',
                           f'{options.width}x{options.height}', '-r', f'{options.fps:g}',
                           '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow',
                           '-crf', str(options.video_crf), '-pix_fmt', 'yuv420p',
                           '-movflags', '+faststart', str(video_temp)]
                process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                           stdout=subprocess.DEVNULL, stderr=encoder_log)
            rows = []
            for index, clock in enumerate(times):
                pixels, values = render(clock)
                if process is not None:
                    process.stdin.write(np.ascontiguousarray(pixels).tobytes())
                row = telemetry.row(index, values)
                if options.frames_dir is not None:
                    name = f'frame_{index:06d}.jpg'
                    Image.fromarray(pixels).save(options.frames_dir/name,
                                                quality=options.jpg_quality, subsampling=0)
                    row['file'] = name
                rows.append(row)
                if index % max(1, round(options.fps)) == 0:
                    print(f'Export: {index+1}/{len(times)} frames', flush=True)
            if process is not None:
                process.stdin.close()
                if process.wait() != 0:
                    encoder_log.seek(0)
                    raise RuntimeError('FFmpeg failed: '+encoder_log.read().decode(errors='replace'))
            if options.frames_dir is not None:
                _write_csv(options.frames_dir/'frames.csv', rows)
            jpg_temp = None
            if options.jpg_path is not None:
                pixels, values = render(snapshot)
                jpg_temp = temp_for(options.jpg_path, '.jpg')
                Image.fromarray(pixels).save(jpg_temp, quality=options.jpg_quality, subsampling=0)
                record.update(snapshot_time_s=snapshot, snapshot_angles_rad=values['angles_rad'].tolist())
            # Publish only after every requested frame has rendered successfully.
            if video_temp is not None:
                csv_path = options.video_path.with_name(options.video_path.stem+'_telemetry.csv')
                csv_temp = temp_for(csv_path, '.csv')
                _write_csv(csv_temp, rows)
                video_temp.replace(options.video_path)
                csv_temp.replace(csv_path)
                record.update(video_path=str(options.video_path), video_telemetry_csv=str(csv_path),
                              video_duration_s=len(times)/options.fps)
            if jpg_temp is not None:
                jpg_temp.replace(options.jpg_path)
                record['jpg_path'] = str(options.jpg_path)
            if options.frames_dir is not None:
                record['frames_dir'] = str(options.frames_dir)
        return record
    except BrokenPipeError as exc:
        raise RuntimeError('FFmpeg stopped while encoding the chain video') from exc
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                process.wait()
            if process.stdin is not None and not process.stdin.closed:
                try:
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    # A failed encoder must not prevent pose/GPU cleanup below.
                    pass
        if renderer is not None:
            renderer.close()
        for path in temporary:
            path.unlink(missing_ok=True)
        vis.offwidth, vis.offheight, model.vis.quality.offsamples = original
        for name, value in saved.items():
            getattr(data, name)[:] = value
        data.time = saved_time
        mujoco.mj_forward(model, data)


def playback_chain(chain, samples):
    """Interactive real-time replay; macOS launches this through mjpython."""
    import mujoco.viewer
    telemetry = ChainTelemetry(samples, chain.module_count)
    camera = chain_camera(chain)
    first, last = telemetry.clocks[0], telemetry.clocks[-1]
    values = telemetry.at(first)
    chain.apply(values['angles_rad'], first, values['angle_velocities_rad_s'])
    with mujoco.viewer.launch_passive(chain.model, chain.data) as viewer:
        viewer.cam.lookat[:] = camera.lookat
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = (
            camera.distance, camera.azimuth, camera.elevation)
        start = time.monotonic()
        while viewer.is_running():
            clock = min(last, first+time.monotonic()-start)
            values = telemetry.at(clock)
            with viewer.lock():
                chain.apply(values['angles_rad'], clock, values['angle_velocities_rad_s'])
            viewer.sync()
            if clock >= last:
                break
            time.sleep(1/60)
