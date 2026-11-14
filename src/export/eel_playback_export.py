"""Viewer and synchronized CAD video/state export for angle-only playback."""
import csv
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import mujoco
import numpy as np

from src.hydrodynamics.eel_fluid import ellipsoid_forces, inertia_box_forces

FORCE_ARROW_SCALE_M_PER_N = .8
INERTIA_FORCE_ARROW_SCALE_M_PER_N = 40.
VELOCITY_ARROW_SCALE_S = 1.0


def draw_velocity_arrow(playback, scene, scale=VELOCITY_ARROW_SCALE_S):
    """Draw the overall COM velocity vector of the eel in the scene."""
    if scene.ngeom >= scene.maxgeom:
        return
    body = (playback.model.jnt_bodyid[playback.chain.root_joint_id]
            if playback.chain.root_joint_id is not None else 0)
    origin = playback.data.subtree_com[body].copy()
    vel = playback.data.subtree_linvel[body]
    speed = np.linalg.norm(vel)
    if speed < 1e-4:
        return
    # Elevate the arrow slightly above the eel so it stays visible.
    origin[2] += .08
    geom = scene.geoms[scene.ngeom]
    color = np.array([.12, .85, .28, 1.], dtype=np.float32)
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), origin,
                        np.eye(3).ravel(), color)
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .004, origin,
                         origin + scale * vel)
    scene.ngeom += 1


def draw_force_arrows(playback, scene, scale=FORCE_ARROW_SCALE_M_PER_N, *,
                      inertia_boxes=False, all_forces=False):
    """Draw local force vectors using a fixed scale across frames."""
    if inertia_boxes:
        ids, forces = inertia_box_forces(playback)
        origins = playback.data.xipos[ids].copy()
        # The generated eel has an 80 mm vertical half-height. Lift the
        # arrows into view while retaining each force's direction and length.
        origins[:, 2] += .095
    else:
        ids, forces = ellipsoid_forces(playback)
        origins = playback.data.geom_xpos[ids].copy()
        up = playback.data.geom_xmat[ids, 6:9]
        origins[:, 2] += np.linalg.norm(up*playback.model.geom_size[ids], axis=1) + .015
    magnitudes = np.linalg.norm(forces, axis=1)
    indices = range(len(forces)) if all_forces else np.argsort(-magnitudes)[:5]
    for i in indices:
        if scene.ngeom >= scene.maxgeom:
            break
        if magnitudes[i] < (1e-12 if all_forces else 1e-6):
            continue
        origin = origins[i]
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), origin,
                            np.eye(3).ravel(), np.array([1., .49, .05, 1.], dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .003, origin,
                             origin+scale*forces[i])
        scene.ngeom += 1


def camera_for(playback):
    c = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(c)
    c.lookat[:] = playback.data.site_xpos[playback.chain.base_site_ids[0]]
    c.lookat[0] += playback.chain.total_length_m*.46
    c.distance = max(.4, playback.chain.total_length_m*1.8)
    # Include the mounted ends in the initial camera framing.
    ends = {'module_0_head', f'module_{playback.chain.module_count-1}_tail'}
    end_positions = []
    for i in range(playback.model.ngeom):
        if playback.model.geom(i).name in ends:
            radius = playback.model.geom_rbound[i]
            center = playback.data.geom_xpos[i, 0]
            end_positions.extend((center-radius, center+radius))
    if end_positions:
        base_x = playback.data.site_xpos[playback.chain.base_site_ids[0], 0]
        left = min(base_x, *end_positions)
        right = max(base_x+playback.chain.total_length_m,
                    playback.data.site_xpos[playback.chain.tip_site_ids[-1], 0], *end_positions)
        c.lookat[0] = (left+right)/2
        c.distance = max(c.distance, (right-left)*1.8)
    c.azimuth, c.elevation = 90., -65.
    if playback.pose_trajectory is not None:
        c.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        c.trackbodyid = playback.model.jnt_bodyid[playback.chain.root_joint_id]
    return c


def frame_times(playback, fps=30., duration_s=None):
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive and finite')
    first, last = playback.trajectory.times_s[[0, -1]]
    if duration_s is not None:
        if not np.isfinite(duration_s) or duration_s <= 0 or duration_s > last-first:
            raise ValueError('duration_s must be positive and within trajectory')
        last = first+duration_s
    return first+np.arange(int(np.floor((last-first)*fps+1e-10))+1)/fps


def state_rows(frame, state, start_positions=None):
    """Long-form SI rows: every base, tip, soft body, and attached CAD geom."""
    for section in state.sections:
        for kind, objects in [('base', (section.base,)), ('tip', (section.tip,)),
                              ('body', section.bodies), ('geom', section.geoms)]:
            for obj in objects:
                row = dict(frame=frame, time_s=state.time_s, section_index=section.index,
                           bend_rad=section.bend_rad, bend_rate_rad_s=section.bend_rate_rad_s,
                           interpolated_gap=int(state.interpolated_gap), kind=kind, name=obj.name)
                for prefix, values, labels in [
                    ('position', obj.position_m, ['x_m','y_m','z_m']),
                    ('quaternion', obj.quaternion_wxyz, ['w','x','y','z']),
                    ('linear_velocity', obj.linear_velocity_m_s, ['x_m_s','y_m_s','z_m_s']),
                    ('angular_velocity', obj.angular_velocity_rad_s, ['x_rad_s','y_rad_s','z_rad_s'])]:
                    row.update({prefix+'_'+k: float(v) for k, v in zip(labels, values)})
                if start_positions is not None:
                    p0 = start_positions.get((kind, obj.name), obj.position_m)
                    row.update({
                        'displacement_x_m': float(obj.position_m[0] - p0[0]),
                        'displacement_y_m': float(obj.position_m[1] - p0[1]),
                        'displacement_z_m': float(obj.position_m[2] - p0[2]),
                    })
                yield row


def export_playback(playback, output_dir, *, video=False, fps=30., duration_s=None,
                    velocity_arrow=True, velocity_arrow_scale=VELOCITY_ARROW_SCALE_S,
                    force_arrows=False, force_arrow_scale=FORCE_ARROW_SCALE_M_PER_N,
                    inertia_boxes=False, all_forces=False, ellipsoid_view=False,
                    width=1920, height=1080):
    """Same evaluated snapshot supplies both video frame and state CSV rows.

    Publish completed temporary files only; leave prior exports intact on error.
    Renderer is never created for data-only exports.
    """
    from PIL import Image, ImageDraw, ImageFont
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    times = frame_times(playback, fps, duration_s)
    encoder = shutil.which('ffmpeg') or '/opt/homebrew/bin/ffmpeg'
    renderer = process = None
    vis = playback.model.vis.global_
    original_size = (vis.offwidth, vis.offheight)
    body = (playback.model.jnt_bodyid[playback.chain.root_joint_id]
            if playback.chain.root_joint_id is not None else 0)
    s0 = playback.at(float(times[0]))
    start_com = playback.data.subtree_com[body].copy()
    start_positions = {}
    for section in s0.sections:
        for kind, objects in [('base', (section.base,)), ('tip', (section.tip,)),
                              ('body', section.bodies), ('geom', section.geoms)]:
            for obj in objects:
                start_positions[(kind, obj.name)] = obj.position_m.copy()
    try:
        with tempfile.TemporaryDirectory(dir=output_dir) as temp, tempfile.TemporaryFile() as log:
            temp = Path(temp)
            if video:
                if width < 2 or height < 2 or width % 2 or height % 2:
                    raise ValueError('Video dimensions must be positive even integers')
                vis.offwidth, vis.offheight = max(width, vis.offwidth), max(height, vis.offheight)
                renderer = mujoco.Renderer(playback.model, height=height, width=width)
                playback.at(times[0]); camera = camera_for(playback)
                follow_com = camera.type == mujoco.mjtCamera.mjCAMERA_TRACKING
                if follow_com:
                    # Set the framing explicitly on every frame. MuJoCo's tracking
                    # camera eases from its initial target during the opening frames.
                    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                    camera_offset = camera.lookat.copy() - start_com
                scene_option = mujoco.MjvOption()
                if ellipsoid_view:
                    scene_option.geomgroup[:] = 0
                    scene_option.geomgroup[3] = 1
                    scene_option.sitegroup[:] = 0
                    camera.elevation = -90.
                process = subprocess.Popen([encoder, '-v', 'error', '-y', '-f', 'rawvideo',
                    '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
                    '-an', '-c:v', 'libx264', '-crf', '19', '-preset', 'fast', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(temp/'motion.mp4')], stdin=subprocess.PIPE, stderr=log)
                font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', max(16, width//64))
            with (temp/'state.csv').open('w', newline='') as csvfile, (temp/'hinges.csv').open('w', newline='') as hf:
                writer, hinges = None, csv.writer(hf)
                hinges.writerow(['frame', 'time_s', *[n+'_rad' for n in playback.chain.joint_names],
                                 *[n+'_rad_s' for n in playback.chain.joint_names]])
                for frame, clock in enumerate(times):
                    state = playback.at(float(clock))
                    for row in state_rows(frame, state, start_positions=start_positions):
                        if writer is None:
                            writer = csv.DictWriter(csvfile, fieldnames=list(row)); writer.writeheader()
                        writer.writerow(row)
                    hinges.writerow([frame, clock, *state.hinge_angles_rad, *state.hinge_rates_rad_s])
                    if video:
                        if follow_com:
                            camera.lookat[:] = playback.data.subtree_com[body] + camera_offset
                        renderer.update_scene(playback.data, camera=camera, scene_option=scene_option)
                        if velocity_arrow:
                            draw_velocity_arrow(playback, renderer.scene, velocity_arrow_scale)
                        if force_arrows:
                            draw_force_arrows(playback, renderer.scene, force_arrow_scale,
                                              inertia_boxes=inertia_boxes, all_forces=all_forces)
                        image = Image.fromarray(renderer.render())
                        draw = ImageDraw.Draw(image)
                        draw.rectangle((0, 0, width, int(height*.15)), fill='#233344')
                        captions = []
                        if velocity_arrow:
                            captions.append(f'velocity arrow (green) · {velocity_arrow_scale:g} s')
                        if force_arrows:
                            selection = 'all' if all_forces else 'top 5'
                            captions.append(f'{selection} fluid forces (orange) · {force_arrow_scale:g} m/N')
                        caption = ' · '.join(captions) if captions else 'recorded bends'
                        draw.text((24, 12), f'ANGLE-DRIVEN EEL | {clock:.3f} s | {caption}', font=font, fill='white')
                        text = '  '.join(f'S{i+1}: {np.rad2deg(b):+.1f} deg' for i, b in enumerate(state.bends_rad))
                        draw.text((24, 16+font.size), text, font=font, fill='#91d9e8')

                        com = playback.data.subtree_com[body]
                        com_vel = playback.data.subtree_linvel[body]
                        dx = start_com[0] - com[0]
                        dy = com[1] - start_com[1]
                        speed = np.linalg.norm(com_vel)

                        kinematics_text = (
                            f'COM Δx (headward +): {dx:+.3f} m   |   ΔY (world): {dy:+.3f} m   |   '
                            f'Headward Vx: {-com_vel[0]:+.3f} m/s   |   Speed: {speed:.3f} m/s'
                        )
                        draw.text((24, 20+2*font.size), kinematics_text, font=font, fill='#7eed96')

                        if state.interpolated_gap:
                            draw.text((24, height-font.size-18), 'Tracking gap: interpolated', font=font, fill='#bd481c')
                        process.stdin.write(np.asarray(image).tobytes())
                        if frame == min(150, len(times)-1):
                            image.save(temp/'snapshot.jpg', quality=95)
                    if frame % 150 == 0:
                        print(f'Export {frame+1}/{len(times)}', flush=True)
            if process:
                process.stdin.close()
                if process.wait() != 0:
                    log.seek(0); raise RuntimeError(log.read().decode(errors='replace'))
            for file in temp.iterdir():
                file.replace(output_dir/file.name)
        return dict(frame_count=len(times), fps=fps, first_time_s=float(times[0]),
                    last_time_s=float(times[-1]), video_duration_s=len(times)/fps if video else None,
                    velocity_arrow=velocity_arrow,
                    velocity_arrow_scale_s=velocity_arrow_scale if velocity_arrow else None)
    finally:
        if process and process.poll() is None:
            process.terminate(); process.wait()
        if process and process.stdin and not process.stdin.closed:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        if renderer:
            renderer.close()
        vis.offwidth, vis.offheight = original_size


def view_playback(playback, *, velocity_arrow=True, velocity_arrow_scale=VELOCITY_ARROW_SCALE_S,
                  force_arrows=False, force_arrow_scale=FORCE_ARROW_SCALE_M_PER_N,
                  inertia_boxes=False, all_forces=False):
    """Free-camera passive viewer; use mjpython on macOS."""
    import mujoco.viewer
    first, last = playback.trajectory.times_s[[0, -1]]
    playback.at(float(first)); camera = camera_for(playback)
    with mujoco.viewer.launch_passive(playback.model, playback.data) as viewer:
        viewer.cam.lookat[:] = camera.lookat
        viewer.cam.type, viewer.cam.trackbodyid = camera.type, camera.trackbodyid
        viewer.cam.distance = camera.distance
        viewer.cam.azimuth, viewer.cam.elevation = camera.azimuth, camera.elevation
        start = time.monotonic()
        while viewer.is_running():
            clock = min(last, first+time.monotonic()-start)
            with viewer.lock():
                playback.at(float(clock))
                if velocity_arrow or force_arrows:
                    viewer.user_scn.ngeom = 0
                    if velocity_arrow:
                        draw_velocity_arrow(playback, viewer.user_scn, velocity_arrow_scale)
                    if force_arrows:
                        draw_force_arrows(playback, viewer.user_scn, force_arrow_scale,
                                          inertia_boxes=inertia_boxes, all_forces=all_forces)
            viewer.sync()
            if clock == last:
                break
            time.sleep(1/60)
