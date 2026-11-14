"""Replay measured section angles with optional swimming dynamics."""
import argparse
import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.geometry.auto_hydraulic_chain import generate_chain
from src.hydrodynamics.eel_fluid import integrate_planar_swim, integrate_straight_swim
from src.kinematics.eel_playback import AngleTrajectory, EelPlayback, HarmonicTrajectory
from src.export.eel_playback_export import (FORCE_ARROW_SCALE_M_PER_N, INERTIA_FORCE_ARROW_SCALE_M_PER_N,
                                 VELOCITY_ARROW_SCALE_S,
                                 export_playback, view_playback)



def attach_end_meshes(model_path, section_count, *, fluid_ellipsoids=True):
    """Mount the supplied millimeter meshes to the generated chain's end slices."""
    tree = ET.parse(model_path)
    root = tree.getroot()
    # Offsets align the measured rear head face and the rectangular tail root.
    # The head also needs a yaw correction to put its nose along the chain's -X.
    mounts = (
        ('head', 'head_rx90.stl', 'module_0_section_base',
         (.0000545457, .04130961, -.086), '0 0 -90'),
        ('tail', 'tail-soft_rx90.stl', f'module_{section_count-1}_section_tip',
         (0., .0005, -.118112507), '0 0 0'),
    )
    for name, filename, site_name, offset, euler in mounts:
        body = next(
            body for body in root.iter('body') if body.find(f"site[@name='{site_name}']") is not None
        )
        site = body.find(f"site[@name='{site_name}']")
        site_pos = [float(value) for value in site.get('pos', '0 0 0').split()]
        pos = ' '.join(str(a+b) for a, b in zip(site_pos, offset))
        source = ROOT/'Geometry'/filename
        asset = model_path.parent/filename
        if source.resolve() != asset.resolve():
            shutil.copyfile(source, asset)
        ET.SubElement(root.find('asset'), 'mesh', name=name, file=filename, scale='.001 .001 .001')
        prefix = site_name.split('section_')[0]
        ET.SubElement(body, 'geom', name=prefix+name, type='mesh', mesh=name, pos=pos, euler=euler,
                      mass='0', contype='0', conaffinity='0', rgba='.28 .34 .41 1')
        if not fluid_ellipsoids:
            continue
        if name == 'head':
            # Keep the nominal 209.19 mm length and 160 mm height. The
            # width matches the STL's projected frontal area (~7988 mm²).
            # The side silhouette centroid sets the longitudinal offset.
            head_fluid_pos = ' '.join(str(value) for value in (site_pos[0]-.09156, *site_pos[1:]))
            ET.SubElement(body, 'geom', name=prefix+'head_fluid', type='ellipsoid',
                          pos=head_fluid_pos, size='.104595 .03178 .08',
                          fluidshape='ellipsoid', mass='0', contype='0', conaffinity='0',
                          group='3', rgba='.1 .65 .85 .35')
        else:
            # The rigid tail mesh spans 230 mm aft of its mount, with a
            # 1 mm thickness and 160 mm height.
            tail_fluid_pos = ' '.join(str(value) for value in (site_pos[0]+.115, *site_pos[1:]))
            ET.SubElement(body, 'geom', name=prefix+'tail_fluid', type='ellipsoid',
                          pos=tail_fluid_pos, size='.115 .0005 .08',
                          fluidshape='ellipsoid', mass='0', contype='0', conaffinity='0',
                          group='3', rgba='.1 .65 .85 .35')
    ET.indent(tree)
    tree.write(model_path, encoding='utf-8', xml_declaration=True)


def attach_inertia_box_visuals(model_path):
    """Keep Group 3 ellipsoids for reference and show drag boxes in Group 4."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    tree = ET.parse(model_path)
    # The inertia mode uses the boxes for drag. Keep the original ellipsoid
    # geometry visible on demand, with its fluid interaction disabled in XML.
    for geom in tree.iter('geom'):
        if geom.get('fluidshape') == 'ellipsoid':
            geom.set('fluidshape', 'none')
    bodies = {body.get('name'): body for body in tree.iter('body')}
    for i in np.flatnonzero(model.body_mass >= mujoco.mjMINVAL):
        name = model.body(i).name
        inertia = model.body_inertia[i]
        # Use the compiled principal axes and the same dimensions as MuJoCo's
        # inertia-box drag, including the engine's lower bound on inertia.
        half_size = np.sqrt(1.5 * np.maximum(mujoco.mjMINVAL, inertia.sum() - 2*inertia)
                            / model.body_mass[i])
        ET.SubElement(bodies[name], 'geom', name=name+'_inertia_box', type='box',
                      pos=' '.join(map(str, model.body_ipos[i])),
                      quat=' '.join(map(str, model.body_iquat[i])),
                      size=' '.join(map(str, half_size)), group='4',
                      rgba='.1 .65 .85 .45', mass='0', contype='0', conaffinity='0',
                      fluidshape='none')
    ET.indent(tree)
    tree.write(model_path, encoding='utf-8', xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True, help='Recording CSV containing timestamps and measured bends')
    parser.add_argument('--model', type=Path, help='Compatible chain XML; omit to generate current CAD sections')
    parser.add_argument('--output-dir', type=Path, help='Omit to use the default folder for this playback mode')
    parser.add_argument('--fit-harmonics', action='store_true',
                        help='Fit 0.3 Hz and 0.9 Hz harmonics over a one-period moving baseline')
    swimming = parser.add_mutually_exclusive_group()
    swimming.add_argument('--swim', action='store_true', help='Integrate forward motion from ellipsoid loads')
    swimming.add_argument('--inertia-swim', action='store_true',
                          help='Use MuJoCo inertia-box loads and predict planar root translation and yaw')
    swimming.add_argument('--planar-swim', action='store_true',
                          help='Use ellipsoid loads and predict planar root translation and yaw')
    parser.add_argument('--mass-kg', type=float,
                        help='Scale the whole-model mass and inertias for planar or inertia-box swimming')
    parser.add_argument('--force-arrows', action='store_true',
                        help='Show the five largest fluid forces in the swim viewer or video')
    parser.add_argument('--all-force-arrows', action='store_true',
                        help='Show every fluid force in the swim viewer or video')
    parser.add_argument('--ellipsoid-view', action='store_true',
                        help='Render the video from above with only Group 3 geometry')
    parser.add_argument('--force-arrow-scale', type=float,
                        help='Visual arrow length in meters per newton (default 0.8 for ellipsoids, 40 for inertia boxes)')
    parser.add_argument('--velocity-arrow', action='store_true', default=True,
                        help='Show the eel COM velocity vector in the swim viewer or video')
    parser.add_argument('--no-velocity-arrow', action='store_false', dest='velocity_arrow',
                        help='Disable drawing the velocity vector arrow')
    parser.add_argument('--velocity-arrow-scale', type=float, default=VELOCITY_ARROW_SCALE_S,
                        help='Visual velocity arrow length in meters per (m/s); default 1.0')
    parser.add_argument('--slice-fluid-geometry', action='store_true',
                        help='Use the earlier one-ellipsoid-per-soft-slice model when generating the chain')
    parser.add_argument('--fluid-dt', type=float, default=.005, help='Maximum swim integration timestep in seconds')
    parser.add_argument('--max-gap-s', type=float, default=.15)
    parser.add_argument('--fps', type=float, default=30.)
    parser.add_argument('--duration', type=float, help='Export this many seconds; default full trajectory')
    parser.add_argument('--video', action='store_true', help='Also render synchronized 1080p MP4')
    parser.add_argument('--view', action='store_true', help='Interactive camera; launch through mjpython on macOS')
    args = parser.parse_args()
    planar_motion = args.planar_swim or args.inertia_swim
    args.force_arrows |= args.all_force_arrows
    if args.ellipsoid_view and not args.video:
        parser.error('--ellipsoid-view requires --video')
    if args.mass_kg is not None and (not planar_motion or not math.isfinite(args.mass_kg) or args.mass_kg <= 0):
        parser.error('--mass-kg requires --planar-swim or --inertia-swim and a positive, finite mass')
    if args.force_arrows and not ((args.swim or planar_motion) and (args.view or args.video)):
        parser.error('--force-arrows requires a swimming mode and either --view or --video')
    if args.force_arrow_scale is None:
        args.force_arrow_scale = (INERTIA_FORCE_ARROW_SCALE_M_PER_N if args.inertia_swim
                                  else FORCE_ARROW_SCALE_M_PER_N)
    if not math.isfinite(args.force_arrow_scale) or args.force_arrow_scale <= 0:
        parser.error('--force-arrow-scale must be positive and finite')
    if not math.isfinite(args.velocity_arrow_scale) or args.velocity_arrow_scale <= 0:
        parser.error('--velocity-arrow-scale must be positive and finite')
    fit_harmonics = args.fit_harmonics
    measured = AngleTrajectory.from_recording_csv(args.csv, max_gap_s=args.max_gap_s,
                                                  allow_camera_angle=fit_harmonics)
    angles = HarmonicTrajectory(measured) if fit_harmonics else measured
    mode_dir = 'recorded_eel_fit' if fit_harmonics else 'recorded_eel'
    if args.swim:
        mode_dir += '_swim'
    elif args.inertia_swim:
        mode_dir += '_inertia_swim'
    elif args.planar_swim:
        mode_dir += '_planar_swim'
    output_dir = args.output_dir or ROOT/'Outputs'/mode_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model = args.model
    if model is None:
        model = output_dir/'eel.xml'
        mass_options = {} if args.mass_kg is None else {'mass_kg': args.mass_kg/angles.section_count}
        generate_chain(model, n_modules=angles.section_count, slices_per_module=10,
                       length_mm=157., rigid_start_mm=26., rigid_end_mm=26., gap_mm=0., root_freejoint=True,
                       fluid_ellipsoids=True,
                       merge_soft_fluid=not args.slice_fluid_geometry,
                       **mass_options)
        attach_end_meshes(model, angles.section_count)
        if args.inertia_swim:
            attach_inertia_box_visuals(model)
    playback = EelPlayback(model, angles)
    if args.swim:
        playback.pose_trajectory, swim = integrate_straight_swim(playback, output_dir, dt_s=args.fluid_dt)
    elif planar_motion:
        playback.pose_trajectory, swim = integrate_planar_swim(
            playback, output_dir, dt_s=args.fluid_dt,
            mass_kg=args.mass_kg if args.model is not None else None,
            ellipsoids=args.planar_swim)
        if args.mass_kg is not None and args.model is None:
            swim['mass_distribution'] = 'equal actuator masses; massless head and tail meshes'
    record = export_playback(playback, output_dir, video=args.video, fps=args.fps, duration_s=args.duration,
                             velocity_arrow=args.velocity_arrow, velocity_arrow_scale=args.velocity_arrow_scale,
                             force_arrows=args.force_arrows, force_arrow_scale=args.force_arrow_scale,
                             inertia_boxes=args.inertia_swim, all_forces=args.all_force_arrows,
                             ellipsoid_view=args.ellipsoid_view)
    record.update(source=str(args.csv.resolve()), source_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest(),
                  model=str(model.resolve()), excluded_rows=measured.excluded_rows, max_gap_s=args.max_gap_s,
                  overall_pose='fixed identity; no inferred head offset', mode='prescribed kinematics only',
                  section_angles='total relative bends; equal internal hinge increments',
                  interpolation='linear; right-hand rates at interior knots, left-hand rate at final sample',
                  units='SI, quaternions wxyz, object velocities world-frame', head_geometry=any(playback.model.geom(i).name == 'module_0_head' for i in range(playback.model.ngeom)),
                  tail_geometry=any(playback.model.geom(i).name == f'module_{angles.section_count-1}_tail'
                                    for i in range(playback.model.ngeom)))
    if args.swim:
        record.update(overall_pose='fixed heading and transverse COM position; integrated longitudinal COM',
                      mode='straight swimming with prescribed bends', swim=swim)
    if planar_motion:
        model_label = 'ellipsoid' if args.planar_swim else 'inertia-box'
        record.update(overall_pose='integrated world X/Y and yaw, fixed depth and level attitude',
                      mode=f'planar swimming with prescribed bends and {model_label} fluid loads', swim=swim)
    if args.swim or planar_motion:
        record['force_arrows'] = args.force_arrows
        if args.force_arrows:
            record['force_arrow_scale_m_per_n'] = args.force_arrow_scale
            record['force_arrow_selection'] = 'all' if args.all_force_arrows else 'top_5'
        record['ellipsoid_view'] = args.ellipsoid_view
        print(f'Swim loads and speed: {output_dir / "swim.csv"}')
    if fit_harmonics:
        record['interpolation'] = 'centered one-period mean plus first and third harmonic fit'
        record['harmonic_fit'] = dict(frequency_hz=angles.frequency_hz, phase_origin_s=angles.phase_origin_s,
                                      basis=['cos(wt)', 'sin(wt)', 'cos(3wt)', 'sin(3wt)'],
                                      coefficients_rad=angles.coefficients_rad.tolist(),
                                      time_range_s=angles.times_s[[0, -1]].tolist())
    (output_dir/'run.json').write_text(json.dumps(record, indent=2)+'\n')
    print(f'States: {output_dir / "state.csv"}')
    if args.view:
        view_playback(playback, velocity_arrow=args.velocity_arrow, velocity_arrow_scale=args.velocity_arrow_scale,
                      force_arrows=args.force_arrows, force_arrow_scale=args.force_arrow_scale,
                      inertia_boxes=args.inertia_swim, all_forces=args.all_force_arrows)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
