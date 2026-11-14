"""Build a section with a discretized soft span and optional welded rigid ends.

Run from the project root:
    .venv/bin/python -m src.geometry.auto_hydraulic_section --preview

The CLI defaults to 26 mm rigid + 105 mm soft (10 divisions) + 26 mm rigid.
The importable function retains its old uncapped defaults for existing studies;
pass length_mm=157., rigid_start_mm=26., rigid_end_mm=26., n=10, gap_mm=0.
for the current layout.

This is a mechanical approximation, not a hydraulic actuator model. The first
slice is clamped. Internal chambers follow the supplied front-view image; the
center is solid. Mass, bending stiffness and damping are unmeasured placeholders.
All XML quantities are SI; CLI dimensions are millimeters.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from src.paths import ROOT


PROFILE = ROOT/"Geometry"/"hydraulic_section_profile.json"


def number(values):
    return " ".join(f"{float(v):.12g}" for v in np.atleast_1d(values))


def read_profile(path, width, height):
    source = json.loads(Path(path).read_text())
    loops = []
    for item in source["loops"]:
        points = np.asarray(item["points"], dtype=float) * [width, height]
        following = np.roll(points, -1, axis=0)
        area = np.sum(points[:, 0]*following[:, 1]-points[:, 1]*following[:, 0]) / 2
        # Material is on the left of each oriented boundary.
        if (area > 0) != (item["kind"] == "outer"):
            points = points[::-1]
        loops.append(points)
    return source, loops


def inside(points, polygon):
    """Even/odd polygon containment, evaluated away from the boundary."""
    result = np.zeros(len(points), dtype=bool)
    y, z = points.T
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if a[1] != b[1]:
            crossing = ((a[1] > z) != (b[1] > z))
            result ^= crossing & (y < (b[0]-a[0]) * (z-a[1]) / (b[1]-a[1]) + a[0])
    return result


def delaunay(points):
    """Small 2-D Bowyer-Watson triangulation using NumPy only."""
    normalized = (points-points.mean(axis=0))/np.ptp(points, axis=0).max()
    count = len(points)
    coordinates = np.vstack([normalized, [[-20., -20.], [20., -20.], [0., 20.]]])
    triangles = [(count, count+1, count+2)]
    for index in range(count):
        t = np.asarray(triangles)
        a, b, c = (coordinates[t[:, i]]-coordinates[index] for i in range(3))
        determinant = ((a*a).sum(axis=1)*(b[:, 0]*c[:, 1]-b[:, 1]*c[:, 0])
                       -(b*b).sum(axis=1)*(a[:, 0]*c[:, 1]-a[:, 1]*c[:, 0])
                       +(c*c).sum(axis=1)*(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]))
        bad = determinant > 1e-13
        if not bad.any():
            raise ValueError("Could not triangulate a profile vertex")
        edges = Counter(tuple(sorted((int(f[i]), int(f[(i+1) % 3]))))
                        for f in t[bad] for i in range(3))
        triangles = [tuple(map(int, f)) for f in t[~bad]]
        for (u, v), occurrences in edges.items():
            if occurrences != 1:
                continue
            p, q = coordinates[v]-coordinates[u], coordinates[index]-coordinates[u]
            orientation = p[0]*q[1]-p[1]*q[0]
            if abs(orientation) > 1e-13:
                triangles.append((u, v, index) if orientation > 0 else (v, u, index))
    return np.asarray([t for t in triangles if max(t) < count], dtype=int)


def triangulate(loops):
    """Recover every polygon boundary as a Delaunay edge before clipping holes.

    Splitting missing boundary segments makes the triangulation conforming;
    centroid clipping is safe only after all boundaries have been recovered.
    """
    for _ in range(18):
        vertices = np.concatenate(loops)
        simplices = delaunay(vertices)
        edges = {tuple(sorted((int(t[i]), int(t[(i+1) % 3]))))
                 for t in simplices for i in range(3)}
        refined, offset, missing = [], 0, 0
        for loop in loops:
            points = []
            for i, point in enumerate(loop):
                points.append(point)
                pair = tuple(sorted((offset+i, offset+(i+1) % len(loop))))
                if pair not in edges:
                    points.append((point + loop[(i+1) % len(loop)]) / 2)
                    missing += 1
            refined.append(np.asarray(points))
            offset += len(loop)
        if not missing:
            break
        loops = refined
    else:
        raise ValueError("Could not recover the profile boundaries for meshing")

    triangles = simplices.copy()
    centers = vertices[triangles].mean(axis=1)
    keep = inside(centers, loops[0])
    for hole in loops[1:]:
        keep &= ~inside(centers, hole)
    triangles = triangles[keep]
    for t in triangles:
        a, b = vertices[t[1]]-vertices[t[0]], vertices[t[2]]-vertices[t[0]]
        if a[0]*b[1]-a[1]*b[0] < 0:
            t[1], t[2] = t[2], t[1]
    return vertices, triangles, loops


def cross_section_properties(loops):
    """Area, centroid and central second moments of the hollow cross-section."""
    area = cy = cz = yy = zz = yz = 0.0
    for p in loops:
        q = np.roll(p, -1, axis=0)
        cross = p[:, 0]*q[:, 1] - q[:, 0]*p[:, 1]
        area += cross.sum()/2
        cy += ((p[:, 0]+q[:, 0])*cross).sum()/6
        cz += ((p[:, 1]+q[:, 1])*cross).sum()/6
        yy += ((p[:, 0]**2+p[:, 0]*q[:, 0]+q[:, 0]**2)*cross).sum()/12
        zz += ((p[:, 1]**2+p[:, 1]*q[:, 1]+q[:, 1]**2)*cross).sum()/12
        yz += ((2*p[:, 0]*p[:, 1]+p[:, 0]*q[:, 1]+q[:, 0]*p[:, 1]
                +2*q[:, 0]*q[:, 1])*cross).sum()/24
    if area <= 0:
        raise ValueError("Profile material area must be positive")
    cy, cz = cy/area, cz/area
    return area, cy, cz, yy/area-cy**2, zz/area-cz**2, yz/area-cy*cz


def write_mesh(path, loops, length):
    points, cap, boundaries = triangulate(loops)
    count = len(points)
    vertices = np.vstack([np.column_stack([np.zeros(count), points]),
                          np.column_stack([np.full(count, length), points])])
    faces = [tuple(map(int, t[::-1])) for t in cap]
    faces += [tuple(int(i)+count for i in t) for t in cap]
    offset = 0
    for loop in boundaries:
        for i in range(len(loop)):
            a, b = offset+i, offset+(i+1) % len(loop)
            faces.extend([(a, b, b+count), (a, b+count, a+count)])
        offset += len(loop)
    # Each undirected edge must have exactly two incident faces.
    edge_counts = Counter(tuple(sorted((f[i], f[(i+1) % 3])))
                          for f in faces for i in range(3))
    if set(edge_counts.values()) != {2}:
        raise ValueError("Extruded mesh is not watertight")
    lines = ["# Hollow section slice; x longitudinal, y lateral, z vertical"]
    lines += ["v " + number(v) for v in vertices]
    lines += ["f " + " ".join(str(i+1) for i in f) for f in faces]
    path.write_text("\n".join(lines)+"\n")
    return len(vertices), len(faces)


def generate_section(output, *, n=5, length_mm=120., width_mm=80., height_mm=160.,
                     mass_kg=0.15, mass_is_measured=False, gap_mm=1.,
                     section_stiffness=0.2, section_damping=0.01,
                     joint_stiffness=None, joint_damping=None, limit_deg=40.,
                     profile=PROFILE, rigid_start_mm=0., rigid_end_mm=0.):
    """Write XML, visual meshes, and a JSON parameter/assumption record.

    n discretizes only the soft span, whose total length is length_mm minus
    both rigid ends. Each end geom is welded to its nearest soft-span body;
    it adds no bending joint. Setting both rigid lengths to zero reproduces
    the legacy full-length slice chain. A nonzero gap is visualization spacing
    within the soft span, not a measured flexible-material length.

    With uniform hinges in series, k_joint=(n-1)*k_section preserves the static
    end-angle/torque compliance as n changes. It does not guarantee convergence
    of dynamics. Per-hinge overrides bypass this scaling explicitly.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be an integer >= 1")
    scalars = [length_mm, width_mm, height_mm, mass_kg, gap_mm,
               rigid_start_mm, rigid_end_mm,
               section_stiffness, section_damping, limit_deg]
    if not np.all(np.isfinite(scalars)):
        raise ValueError("All dimensions and mechanical parameters must be finite")
    if min(length_mm, width_mm, height_mm, mass_kg, limit_deg) <= 0:
        raise ValueError("Dimensions, mass and joint limit must be positive")
    if min(gap_mm, rigid_start_mm, rigid_end_mm, section_stiffness, section_damping) < 0:
        raise ValueError("Gap, rigid-end lengths, stiffness and damping cannot be negative")
    for value in [joint_stiffness, joint_damping]:
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError("Joint stiffness/damping must be finite and nonnegative")

    length, width, height, gap, rigid_start, rigid_end = np.array(
        [length_mm, width_mm, height_mm, gap_mm, rigid_start_mm, rigid_end_mm])/1000
    soft_length = length-rigid_start-rigid_end
    if soft_length <= 0:
        raise ValueError("Rigid ends must leave a positive soft-span length")
    if n == 1:
        gap = 0.
    slice_length = (soft_length-(n-1)*gap)/n
    if slice_length <= 0:
        raise ValueError("Too many divisions or too large a gap for the soft-span length")
    k = (n-1)*section_stiffness if joint_stiffness is None else joint_stiffness
    c = (n-1)*section_damping if joint_damping is None else joint_damping
    source, loops = read_profile(profile, width, height)
    area, cy, cz, yy, zz, yz = cross_section_properties(loops)
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh_path = output.with_name(output.stem+"_slice.obj")
    mesh_size = write_mesh(mesh_path, loops, slice_length)
    rigid_mesh_paths = {}
    for end, end_length in (("start", rigid_start), ("end", rigid_end)):
        if end_length:
            path = output.with_name(output.stem+f"_rigid_{end}.obj")
            write_mesh(path, loops, end_length)
            rigid_mesh_paths[end] = path

    root = ET.Element("mujoco", model="single_hydraulic_section")
    root.append(ET.Comment("Mechanical approximation only; mass/stiffness/damping require identification."))
    ET.SubElement(root, "compiler", angle="degree")
    opt = ET.SubElement(root, "option", timestep="0.001", integrator="implicitfast",
                        gravity="0 0 -9.81", density="0", viscosity="0")
    ET.SubElement(opt, "flag", contact="disable")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="800", offheight="640")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient", width="64", height="64",
                  rgb1="0.95 0.96 0.98", rgb2="0.95 0.96 0.98")
    ET.SubElement(asset, "mesh", name="slice_mesh", file=mesh_path.name)
    for end, path in rigid_mesh_paths.items():
        ET.SubElement(asset, "mesh", name=f"rigid_{end}_mesh", file=path.name)
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -0.3 0.5", dir="0 0 -1")
    ET.SubElement(world, "light", pos="-0.3 0.1 0.3", dir="1 0 -1")
    parent = ET.SubElement(world, "body", name="fixture")
    ET.SubElement(parent, "site", name="section_base", pos="0 0 0", size="0.002", rgba="0.1 0.8 0.2 1")
    # Retain one effective density while accounting for the material carried by
    # each body. Actual rigid/soft/liquid densities remain unmeasured. Caps are
    # contiguous with the closest soft division and share that body's inertia.
    material_length = n*slice_length+rigid_start+rigid_end
    for i in range(n):
        # Joint origins are at gap midpoints; the first slice is welded to fixture.
        position = ([0, 0, 0] if i == 0 else
                    [slice_length+(rigid_start+gap/2 if i == 1 else gap), 0, 0])
        body = ET.SubElement(parent, "body", name=f"slice_{i}", pos=number(position))
        mesh_x = rigid_start if i == 0 else gap/2
        body_length = slice_length+(rigid_start if i == 0 else 0.)+(rigid_end if i == n-1 else 0.)
        material_x = 0. if i == 0 else gap/2
        mass = mass_kg*body_length/material_length
        inertia = [mass*(yy+zz), mass*(body_length**2/12+zz),
                   mass*(body_length**2/12+yy), 0., 0., -mass*yz]
        ET.SubElement(body, "inertial", pos=number([material_x+body_length/2, cy, cz]),
                      mass=number(mass), fullinertia=number(inertia))
        if i == 0:
            ET.SubElement(body, "site", name="soft_start", pos=number([mesh_x, 0, 0]),
                          size="0.002", rgba="0.2 0.7 0.9 1")
            if rigid_start:
                ET.SubElement(body, "geom", name="rigid_start_visual", type="mesh", mesh="rigid_start_mesh",
                              pos="0 0 0", rgba="0.76 0.78 0.81 1", contype="0", conaffinity="0", mass="0")
                for side, x in (("outer", 0.), ("inner", rigid_start)):
                    ET.SubElement(body, "site", name=f"rigid_start_{side}", pos=number([x, 0, 0]),
                                  size="0.0015", rgba="0.5 0.5 0.5 1")
        if i:
            ET.SubElement(body, "joint", name=f"bend_{i}", type="hinge", axis="0 0 1",
                          limited="true", range=number([-limit_deg, limit_deg]),
                          stiffness=number(k), damping=number(c), springref="0",
                          solreflimit="0.004 1", solimplimit="0.95 0.99 0.001")
            ET.SubElement(body, "site", name=f"hinge_{i}", type="sphere", size="0.0025",
                          pos=number([0, 0, height/2+0.003]),
                          rgba="1 0.5 0.1 1")
        color = "0.30 0.40 0.53 1" if i % 2 == 0 else "0.42 0.53 0.66 1"
        ET.SubElement(body, "geom", name=f"slice_visual_{i}", type="mesh", mesh="slice_mesh",
                      pos=number([mesh_x, 0, 0]), rgba=color, contype="0", conaffinity="0", mass="0")
        # Two small rigid half-strips illustrate a compliant interface. The actual
        # compliance is the hinge spring/damper, not deformation of these strips.
        if gap:
            if i:
                ET.SubElement(body, "geom", name=f"connector_after_{i}", type="box",
                              pos=number([gap/4, 0, 0]), size=number([gap/4, width*0.10, height*0.18]),
                              rgba="0.95 0.65 0.25 1", mass="0")
            if i < n-1:
                ET.SubElement(body, "geom", name=f"connector_before_{i+1}", type="box",
                              pos=number([mesh_x+slice_length+gap/4, 0, 0]),
                              size=number([gap/4, width*0.10, height*0.18]),
                              rgba="0.95 0.65 0.25 1", mass="0")
        ET.SubElement(body, "site", name=f"marker_{i}", pos=number([mesh_x+slice_length/2, 0, height/2]),
                      size="0.002", rgba="0.9 0.3 0.15 1")
        if i == n-1:
            ET.SubElement(body, "site", name="soft_end", pos=number([mesh_x+slice_length, 0, 0]),
                          size="0.002", rgba="0.2 0.7 0.9 1")
            if rigid_end:
                ET.SubElement(body, "geom", name="rigid_end_visual", type="mesh", mesh="rigid_end_mesh",
                              pos=number([mesh_x+slice_length, 0, 0]), rgba="0.76 0.78 0.81 1",
                              contype="0", conaffinity="0", mass="0")
                for side, x in (("inner", mesh_x+slice_length), ("outer", mesh_x+slice_length+rigid_end)):
                    ET.SubElement(body, "site", name=f"rigid_end_{side}", pos=number([x, 0, 0]),
                                  size="0.0015", rgba="0.5 0.5 0.5 1")
        parent = body
    last_body_tip_offset = mesh_x+slice_length+rigid_end
    ET.SubElement(parent, "site", name="section_tip", pos=number([last_body_tip_offset, 0, 0]),
                  size="0.003", rgba="0.1 0.8 0.2 1")
    if n > 1:
        actuator = ET.SubElement(root, "actuator")
        sensor = ET.SubElement(root, "sensor")
        for i in range(1, n):
            # Joint torque inputs for mechanical tests, not separate physical pumps.
            ET.SubElement(actuator, "motor", name=f"test_torque_{i}", joint=f"bend_{i}", gear="1")
            ET.SubElement(sensor, "jointpos", name=f"angle_{i}", joint=f"bend_{i}")
            ET.SubElement(sensor, "jointvel", name=f"velocity_{i}", joint=f"bend_{i}")
    ET.indent(root)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)

    metadata = dict(n=n, number_of_soft_hinges=n-1, length_mm=length_mm, width_mm=width_mm,
                    height_mm=height_mm, gap_mm=gap*1000, slice_length_mm=slice_length*1000,
                    soft_length_mm=soft_length*1000, soft_divisions=n,
                    nominal_soft_division_length_mm=soft_length*1000/n,
                    rigid_start_mm=rigid_start_mm, rigid_end_mm=rigid_end_mm,
                    rigid_attachment="Rigid start/end geoms welded to first/last soft-division bodies; no added joint",
                    last_body_tip_offset_m=last_body_tip_offset,
                    mesh_files=[str(mesh_path)]+[str(path) for path in rigid_mesh_paths.values()],
                    rigid_mesh_files={end: str(path) for end, path in rigid_mesh_paths.items()},
                    total_mass_kg=mass_kg, mass_status="user supplied" if mass_is_measured else "PLACEHOLDER: not measured",
                    joint_stiffness_Nm_per_rad=k, joint_damping_Nms_per_rad=c,
                    equivalent_section_stiffness_Nm_per_rad=k/(n-1) if n > 1 else None,
                    equivalent_section_damping_Nms_per_rad=c/(n-1) if n > 1 else None,
                    mechanical_parameters_status="Uncalibrated model parameters",
                    bending_axis="z; bending in x-y plane", fixed_part="first slice",
                    center="solid", number_of_openings=len(loops)-1,
                    cross_section_area_m2=area, material_volume_m3=area*material_length,
                    mass_distribution="Uniform effective density across soft-span divisions and rigid ends; includes specified total mass only",
                    mesh_vertices=mesh_size[0], mesh_faces=mesh_size[1],
                    profile_notes=source["description"],
                    limitations=["Opening shapes estimated from image, not CAD dimensions",
                                 "Rigid/soft/liquid density differences are not measured; liquid motion is not modeled",
                                 "No pump, pressure, chamber-volume or hysteresis model",
                                 "Hinges approximate flexibility; visible connectors are rigid half-strips",
                                 "No surrounding-water forces or self-contact"])
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
    return output, metadata


def write_preview(xml_path, metadata):
    """Render straight/end and bent poses. Poses are illustrations, not predictions."""
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    panels = []
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
        footer_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 17)
    except OSError:
        font = footer_font = ImageFont.load_default()
    with mujoco.Renderer(model, height=640, width=800) as renderer:
        for end_view, bend, label in [(True, 0, "End view: two openings, solid center"),
                                      (False, 0, "Straight section"),
                                      (False, 35, "Illustrative bend: +35 degrees"),
                                      (False, -35, "Illustrative bend: -35 degrees")]:
            data.qpos[:] = 0
            if model.nq:
                data.qpos[:] = np.deg2rad(bend)/model.nq
            elif bend:
                label = "Single rigid slice (no bending hinges)"
            mujoco.mj_forward(model, data)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [metadata["length_mm"]/2000, 0, 0]
            camera.distance = 0.36
            camera.azimuth, camera.elevation = (180, 0) if end_view else (145, -18)
            renderer.update_scene(data, camera=camera)
            panel = Image.fromarray(renderer.render().copy())
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, 800, 35), fill=(245, 246, 248))
            draw.text((12, 7), label, fill=(20, 25, 35), font=font)
            panels.append(panel)
    canvas = Image.new("RGB", (1600, 1320), (245, 246, 248))
    for i, panel in enumerate(panels):
        canvas.paste(panel, ((i % 2)*800, (i//2)*640))
    end_label = (f"{metadata.get('rigid_start_mm', 0):g} rigid + {metadata.get('soft_length_mm', metadata['length_mm']):g} soft + "
                 f"{metadata.get('rigid_end_mm', 0):g} rigid mm; gray ends welded")
    ImageDraw.Draw(canvas).text((14, 1292),
        f"{end_label} | {metadata['n']} soft divisions | prescribed bends, no hydraulic dynamics",
        fill=(20, 25, 35), font=footer_font)
    destination = xml_path.with_name(xml_path.stem+"_preview.png")
    canvas.save(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=10, help="Number of soft-span divisions (n-1 compliant hinges)")
    parser.add_argument("--output", type=Path)
    for name, default in [("length-mm", 157.), ("width-mm", 80.), ("height-mm", 160.), ("gap-mm", 0.),
                          ("rigid-start-mm", 26.), ("rigid-end-mm", 26.),
                          ("section-stiffness", 0.2), ("section-damping", 0.01), ("limit-deg", 40.)]:
        parser.add_argument("--"+name, type=float, default=default)
    parser.add_argument("--mass-kg", type=float, default=None, help="Measured total mass; default 0.15 kg is a placeholder")
    parser.add_argument("--joint-stiffness", type=float, help="Override per-interface stiffness (Nm/rad)")
    parser.add_argument("--joint-damping", type=float, help="Override per-interface damping (Nms/rad)")
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--preview", action="store_true", help="Render end/straight/bent pose PNG")
    args = vars(parser.parse_args())
    preview = args.pop("preview")
    args["output"] = args["output"] or ROOT/"Outputs"/"geometry"/f"section_n{args['n']}.xml"
    args["mass_is_measured"] = args["mass_kg"] is not None
    args["mass_kg"] = args["mass_kg"] if args["mass_kg"] is not None else 0.15
    try:
        output, metadata = generate_section(**args)
    except ValueError as exc:
        parser.error(str(exc))
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(output))
    print(f"XML: {output}")
    print(f"Soft divisions: {metadata['n']}; bending hinges: {model.njnt}")
    print(f"Layout: {metadata['rigid_start_mm']:g} mm rigid + {metadata['soft_length_mm']:g} mm soft + "
          f"{metadata['rigid_end_mm']:g} mm rigid")
    print(f"Dimensions: {metadata['length_mm']:g} x {metadata['width_mm']:g} x {metadata['height_mm']:g} mm")
    print(f"Mass: {metadata['total_mass_kg']:g} kg ({metadata['mass_status']})")
    print("Stiffness/damping are uncalibrated. Torque inputs are for mechanical tests, not a pump model.")
    if preview:
        print(f"Preview: {write_preview(output, metadata)}")


if __name__ == "__main__":
    main()
