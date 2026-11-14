"""Join complete hydraulic sections end to end for prescribed-motion playback.

Each module keeps its own internal bending hinges. Optional physical rigid ends
are welded to their nearest soft-division bodies. The next module's first body
is welded to the previous module's final face, with no extra interface hinge.
The first module is clamped to the world. This geometry does not add pump
dynamics or surrounding-water forces.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

from src.paths import ROOT

from src.geometry.auto_hydraulic_section import PROFILE, generate_section, number


# Alternating shades within each module reveal slices; the hue identifies the
# independently commanded actuator. Colors repeat for chains longer than six.
MODULE_COLORS = (
    (0.22, 0.42, 0.67),
    (0.12, 0.57, 0.54),
    (0.65, 0.40, 0.21),
    (0.53, 0.38, 0.67),
    (0.57, 0.59, 0.18),
    (0.65, 0.31, 0.40),
)


def _prefix_names(element, prefix):
    """Prefix local names and joint references while retaining the shared mesh."""
    for child in element.iter():
        for attribute in ("name", "joint"):
            if attribute in child.attrib:
                child.set(attribute, prefix + child.get(attribute))


def generate_chain(output, *, n_modules=5, slices_per_module=10,
                   length_mm=120., width_mm=80., height_mm=160., gap_mm=1.,
                   rigid_start_mm=0., rigid_end_mm=0.,
                   mass_kg=0.15, mass_is_measured=False,
                   section_stiffness=0.2, section_damping=0.01,
                   joint_stiffness=None, joint_damping=None, limit_deg=40.,
                   profile=PROFILE, root_freejoint=False, fluid_ellipsoids=False,
                   merge_soft_fluid=False):
    """Write a serial chain XML, its shared mesh assets, and a JSON record.

    ``length_mm`` and ``mass_kg`` describe one complete actuator module. Module
    indices start at zero; internal hinge indices start at one. With rigid ends,
    ``slices_per_module`` counts only divisions of the remaining soft region.
    End pieces are welded to those bodies and add no degrees of freedom. The
    returned dictionary includes ``xml_path`` and ordered ``joint_groups``.
    Set ``fluid_ellipsoids`` to add fluid geoms. With ``merge_soft_fluid``,
    one geom on a middle slice represents the complete soft span.

    Python API defaults retain the earlier uncapped geometry for existing
    investigations. The CLI and current traveling-wave example explicitly use
    26 mm rigid ends, a 105 mm soft region, and zero artificial visual gap.

    Mechanical properties and hollow profile are copied from generate_section.
    They remain placeholders unless measured values are explicitly supplied.
    The intended demo prescribes joint poses with mj_forward; mass, stiffness,
    damping and loads therefore do not determine bending in that demo.
    """
    if isinstance(n_modules, bool) or not isinstance(n_modules, int) or n_modules < 1:
        raise ValueError("n_modules must be an integer >= 1")
    if (isinstance(slices_per_module, bool)
            or not isinstance(slices_per_module, int) or slices_per_module < 2):
        raise ValueError("slices_per_module must be an integer >= 2")

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh_path = output.with_name(output.stem + "_slice.obj")

    # Use the existing generator as the single source of slice dimensions,
    # hollow-profile triangulation, inertia and compliant-joint definitions.
    # Temporary files keep the standalone section artifacts untouched.
    with tempfile.TemporaryDirectory(prefix="hydraulic_chain_") as directory:
        template_path, section = generate_section(
            Path(directory) / "template.xml", n=slices_per_module,
            length_mm=length_mm, width_mm=width_mm, height_mm=height_mm,
            rigid_start_mm=rigid_start_mm, rigid_end_mm=rigid_end_mm,
            gap_mm=gap_mm, mass_kg=mass_kg, mass_is_measured=mass_is_measured,
            section_stiffness=section_stiffness, section_damping=section_damping,
            joint_stiffness=joint_stiffness, joint_damping=joint_damping,
            limit_deg=limit_deg, profile=profile,
        )
        root = ET.parse(template_path).getroot()
        # Every cap may have a different extrusion length and mesh. Copy all
        # assets before the template directory disappears, retaining local XML
        # references so the assembled model can be moved with its mesh files.
        mesh_paths = []
        copied_sources = {}
        for mesh in root.findall("asset/mesh"):
            source = (template_path.parent / mesh.get("file")).resolve()
            if source not in copied_sources:
                suffix = source.stem.removeprefix(template_path.stem)
                if not suffix or suffix == source.stem:
                    suffix = "_" + mesh.get("name", source.stem)
                destination = output.with_name(output.stem + suffix + source.suffix)
                shutil.copyfile(source, destination)
                copied_sources[source] = destination
                mesh_paths.append(str(destination))
            mesh.set("file", copied_sources[source].name)

        # Keep the nested section record as portable as the XML. The template
        # assets are temporary; all saved asset paths must identify their copies.
        section["mesh_files"] = list(mesh_paths)
        section["rigid_mesh_files"] = {
            end: str(copied_sources[Path(path).resolve()])
            for end, path in section.get("rigid_mesh_files", {}).items()
        }

    root.set("model", "hydraulic_actuator_chain")
    root.insert(0, ET.Comment(
        "Prescribed kinematics demo: no external water, contact or pump dynamics. "
        "Mass and compliance remain uncalibrated unless measured inputs are supplied."
    ))
    root.find("visual/global").set("offwidth", "1600")
    root.find("visual/global").set("offheight", "900")
    fixture = root.find("worldbody/body[@name='fixture']")
    template_slice = deepcopy(fixture.find("body"))
    template_base = deepcopy(fixture.find("site[@name='section_base']"))
    for child in list(fixture):
        fixture.remove(child)
    actuator = root.find("actuator")
    sensor = root.find("sensor")
    template_actuators = [deepcopy(item) for item in actuator]
    template_sensors = [deepcopy(item) for item in sensor]
    actuator.clear()
    sensor.clear()

    # The endpoint belongs to the final slice body. Read its actual offset,
    # including any welded rear rigid piece, instead of assuming slice length.
    template_tip = template_slice.find(".//site[@name='section_tip']")
    if template_tip is None:
        raise ValueError("Section template is missing its final section_tip")
    last_face = [float(value) for value in template_tip.get("pos", "0 0 0").split()]
    if root_freejoint:
        ET.SubElement(fixture, "freejoint", name="eel_root")
    fluid_count = 0
    if fluid_ellipsoids:
        half_width = section["width_mm"] / 2000.
        half_height = section["height_mm"] / 2000.
        half_lengths = {
            "slice_mesh": section["slice_length_mm"] / 2000.,
            "rigid_start_mesh": section["rigid_start_mm"] / 2000.,
            "rigid_end_mesh": section["rigid_end_mm"] / 2000.,
        }
        slice_length = section["slice_length_mm"] / 1000.
        gap = section["gap_mm"] / 1000.
        middle_index = (slices_per_module-1)//2
        for i, body in enumerate(template_slice.iter("body")):
            for visual in body.findall("geom"):
                mesh = visual.get("mesh")
                if mesh in half_lengths:
                    half_length = half_lengths[mesh]
                    x, y, z = map(float, visual.get("pos", "0 0 0").split())
                    name = visual.get("name").replace("_visual", "_fluid")
                    if merge_soft_fluid and mesh == "slice_mesh":
                        if i != middle_index:
                            continue
                        name = "soft_span_fluid"
                        half_length = section["soft_length_mm"] / 2000.
                        x -= i*(slice_length+gap)
                    ET.SubElement(body, "geom", name=name,
                                  type="ellipsoid", pos=number([x+half_length, y, z]),
                                  size=number([half_length, half_width, half_height]),
                                  fluidshape="ellipsoid", mass="0", contype="0", conaffinity="0",
                                  group="3", rgba=".1 .65 .85 .35")
                    fluid_count += 1
    parent = fixture
    joint_groups = []
    for module in range(n_modules):
        prefix = f"module_{module}_"
        first_slice = deepcopy(template_slice)
        _prefix_names(first_slice, prefix)
        # The previous final body's origin lies at its hinge midpoint. Advance
        # to its mesh's final face before welding the next module's first slice.
        first_slice.set("pos", number([0., 0., 0.] if module == 0 else last_face))
        base = deepcopy(template_base)
        _prefix_names(base, prefix)
        first_slice.insert(0, base)
        color = MODULE_COLORS[module % len(MODULE_COLORS)]
        bodies = list(first_slice.iter("body"))
        for index, body in enumerate(bodies):
            visual = body.find(f"geom[@name='{prefix}slice_visual_{index}']")
            shade = 0.12 if index % 2 else 0.
            visual.set("rgba", number([min(c + shade, 1.) for c in color] + [1.]))
        parent.append(first_slice)
        parent = bodies[-1]
        for source, destination in ((template_actuators, actuator), (template_sensors, sensor)):
            for item in source:
                copied = deepcopy(item)
                _prefix_names(copied, prefix)
                destination.append(copied)
        joint_groups.append([f"{prefix}bend_{i}" for i in range(1, slices_per_module)])

    ET.indent(root)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    record = {
        "xml_path": str(output),
        "mesh_path": str(mesh_path),
        "mesh_paths": mesh_paths,
        "module_count": n_modules,
        "slices_per_module": slices_per_module,
        "soft_divisions_per_module": slices_per_module,
        "soft_length_m": section.get("soft_length_mm", length_mm) / 1000.,
        "rigid_start_mm": rigid_start_mm,
        "rigid_end_mm": rigid_end_mm,
        "rigid_end_count_per_module": int(rigid_start_mm > 0) + int(rigid_end_mm > 0),
        "rigid_slice_count": n_modules * slices_per_module,
        "actuator_fluid_ellipsoid_count": n_modules * fluid_count,
        "bending_joint_count": n_modules * (slices_per_module - 1),
        "module_length_m": length_mm / 1000.,
        "total_length_m": n_modules * length_mm / 1000.,
        "joint_groups": joint_groups,
        "base_sites": [f"module_{i}_section_base" for i in range(n_modules)],
        "tip_sites": [f"module_{i}_section_tip" for i in range(n_modules)],
        "module_mass_kg": mass_kg,
        "total_mass_kg": n_modules * mass_kg,
        "mass_status": section["mass_status"],
        "mechanical_parameters_status": section["mechanical_parameters_status"],
        "surrounding_fluid_density_kg_m3": 0.,
        "surrounding_fluid_viscosity_Pa_s": 0.,
        "contact_enabled": False,
        "fixed_part": None if root_freejoint else "module_0_slice_0",
        "root_freejoint": bool(root_freejoint),
        "module_connection": "First slice welded to previous module's final face; no interface DOF",
        "rigid_end_attachment": "Front/rear rigid pieces share the nearest soft-division body; no extra joints",
        "motion_mode": ("Prescribed root and hinge poses; caller chooses kinematic refresh" if root_freejoint
                        else "Prescribed joint poses via mj_forward; no mechanical load feedback"),
        "section_parameters": section,
        "limitations": section["limitations"] + [
            "Module boundary is rigid; only internal slice hinges bend",
            "Soft region is assumed to preserve constant curvature internally",
            "A prescribed traveling wave does not demonstrate swimming or thrust",
        ],
    }
    output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "Outputs" / "geometry" / "hydraulic_chain.xml")
    parser.add_argument("--modules", dest="n_modules", type=int, default=5)
    parser.add_argument("--slices-per-module", "--soft-divisions", type=int, default=10,
                        help="Number of soft-region divisions per actuator")
    for name, default in (("length-mm", 157.), ("width-mm", 80.),
                          ("height-mm", 160.), ("gap-mm", 0.),
                          ("rigid-start-mm", 26.), ("rigid-end-mm", 26.)):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--root-freejoint", action="store_true")
    try:
        record = generate_chain(**vars(parser.parse_args()))
    except ValueError as exc:
        parser.error(str(exc))
    print(f"XML: {record['xml_path']}")
    print(f"Modules: {record['module_count']}; internal hinges: {record['bending_joint_count']}")
    print("Root pose prescribed" if record["root_freejoint"] else "First slice clamped")
    print("Module boundaries welded; no surrounding-water forces.")


if __name__ == "__main__":
    main()
