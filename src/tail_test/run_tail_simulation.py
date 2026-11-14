"""Run Tail Testing Simulation and Draw Force Arrows.

This script executes the pivoted fish tail simulation in water, calculates
MuJoCo's exact hydrodynamic drag and lift forces on the fluid geometry, draws
the force arrow directly on the tail, and outputs synchronized telemetry with
each force component in the CSV, high-resolution video, snapshot images, and
analytical plots.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import mujoco

from src.paths import ROOT
from src.tail_test.tail_model import (
    DEFAULT_PIVOT_RADIUS_M,
    DEFAULT_TAIL_HEIGHT_M,
    DEFAULT_TAIL_LENGTH_M,
    DEFAULT_TAIL_WIDTH_M,
    TailSimulator,
)


def draw_scene_force_arrow(scene, sample, scale, color=(1.0, 0.45, 0.05, 1.0),
                           arrow_radius=0.006):
    """Inject a 3D force arrow into the MuJoCo visualization scene."""
    if scene.ngeom >= scene.maxgeom:
        return
    if sample.fluid_force_mag_n < 1e-6:
        return

    origin = sample.arrow_origin.copy()
    force_vec = np.array([sample.fluid_force_x_n, sample.fluid_force_y_n, sample.fluid_force_z_n])
    target = origin + scale * force_vec

    geom = scene.geoms[scene.ngeom]
    col = np.array(color, dtype=np.float32)
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), origin,
                        np.eye(3).ravel(), col)
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, arrow_radius, origin, target)
    scene.ngeom += 1


def setup_camera(sim, view_mode="top"):
    """Configure camera for top-down diagrammatic or 3D perspective framing."""
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat = [0.115, 0.0, 0.0]

    if view_mode == "top":
        cam.distance = 0.52
        cam.azimuth = 90.0
        cam.elevation = -90.0
    else:  # perspective
        cam.distance = 0.55
        cam.azimuth = 120.0
        cam.elevation = -42.0

    return cam


def render_overlay(frame_bgr, sample, scale, view_mode):
    """Add HUD telemetry and explanatory labels onto the rendered frame."""
    h, w, _ = frame_bgr.shape

    # Semi-transparent top HUD bar
    overlay = frame_bgr.copy()
    hud_height = int(h * 0.16)
    cv2.rectangle(overlay, (0, 0), (w, hud_height), (24, 32, 42), -1)
    alpha = 0.88
    cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0, frame_bgr)

    # Title
    cv2.putText(frame_bgr, "TAIL TESTING SIMULATION | Hydrodynamic Force Vector",
                (24, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Kinematics line
    kin_text = (
        f"Time: {sample.time_s:.3f} s   |   "
        f"Angle theta: {sample.angle_deg:+5.1f} deg   |   "
        f"Angular Vel omega: {sample.angular_velocity_deg_s:+6.1f} deg/s"
    )
    cv2.putText(frame_bgr, kin_text, (24, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (220, 200, 100), 2, cv2.LINE_AA)

    # Fluid force telemetry
    thrust_sign = "+" if sample.fluid_force_x_n >= 0 else ""
    force_text = (
        f"Force F: |F| = {sample.fluid_force_mag_n:5.2f} N   |   "
        f"Thrust Fx: {thrust_sign}{sample.fluid_force_x_n:5.2f} N   |   "
        f"Lateral Fy: {sample.fluid_force_y_n:+5.2f} N   |   "
        f"Torque tau: {sample.total_fluid_torque_nm:+5.2f} N*m"
    )
    cv2.putText(frame_bgr, force_text, (24, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (40, 160, 255), 2, cv2.LINE_AA)

    # Scale note
    cv2.putText(frame_bgr, f"Force Arrow (Orange): Scale = {scale:g} m/N",
                (24, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)

    # In top view, label the Pivot and Fluid Geometry matching the user diagram
    if view_mode == "top":
        # Draw explanatory label badges
        # Pivot label
        pivot_px = (int(w * 0.28), int(h * 0.52))
        cv2.circle(frame_bgr, pivot_px, 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(frame_bgr, "Pivot", (pivot_px[0] - 25, pivot_px[1] + 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (30, 30, 30), 2, cv2.LINE_AA)

        # Fluid Geometry label
        geom_px = (int(w * 0.52), int(h * 0.32))
        cv2.putText(frame_bgr, "Fluid Geometry (Ellipsoid)",
                    (geom_px[0] - 110, geom_px[1]),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (160, 90, 20), 2, cv2.LINE_AA)

    return frame_bgr


def generate_diagrammatic_frame(sample, scale, width=1280, height=720):
    """Render a clean 2D vector schematic matching the user's sketch."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Center origin for Pivot
    pivot_x = int(width * 0.25)
    pivot_y = int(height * 0.55)

    # Scale in pixels per meter
    ppm = width * 1.8  # ~2300 pixels per meter

    theta = sample.angle_rad
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Tail length and thickness in pixels
    tail_len_px = DEFAULT_TAIL_LENGTH_M * ppm
    tail_width_px = DEFAULT_TAIL_WIDTH_M * ppm
    pivot_radius_px = int(DEFAULT_PIVOT_RADIUS_M * ppm)

    # Generate ellipse polygon along the rotated axis
    n_pts = 64
    ellipse_angles = np.linspace(0, 2 * np.pi, n_pts)
    a = tail_len_px / 2.0
    b = tail_width_px / 2.0
    c_x = a  # Center offset along the tail
    c_y = 0.0

    poly_pts = []
    for ang in ellipse_angles:
        ex = c_x + a * np.cos(ang)
        ey = c_y + b * np.sin(ang)
        # Rotate by theta (note: image Y is downward, so -sin for counter-clockwise in math)
        rx = pivot_x + (ex * cos_t + ey * sin_t)
        ry = pivot_y - (ex * sin_t - ey * cos_t)
        poly_pts.append([int(rx), int(ry)])

    poly_np = np.array([poly_pts], dtype=np.int32)

    # Draw fluid geometry (fill with light cyan and outline)
    cv2.fillPoly(img, poly_np, (240, 230, 210))
    cv2.polylines(img, poly_np, True, (120, 120, 120), 2, cv2.LINE_AA)

    # Draw spine center line
    tip_x = int(pivot_x + tail_len_px * cos_t)
    tip_y = int(pivot_y - tail_len_px * sin_t)
    cv2.line(img, (pivot_x, pivot_y), (tip_x, tip_y), (180, 160, 140), 1, cv2.LINE_AA)

    # Draw Pivot circle
    cv2.circle(img, (pivot_x, pivot_y), pivot_radius_px, (220, 220, 220), -1, cv2.LINE_AA)
    cv2.circle(img, (pivot_x, pivot_y), pivot_radius_px, (100, 100, 100), 2, cv2.LINE_AA)
    cv2.circle(img, (pivot_x, pivot_y), 4, (40, 40, 220), -1, cv2.LINE_AA)

    # Labels matching the user's uploaded image
    cv2.putText(img, "Pivot", (pivot_x - 35, pivot_y + pivot_radius_px + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2, cv2.LINE_AA)

    # Fluid Geometry label positioned above the body
    geom_label_x = int(pivot_x + (tail_len_px * 0.45) * cos_t - 90)
    geom_label_y = int(pivot_y - (tail_len_px * 0.45) * sin_t - 60)
    cv2.putText(img, "Fluid Geometry", (geom_label_x, geom_label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2, cv2.LINE_AA)

    # Draw Force Arrow from the center of pressure
    # Origin at center of fluid geometry
    arrow_orig_x = int(pivot_x + (tail_len_px * 0.5) * cos_t)
    arrow_orig_y = int(pivot_y - (tail_len_px * 0.5) * sin_t)

    # Force vector in pixels
    f_scale_px = scale * ppm * 0.8
    arrow_dx = sample.fluid_force_x_n * f_scale_px
    arrow_dy = -sample.fluid_force_y_n * f_scale_px

    arrow_end_x = int(arrow_orig_x + arrow_dx)
    arrow_end_y = int(arrow_orig_y + arrow_dy)

    if sample.fluid_force_mag_n > 0.05:
        # Draw thick black arrow as in the user's diagram
        cv2.arrowedLine(img, (arrow_orig_x, arrow_orig_y), (arrow_end_x, arrow_end_y),
                        (20, 20, 20), 4, tipLength=0.15, line_type=cv2.LINE_AA)

        # Label force
        f_lbl = f"F_hydro = {sample.fluid_force_mag_n:.2f} N"
        cv2.putText(img, f_lbl, (arrow_end_x + 15, arrow_end_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 100, 220), 2, cv2.LINE_AA)

    # HUD readout box
    cv2.rectangle(img, (20, 20), (520, 140), (245, 245, 245), -1)
    cv2.rectangle(img, (20, 20), (520, 140), (200, 200, 200), 1)
    cv2.putText(img, f"Time: {sample.time_s:.2f} s | Angle: {sample.angle_deg:+5.1f} deg",
                (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(img, f"Thrust (Fx): {sample.fluid_force_x_n:+5.2f} N",
                (35, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 140, 40), 2, cv2.LINE_AA)
    cv2.putText(img, f"Lateral (Fy): {sample.fluid_force_y_n:+5.2f} N",
                (35, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 60, 200), 2, cv2.LINE_AA)

    return img


def run_simulation(args):
    """Main simulation runner."""
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_path = output_dir / "tail.xml"
    if not xml_path.exists():
        xml_path = ROOT / "Geometry/tail_test/tail.xml"

    print("=" * 70)
    print("TAIL TESTING SIMULATION: Hydrodynamic Flapping & Force Arrow")
    print("=" * 70)
    print(f"Model: {xml_path}")
    print(f"Flapping Frequency: {args.frequency:.2f} Hz | Amplitude: {args.amplitude_deg:.1f} deg")
    print(f"Duration: {args.duration:.2f} s | FPS: {args.fps}")
    print(f"Force Arrow Scale: {args.force_arrow_scale} m/N")
    print(f"Output Directory: {output_dir}")
    print("-" * 70)

    sim = TailSimulator(
        xml_path=xml_path,
        frequency_hz=args.frequency,
        amplitude_deg=args.amplitude_deg,
        force_arrow_scale=args.force_arrow_scale,
    )

    times = np.linspace(0, args.duration, int(args.duration * args.fps) + 1)
    samples = []

    print(f"Evaluating {len(times)} simulation frames with full force decomposition...")
    for t in times:
        sample = sim.evaluate_at(t)
        samples.append(sample)

    # 1. Save Telemetry CSV with each individual force
    csv_path = output_dir / "tail_telemetry.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_s",
            "angle_deg",
            "angle_rad",
            "angular_velocity_deg_s",
            "angular_velocity_rad_s",
            "angular_acceleration_deg_s2",
            "angular_acceleration_rad_s2",

            # Total Net Hydrodynamic Force
            "fluid_force_x_n",
            "fluid_force_y_n",
            "fluid_force_z_n",
            "fluid_force_mag_n",

            # Body / Aerodynamic Frame Forces
            "tail_normal_force_n",
            "tail_axial_force_n",
            "hydrodynamic_lift_n",
            "hydrodynamic_drag_n",

            # Decomposed Hydrodynamic Force Components
            "blunt_drag_force_x_n",
            "blunt_drag_force_y_n",
            "slender_drag_force_x_n",
            "slender_drag_force_y_n",
            "kutta_lift_force_x_n",
            "kutta_lift_force_y_n",
            "magnus_lift_force_x_n",
            "magnus_lift_force_y_n",
            "viscous_base_force_x_n",
            "viscous_base_force_y_n",

            # Torques on Pivot Joint
            "total_fluid_torque_nm",
            "angular_drag_torque_nm",
            "blunt_drag_torque_nm",
            "joint_damping_torque_nm",
            "inertial_torque_nm",
            "actuator_required_torque_nm",

            # Positions
            "geom_x_m",
            "geom_y_m",
            "geom_z_m",
        ])
        for s in samples:
            writer.writerow([
                f"{s.time_s:.4f}",
                f"{s.angle_deg:.4f}",
                f"{s.angle_rad:.6f}",
                f"{s.angular_velocity_deg_s:.4f}",
                f"{s.angular_velocity_rad_s:.6f}",
                f"{s.angular_acceleration_deg_s2:.4f}",
                f"{s.angular_acceleration_rad_s2:.6f}",

                f"{s.fluid_force_x_n:.5f}",
                f"{s.fluid_force_y_n:.5f}",
                f"{s.fluid_force_z_n:.5f}",
                f"{s.fluid_force_mag_n:.5f}",

                f"{s.tail_normal_force_n:.5f}",
                f"{s.tail_axial_force_n:.5f}",
                f"{s.hydrodynamic_lift_n:.5f}",
                f"{s.hydrodynamic_drag_n:.5f}",

                f"{s.blunt_drag_force_x_n:.5f}",
                f"{s.blunt_drag_force_y_n:.5f}",
                f"{s.slender_drag_force_x_n:.5f}",
                f"{s.slender_drag_force_y_n:.5f}",
                f"{s.kutta_lift_force_x_n:.5f}",
                f"{s.kutta_lift_force_y_n:.5f}",
                f"{s.magnus_lift_force_x_n:.5f}",
                f"{s.magnus_lift_force_y_n:.5f}",
                f"{s.viscous_base_force_x_n:.5f}",
                f"{s.viscous_base_force_y_n:.5f}",

                f"{s.total_fluid_torque_nm:.5f}",
                f"{s.angular_drag_torque_nm:.5f}",
                f"{s.blunt_drag_torque_nm:.5f}",
                f"{s.joint_damping_torque_nm:.5f}",
                f"{s.inertial_torque_nm:.5f}",
                f"{s.actuator_required_torque_nm:.5f}",

                f"{s.fluid_geom_pos[0]:.5f}",
                f"{s.fluid_geom_pos[1]:.5f}",
                f"{s.fluid_geom_pos[2]:.5f}",
            ])
    print(f"✓ Saved telemetry CSV with each force: {csv_path.name}")

    # 2. Generate Analysis Plots
    plot_path = output_dir / "tail_forces_plot.png"
    t_arr = np.array([s.time_s for s in samples])
    ang_arr = np.array([s.angle_deg for s in samples])
    vel_arr = np.array([s.angular_velocity_deg_s for s in samples])
    fx_arr = np.array([s.fluid_force_x_n for s in samples])
    fy_arr = np.array([s.fluid_force_y_n for s in samples])
    fmag_arr = np.array([s.fluid_force_mag_n for s in samples])
    lift_arr = np.array([s.hydrodynamic_lift_n for s in samples])
    drag_arr = np.array([s.hydrodynamic_drag_n for s in samples])
    blunt_fy_arr = np.array([s.blunt_drag_force_y_n for s in samples])
    tau_arr = np.array([s.total_fluid_torque_nm for s in samples])
    tau_act_arr = np.array([s.actuator_required_torque_nm for s in samples])

    fig, axs = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    fig.suptitle(f"Tail Testing Hydrodynamics ({args.frequency} Hz, ±{args.amplitude_deg}°)",
                 fontsize=14, fontweight="bold")

    # Panel 1: Kinematics
    ax1 = axs[0]
    ax1.plot(t_arr, ang_arr, "b-", label="Flapping Angle θ (deg)", lw=2)
    ax1.set_ylabel("Angle (deg)", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.grid(True, alpha=0.3)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(t_arr, vel_arr, "g--", label="Angular Velocity ω (deg/s)", lw=1.5, alpha=0.7)
    ax1_twin.set_ylabel("Ang Vel (deg/s)", color="g")
    ax1_twin.tick_params(axis="y", labelcolor="g")
    ax1.set_title("Kinematics: Oscillating Flap around Pivot")

    # Panel 2: Thrust and Lateral Fluid Forces
    axs[1].plot(t_arr, fx_arr, "darkgreen", label="Thrust Force Fx (N)", lw=2)
    axs[1].plot(t_arr, fy_arr, "navy", label="Lateral Force Fy (N)", lw=1.8, ls="--")
    axs[1].plot(t_arr, drag_arr, "crimson", label="Hydro Drag (N)", lw=1.2, ls=":")
    axs[1].axhline(0, color="gray", lw=0.8, ls=":")
    mean_fx = np.mean(fx_arr)
    axs[1].axhline(mean_fx, color="red", lw=1.2, ls="-.", label=f"Mean Thrust: {mean_fx:+.3f} N")
    axs[1].set_ylabel("Force (N)")
    axs[1].set_title("Fluid Forces: Thrust (Fx), Lateral Force (Fy) & Drag")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="upper right", ncol=2)

    # Panel 3: Individual Hydrodynamic Force Decompositions
    axs[2].plot(t_arr, fmag_arr, "coral", label="Total Resultant |F| (N)", lw=2)
    axs[2].plot(t_arr, np.abs(blunt_fy_arr), "purple", label="|Blunt Drag Fy| (N)", lw=1.5, ls="--")
    axs[2].plot(t_arr, lift_arr, "teal", label="Hydro Lift (N)", lw=1.5, ls="-.")
    axs[2].set_ylabel("Magnitude (N)")
    axs[2].set_title("Force Components: Total |F|, Pressure Drag, and Lift")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc="upper right", ncol=3)

    # Panel 4: Torques on Pivot
    axs[3].plot(t_arr, tau_arr, "darkred", label="Fluid Torque τ_fluid (N·m)", lw=2)
    axs[3].plot(t_arr, tau_act_arr, "dimgray", label="Actuator Required Torque τ_act (N·m)", lw=1.5, ls="--")
    axs[3].axhline(0, color="gray", lw=0.8, ls=":")
    axs[3].set_ylabel("Torque (N·m)")
    axs[3].set_xlabel("Time (s)")
    axs[3].set_title("Pivot Torques: Fluid Resistance & Required Actuator Drive")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✓ Saved analytical plots: {plot_path.name}")

    # 3. Render 3D MuJoCo Simulation Video & Snapshots
    video_path = output_dir / "tail_simulation.mp4"
    diag_video_path = output_dir / "tail_diagrammatic.mp4"
    snapshot_path = output_dir / "tail_snapshot.png"
    diagram_snapshot_path = output_dir / "tail_diagram_snapshot.png"

    print("Rendering simulation frames with 3D force arrows...")
    width, height = 1280, 720
    renderer = mujoco.Renderer(sim.model, height=height, width=width)
    camera = setup_camera(sim, view_mode=args.camera)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (width, height))
    diag_writer = cv2.VideoWriter(str(diag_video_path), fourcc, args.fps, (width, height))

    # Representative frame for snapshot: peak forward sweep around t = 0.10s (approx frame 6)
    snapshot_frame_idx = int(0.10 * args.fps)

    for i, sample in enumerate(samples):
        # Update simulation state
        sim.data.qpos[sim.root_qpos:sim.root_qpos+3] = [0, 0, 0]
        sim.data.qpos[sim.root_qpos+3:sim.root_qpos+7] = [1, 0, 0, 0]
        sim.data.qvel[sim.root_dof:sim.root_dof+6] = 0
        sim.data.qpos[sim.pivot_qpos] = sample.angle_rad
        sim.data.qvel[sim.pivot_dof] = sample.angular_velocity_rad_s
        mujoco.mj_fwdPosition(sim.model, sim.data)

        # Update scene
        renderer.update_scene(sim.data, camera=camera)

        # Draw the force arrow directly into the MuJoCo scene!
        draw_scene_force_arrow(renderer.scene, sample, scale=args.force_arrow_scale)

        # Render RGB image from MuJoCo
        rgb = renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Apply HUD overlay
        annotated_bgr = render_overlay(bgr, sample, scale=args.force_arrow_scale,
                                       view_mode=args.camera)
        video_writer.write(annotated_bgr)

        # Also render 2D diagrammatic frame matching the user sketch
        diag_bgr = generate_diagrammatic_frame(sample, scale=args.force_arrow_scale,
                                               width=width, height=height)
        diag_writer.write(diag_bgr)

        # Save snapshots at peak force
        if i == snapshot_frame_idx:
            cv2.imwrite(str(snapshot_path), annotated_bgr)
            cv2.imwrite(str(diagram_snapshot_path), diag_bgr)

    video_writer.release()
    diag_writer.release()
    print(f"✓ Saved 3D simulation video: {video_path.name}")
    print(f"✓ Saved 2D diagrammatic video: {diag_video_path.name}")
    print(f"✓ Saved snapshot image: {snapshot_path.name}")
    print(f"✓ Saved diagram snapshot: {diagram_snapshot_path.name}")

    # 4. Save Run Metadata JSON
    metadata = {
        "simulation": "tail testing",
        "description": "Pivoted fish tail simulation with MuJoCo hydrodynamic drag and lift forces",
        "model": str(xml_path),
        "parameters": {
            "frequency_hz": args.frequency,
            "amplitude_deg": args.amplitude_deg,
            "duration_s": args.duration,
            "fps": args.fps,
            "force_arrow_scale_m_per_n": args.force_arrow_scale,
            "camera_view": args.camera,
        },
        "fluid_properties": {
            "density_kg_m3": 998.0,
            "viscosity_pa_s": 0.001,
            "fluid_coefficients": [0.5, 0.25, 1.5, 1.0, 1.0],
            "fluid_coefficients_legend": ["blunt_drag", "slender_drag", "angular_drag", "kutta_lift", "magnus_lift"],
        },
        "geometry": {
            "tail_length_m": DEFAULT_TAIL_LENGTH_M,
            "tail_height_m": DEFAULT_TAIL_HEIGHT_M,
            "tail_width_m": DEFAULT_TAIL_WIDTH_M,
            "pivot_radius_m": DEFAULT_PIVOT_RADIUS_M,
        },
        "telemetry_csv_columns": [
            "time_s", "angle_deg", "angle_rad", "angular_velocity_deg_s", "angular_velocity_rad_s",
            "angular_acceleration_deg_s2", "angular_acceleration_rad_s2",
            "fluid_force_x_n", "fluid_force_y_n", "fluid_force_z_n", "fluid_force_mag_n",
            "tail_normal_force_n", "tail_axial_force_n", "hydrodynamic_lift_n", "hydrodynamic_drag_n",
            "blunt_drag_force_x_n", "blunt_drag_force_y_n", "slender_drag_force_x_n", "slender_drag_force_y_n",
            "kutta_lift_force_x_n", "kutta_lift_force_y_n", "magnus_lift_force_x_n", "magnus_lift_force_y_n",
            "viscous_base_force_x_n", "viscous_base_force_y_n",
            "total_fluid_torque_nm", "angular_drag_torque_nm", "blunt_drag_torque_nm",
            "joint_damping_torque_nm", "inertial_torque_nm", "actuator_required_torque_nm",
            "geom_x_m", "geom_y_m", "geom_z_m",
        ],
        "results_summary": {
            "max_fluid_force_n": float(np.max(fmag_arr)),
            "mean_thrust_fx_n": float(mean_fx),
            "max_thrust_fx_n": float(np.max(fx_arr)),
            "max_lateral_fy_n": float(np.max(np.abs(fy_arr))),
            "max_torque_nm": float(np.max(np.abs(tau_arr))),
        },
        "artifacts": [
            str(video_path.name),
            str(diag_video_path.name),
            str(snapshot_path.name),
            str(diagram_snapshot_path.name),
            str(plot_path.name),
            str(csv_path.name),
        ],
    }

    run_json_path = output_dir / "run.json"
    with open(run_json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved run metadata: {run_json_path.name}")

    # 5. Interactive Viewer (optional)
    if args.view:
        print("\nOpening interactive rotatable MuJoCo viewer...")
        sim.reset()
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            import time
            start_clock = time.time()
            while viewer.is_running():
                t = (time.time() - start_clock) % args.duration
                sample = sim.evaluate_at(t)
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    draw_scene_force_arrow(viewer.user_scn, sample, scale=args.force_arrow_scale)
                viewer.sync()
                time.sleep(1.0 / args.fps)

    print("\nSimulation complete! All outputs generated successfully in:")
    print(f"  {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Pivoted fish tail simulation with force arrow visualization.")
    parser.add_argument("--duration", type=float, default=4.0, help="Simulation duration in seconds (default: 4.0)")
    parser.add_argument("--frequency", type=float, default=1.0, help="Flapping frequency in Hz (default: 1.0)")
    parser.add_argument("--amplitude-deg", type=float, default=25.0, help="Flapping amplitude in degrees (default: 25.0)")
    parser.add_argument("--fps", type=int, default=60, help="Video and sampling rate in frames per second (default: 60)")
    parser.add_argument("--force-arrow-scale", type=float, default=0.08, help="Scale of force arrow in m/N (default: 0.08)")
    parser.add_argument("--camera", choices=["top", "perspective"], default="top", help="Camera framing (default: top)")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "Outputs/tail_test"), help="Output directory path")
    parser.add_argument("--view", action="store_true", help="Launch interactive 3D viewer")

    args = parser.parse_args()
    run_simulation(args)


if __name__ == "__main__":
    main()
