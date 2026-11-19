# Piper assets for Dream-1

This directory is self-contained. Keep it intact when moving the deployment.

- `piper_arm.json`: Dream arm description, six-joint rest pose and tool axes.
- `piper_arm.urdf`: one arm in SDK joint coordinates, including the gripper.
- `meshes/`: ten binary STL files, in metres, used for visuals and collision geometry.
- `LICENSE`: retained copyright notices and permission for the robot assets.

The six revolute joints from `base_link` to `link6` are Dream's IK chain.
The fingers are prismatic joints outside that chain; their opening is controlled
separately as a normalized gripper action. A full opening of 0.07 m corresponds
to `finger_left=0.035` and its mimicked `finger_right=-0.035`.

All mesh paths resolve relative to the URDF. The bundle has no ROS package URLs,
external textures, mesh downloads or machine-specific absolute paths.

The current tool offset is a provisional 0.145 m along flange +Z. The stock
finger joints are at 0.1358 m; do not infer a measured TCP from the nominal
offset. Confirm the actual point between your closed fingertips and edit the
JSON if necessary. The approach is +Z and the jaw axis is +Y, matching the
finger joints' transformed sliding axes (the two fingers move in opposite
directions). The tool remains unconfirmed until physically checked.

The official Dream client extracts the base-to-flange chain and discards meshes
for IK. Its custom-arm collision model uses link spheres, not these triangle
meshes. Bundled collision geometry therefore does not imply mesh-accurate,
inter-arm or table collision checking in the inference worker. The configured
per-arm regions and keep-out volumes must match the real fixture.

The preflight checks referenced files before opening Dream. Missing files stop
startup. Tests also parse every STL and load the URDF's visual/collision scenes.
