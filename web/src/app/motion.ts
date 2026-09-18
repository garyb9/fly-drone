import * as THREE from "three";

// Server frames arrive at the sim's real-time rate (well under the display refresh), so the
// drone and moving objects are eased toward each new pose instead of snapping. The frame
// logic writes these targets; the render loop eases the scene objects toward them.
export const droneTarget = { pos: new THREE.Vector3(), quat: new THREE.Quaternion() };
export const targetTarget = new THREE.Vector3();
export const obstacleTarget = new THREE.Vector3();
export const POSE_TAU = 0.06;
