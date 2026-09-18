import * as THREE from "three";

// MuJoCo world (Z-up) -> Three.js scene (Y-up) axis mapping, used by every pose conversion.
export const vector = (v: number[]): THREE.Vector3 => new THREE.Vector3(v[0], v[2], -v[1]);
export const WORLD_ROTATION = new THREE.Quaternion().setFromAxisAngle(
  new THREE.Vector3(1, 0, 0),
  -Math.PI / 2,
);

export function vectorInto(target: THREE.Vector3, v: number[]): THREE.Vector3 {
  return target.set(v[0], v[2], -v[1]);
}

// MuJoCo quaternion is (w, x, y, z); Three.js is (x, y, z, w).
export function quaternionInto(target: THREE.Quaternion, v: number[]): THREE.Quaternion {
  return target.set(v[1], v[2], v[3], v[0]);
}
