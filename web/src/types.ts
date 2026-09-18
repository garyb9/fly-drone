import type { Room } from "./scene/world";

export type Cell = {
  id: string;
  type: string;
  side: string;
  group?: number;
  position: number[];
  measured: boolean;
};

export type Metadata = {
  ids: number[];
  cells: Cell[];
  links: number[][];
  groups: string[];
  cameras?: { count: number; splay: number; fovy_deg: number; pos: number[] };
  neurons: number;
  features: number;
  policy: string;
  tasks: string[];
  ablations: string[];
  levels: number[];
  dataset_hash: string;
  room: Room;
  task_policy_status: Record<string, "loaded" | "none">;
};

export type FreeRoamEvent = {
  type: string;
  time?: number;
  side?: number;
  count?: number;
  kinds?: string[];
  min_distance?: number;
  hit?: boolean;
  dodged?: boolean;
};

export type FreeRoam = {
  level: number;
  beacons: number;
  collisions: number;
  collision_kinds: Record<string, number>;
  threats_finished: number;
  threats_dodged: number;
  threats_hit: number;
  visited_cells: number;
  beacon_visible: boolean;
  clearance: number;
  ghost: boolean;
  silenced: string[];
  events: FreeRoamEvent[];
  elapsed: number;
  beacons_per_min: number;
  collisions_per_min: number;
  coverage: number;
};

export type Outcome = {
  bearing: number;
  target_distance: number;
  climb: number;
  obstacle_distance: number;
  launched: boolean;
  displacement: number;
  result: { success: boolean; terminated: boolean; collision: boolean } | null;
};

export type AttributionChannel = {
  types: { type: string; value: number }[];
  cells: number[];
};

export type Attribution = {
  command: number[];
  channels: Record<string, AttributionChannel>;
};

export type Frame = {
  seq: number;
  episode: number;
  tick: number;
  physics_tick: number;
  time: number;
  paused: boolean;
  state: {
    position: number[];
    quaternion: number[];
    velocity: number[];
    angular_velocity?: number[];
    actual_rpm: number[];
    commanded_rpm: number[];
    rotor_phase: number[];
    target: number[];
    obstacle: number[];
  };
  fly: { position: number[]; quaternion: number[]; ticks: number };
  activity: number[];
  readouts: Record<string, number>;
  cues: number[];
  sensory?: Record<string, number>;
  command: number[];
  cameras: string[];
  real_time_factor: number;
  missed_deadlines: number;
  budget?: {
    tick_ms_p50: number | null;
    tick_ms_p95: number | null;
    target_ms: number;
    samples: number;
  };
  error?: string;
  task: string;
  ablation: string;
  seed: number;
  active_policy: string | null;
  policy_status?: "none" | "loaded" | "limits mismatch";
  outcome: Outcome;
  attribution: Attribution | null;
  attribution_seq: number;
  free_roam: FreeRoam | null;
};
