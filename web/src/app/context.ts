import type { Stage } from "../scene/stage";
import type { Frame } from "../types";

// Shared viewer context: the scene refs plus the last frame and the socket send function.
// Passed explicitly to the frame logic, controls and render loop so none of them reach for
// module-level singletons.
export type Viewer = {
  stage: Stage;
  latest: Frame | undefined;
  send: (message: object) => void;
};

export function createViewer(stage: Stage, send: (message: object) => void): Viewer {
  return { stage, latest: undefined, send };
}
