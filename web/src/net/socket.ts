// WebSocket transport for the viewer. Kept separate from the frame/UI code: it owns the
// socket lifecycle and reconnect loop, and hands raw parsed messages to whatever callbacks
// the boot file supplies.

import { els, setText } from "../dom";
import type { Room } from "../scene/world";
import type { Frame, Metadata } from "../types";

export type SocketHandlers = {
  onMetadata(message: Metadata): void;
  onFrame(message: Frame): void;
  onRoom(message: Room): void;
};

let socket: WebSocket;
let handlers: SocketHandlers;

export function send(message: object): void {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
}

export function connect(next: SocketHandlers): void {
  handlers = next;
  open();
}

function open(): void {
  socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  socket.onmessage = (e) => {
    const message = JSON.parse(e.data);
    if (message.type === "metadata") handlers.onMetadata(message);
    else if (message.type === "frame") handlers.onFrame(message);
    else if (message.type === "room") handlers.onRoom(message);
    else if (message.error) setText(els.error, message.error);
  };
  socket.onclose = () => {
    setText(els.status, "Disconnected · reconnecting");
    els.dot.classList.remove("live");
    setTimeout(open, 1500);
  };
  socket.onerror = () => {
    setText(els.error, "Start the local service: fly-drone serve");
  };
}
