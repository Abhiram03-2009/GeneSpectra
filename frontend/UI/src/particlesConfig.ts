// frontend/src/particlesConfig.ts
import type { ISourceOptions } from "tsparticles-engine";

export const particlesConfig: ISourceOptions = {
  // The background is now a deep, cybernetic blue to match the inspiration
  background: {
    color: {
      value: "#040c18", // Deep space blue
    },
  },
  fpsLimit: 60,
  interactivity: {
    events: {
      onHover: {
        enable: true,
        mode: "repulse",
      },
      resize: true,
    },
    modes: {
      repulse: {
        distance: 100,
        duration: 0.4,
      },
    },
  },
  particles: {
    // The particle colors are now a mix of glowing blues and cyans
    color: {
      value: ["#0ea5e9", "#06b6d4", "#67e8f9"], // Sky blue, Cyan, Light Cyan
    },
    // The links between particles are a more subtle, techy blue
    links: {
      color: "#3b82f6", // A luminous blue
      distance: 150,
      enable: true,
      opacity: 0.2, // Reduced opacity for a cleaner look
      width: 1,
    },
    move: {
      direction: "none",
      enable: true,
      outModes: {
        default: "out",
      },
      random: true,
      speed: 1.2, // Slightly slower for a more ambient, sophisticated feel
      straight: false,
    },
    number: {
      density: {
        enable: true,
        area: 800,
      },
      value: 90, // Slightly more particles for a denser data field
    },
    opacity: {
      value: { min: 0.3, max: 0.8 }, // Variable opacity for more depth
    },
    shape: {
      type: "circle",
    },
    size: {
      value: { min: 1, max: 3 },
    },
  },
  detectRetina: true,
};