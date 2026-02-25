import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        mono: ['"JetBrains Mono"', '"Fira Code"', "monospace"],
      },
      colors: {
        studio: {
          bg: "#0d1117",
          surface: "#161b22",
          elevated: "#1c2128",
          border: "#30363d",
          text: "#e6edf3",
          muted: "#8b949e",
          accent: "#388bfd",
          green: "#3fb950",
          red: "#f85149",
          yellow: "#e3b341",
        },
      },
    },
  },
  plugins: [],
};

export default config;