import js from "@eslint/js";
import tseslint from "typescript-eslint";
import globals from "globals";

export default tseslint.config(
  {
    ignores: [
      "web/dist/",
      "target/",
      "pipeline/",
      "vendor/",
      ".venv/",
      "runs/",
      "**/*.d.ts",
      "**/*.config.*",
      "node_modules/",
    ],
  },
  {
    files: ["web/src/**/*.ts"],
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    languageOptions: { globals: globals.browser },
  },
  {
    files: ["scripts/**/*.mjs"],
    extends: [js.configs.recommended],
    // Playwright callbacks execute in the page, so browser globals are valid too.
    languageOptions: { globals: { ...globals.node, ...globals.browser } },
  },
);
