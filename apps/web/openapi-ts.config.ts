import { defaultPlugins, defineConfig } from "@hey-api/openapi-ts";

export default defineConfig({
  input: "http://localhost:8000/openapi.json",
  output: "lib/open-api",
  logs: {
    path: "./logs",
  },
  plugins: [
    ...defaultPlugins,
    {
      name: "@hey-api/client-next",
      runtimeConfigPath: "./lib/hey-api.ts",
    },
    "@hey-api/client-fetch",
    {
      enums: false,
      name: "@hey-api/typescript",
    },
    "zod",
    {
      name: "@hey-api/sdk",
      validator: true,
    },
    {
      name: "@hey-api/transformers",
      dates: true,
    },
    {
      name: "@hey-api/sdk",
      transformer: true,
    },
    "@tanstack/react-query",
  ],
  // watch: true,
});
