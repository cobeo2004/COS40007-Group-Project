import type { CreateClientConfig } from "./open-api/client.gen";

export const createClientConfig: CreateClientConfig = (config) => ({
  ...config,
  baseUrl: "http://localhost:8000",
});
