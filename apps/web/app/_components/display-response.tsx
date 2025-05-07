"use client";

import { useSuspenseQuery } from "@tanstack/react-query";
import { sayHelloGetOptions } from "../../lib/open-api/@tanstack/react-query.gen";

export default function DisplayResponse() {
  const { data } = useSuspenseQuery(sayHelloGetOptions());
  return <div>{JSON.stringify(data)}</div>;
}
