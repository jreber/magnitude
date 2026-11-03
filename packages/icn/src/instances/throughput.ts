import { Effect } from "effect"
import type { IcnClientService } from "../client.js"
import type { IcnInstancesService } from "./index.js"
import type { ModelInstanceThroughput } from "@magnitudedev/icn-protocol/schemas"

export type IcnThroughputResult =
  | { readonly _tag: "Ready"; readonly value: ModelInstanceThroughput }
  | { readonly _tag: "Absent" }

export const readThroughput = (
  client: IcnClientService,
  instances: IcnInstancesService,
): Effect.Effect<IcnThroughputResult, Error> =>
  Effect.gen(function* () {
    const snapshot = yield* instances.get
    const candidate = snapshot.instances
      .filter((i) => i.lifecycle._tag === "Ready")
      .at(-1)
    if (candidate === undefined) return { _tag: "Absent" as const }
    const result = yield* client.models.getModelInstanceThroughput({
      path: { instance_id: candidate.id },
    })
    if (result.instanceId !== candidate.id) return { _tag: "Absent" as const }
    return { _tag: "Ready" as const, value: result }
  })
