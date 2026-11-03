import { Effect } from "effect"
import type { IcnClientService } from "@magnitudedev/icn"
import type { IcnInstancesService } from "@magnitudedev/icn/instances"
import { readThroughput } from "@magnitudedev/icn/instances"
import type { EngineThroughputSnapshot } from "@magnitudedev/acn-protocol/schemas/serving-usage"

export const readEngineThroughput = (
  client: IcnClientService,
  instances: IcnInstancesService,
): Effect.Effect<EngineThroughputSnapshot> =>
  Effect.gen(function* () {
    const result = yield* readThroughput(client, instances).pipe(
      Effect.map((r) => (r._tag === "Absent" ? null : r.value)),
      Effect.catchAll(() => Effect.succeed(null as null)),
    )
    if (result === null) {
      return { _tag: "Unavailable", message: "no ready model instance" } as EngineThroughputSnapshot
    }
    if (result.state === "warming-up") {
      return {
        _tag: "WarmingUp",
        instanceId: result.instanceId,
        modelId: result.modelId,
        windowMs: result.windowMs,
      } as EngineThroughputSnapshot
    }
    return {
      _tag: "Ready",
      instanceId: result.instanceId,
      modelId: result.modelId,
      windowMs: result.windowMs,
      tokensPerSecond: result.tokensPerSecond,
    } as EngineThroughputSnapshot
  })
