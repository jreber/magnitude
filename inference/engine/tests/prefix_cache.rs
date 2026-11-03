//! Real-device qualification of the prefix cache: conversation turns resume
//! at the previous reply, rewritten replies at the previous prompt, arbitrary
//! request sequences at their deepest cached prefix, and requests with images
//! keep their image conditioning across eviction and cached resumption.
//!
//! `MAGNITUDE_TEST_GGUF=M.gguf MAGNITUDE_TEST_MMPROJ=P.gguf
//! [VISION_IMAGE=I.jpeg] cargo test --release -p magnitude-engine
//! --test prefix_cache -- --ignored --nocapture`. The model must accept
//! images through its projector; the image defaults to llama.cpp's
//! `test-1.jpeg`.

use magnitude_chat::{request::ImageInput, ByteBpeTokenizer, SpecialTokens};
use magnitude_engine::{
    build_native_domain,
    composition::{EngineConfiguration, ResolvedEngineConfiguration},
    host::HostArtifacts,
    options::{ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection},
};
use magnitude_executor::{
    platform::{DeviceRequest, ReleaseRate},
    CompletionWake, Demand, ExecutionPath, ExecutorDomain, Operation, Outcome, PhysicalDecision, RequestId,
    ReservedResources, StateBindings, TokenId, WorkKind,
};
use magnitude_family_contracts::{PreparedModelInput, TokenPlan};
use magnitude_generation::{
    EndOfGeneration, Generation, InputLayout, MethodChoice, Options, Sampling, Shaping,
};
use tracing::subscriber::with_default;
use magnitude_scheduler::{
    owner::Owner,
    prefix_cache::{PrefixCacheCapacity, PrefixRetention, MIN_PREFIX_HIT},
    ServiceLimits,
};
use magnitude_state::{ExportedState, KvCodec};
use owner_host::Host;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::time::Duration;

mod owner_host;

const CONTEXT: usize = 4096;

fn resolved(projector: bool) -> ResolvedEngineConfiguration {
    resolved_ctx(projector, CONTEXT)
}

/// [`resolved`] with an explicit context window: the unit-06 tests prompt
/// past `CONTEXT`.
fn resolved_ctx(projector: bool, context_tokens: usize) -> ResolvedEngineConfiguration {
    let variable = |name: &str| {
        PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("set {name}")))
    };
    EngineConfiguration {
        package: PackageOptions::single(
            variable("MAGNITUDE_TEST_GGUF"),
            if projector {
                ProjectorSelection::Explicit(variable("MAGNITUDE_TEST_MMPROJ"))
            } else {
                ProjectorSelection::Disabled
            },
            None,
        ),
        model: ModelPolicy {
            draft_policy: None,
            method: ModelMethod::Plain,
            mtp_proposals: None,
            kv_codec: Some(KvCodec::AffineK8V4),
            lookahead: false,
            exported_logits_rows: 512,
            numerical_profile: None,
            weight_bound: None,
            record_routes: false,
        },
        context_tokens: Some(context_tokens),
        service: limits(),
        path: ExecutionPath::Native,
        device: DeviceRequest::Automatic,
        kernel_cache: std::env::var_os("MAGNITUDE_TEST_KERNEL_CACHE").map(PathBuf::from),
        release: ReleaseRate::of_machine(None),
        model_sha256: None,
    }
    .resolve()
    .unwrap()
}

fn limits() -> ServiceLimits {
    ServiceLimits {
        prefill_tokens: 512,
        decode_tokens: 32,
        decode_share: 0.5,
        locality_seconds: 1.0,
        conversations: std::num::NonZeroUsize::MIN,
    }
}

/// A plain owner caching up to 64 prefixes (none are evicted by count in
/// these tests), with its vocabulary and tokenizer.
fn owner() -> (Host, usize, std::sync::Arc<ByteBpeTokenizer>) {
    let resolved = resolved(false);
    let vocabulary = resolved.manifest.definition.decoder.vocabulary as usize;
    let tokenizer = resolved.host.shared_tokenizer();
    let (domain, bindings, _) =
        build_native_domain(&resolved.manifest, resolved.host.shared_package()).unwrap();
    let host = Host::new(|wakes| {
        Owner::with_prefix_cache_capacity(
            domain,
            bindings,
            limits(),
            PrefixCacheCapacity {
                max_entries: 64,
                workload_state_bytes: u64::MAX,
            },
            wakes,
            None,
            None,
            0,
        )
    });
    (host, vocabulary, tokenizer)
}

/// Deterministic English text of about `len` tokens; distinct seeds give
/// distinct texts.
fn text(tokenizer: &ByteBpeTokenizer, seed: u32, len: usize) -> Vec<TokenId> {
    let text = (1..=64)
        .map(|line| {
            format!(
                "Note {seed}.{line}: the {} crossed the {} at {} past {}.",
                ["heron", "barge", "courier", "glacier", "comet"][(seed as usize + line) % 5],
                ["river", "square", "ridge", "harbour"][line % 4],
                line * 3 + seed as usize,
                ["noon", "dawn", "dusk"][line % 3],
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    let mut tokens = tokenizer.encode(&text, SpecialTokens::Literal).unwrap();
    assert!(tokens.len() >= len);
    tokens.truncate(len);
    tokens
}

fn greedy(prompt: &[TokenId], vocabulary: usize, max_tokens: usize) -> Generation {
    greedy_ctx(prompt, vocabulary, max_tokens, CONTEXT)
}

/// [`greedy`] with an explicit context limit for prompts past `CONTEXT`.
fn greedy_ctx(
    prompt: &[TokenId],
    vocabulary: usize,
    max_tokens: usize,
    context_limit: usize,
) -> Generation {
    Generation::new(
        prompt.to_vec(),
        InputLayout::new(prompt.len(), vec![]).unwrap(),
        Options {
            max_tokens,
            output_capacity: max_tokens,
            context_limit,
            vocabulary,
            stop_tokens: BTreeSet::new(),
            suppressed_tokens: BTreeSet::new(),
            sampling: Sampling::Greedy,
            shaping: Shaping {
                temperature: 0.0,
                ..Shaping::default()
            },
            seed: 0,
            forced_quantum: 0,
            method: MethodChoice::Plain,
            end_of_generation: EndOfGeneration::Suppress,
            reasoning_budget: None,
        },
        None,
    )
    .unwrap()
}

/// The text input the model's family prepares for `prompt`: each row at its
/// own position.
fn text_input(prompt: &[TokenId]) -> PreparedModelInput {
    PreparedModelInput::from_text_coordinates(
        TokenPlan::new(
            prompt.to_vec(),
            InputLayout::new(prompt.len(), vec![]).unwrap(),
        )
        .unwrap(),
        (0..prompt.len()).map(|row| [row as i32; 3]).collect(),
    )
    .unwrap()
}

/// One admitted request of a sequence and what it produced.
struct Turn {
    prompt: Vec<TokenId>,
    output: Vec<TokenId>,
    cached: usize,
}

/// Admit `prompts` together with the prefix cache and run them all to their
/// end. A request completes on its stream only once its terminal prefix is
/// retained (after any reconciliation round the prefix cache needs), and the
/// owner retires it then. Admitting several at once lets later ones wait for
/// an earlier one's shared prefix.
fn run(host: &mut Host, vocabulary: usize, prompts: &[Vec<TokenId>], max_tokens: usize) -> Vec<Turn> {
    let requests = prompts
        .iter()
        .map(|prompt| (prompt.clone(), Vec::new()))
        .collect::<Vec<_>>();
    run_with_cache_points(host, vocabulary, &requests, max_tokens, CONTEXT)
}

/// [`run`] with each prompt's declared cache points.
fn run_with_cache_points(
    host: &mut Host,
    vocabulary: usize,
    requests: &[(Vec<TokenId>, Vec<usize>)],
    max_tokens: usize,
    context_limit: usize,
) -> Vec<Turn> {
    let prompts = requests
        .iter()
        .map(|(prompt, _)| prompt.clone())
        .collect::<Vec<_>>();
    let mut streams = requests
        .iter()
        .map(|(prompt, cache_points)| {
            host.admit(
                greedy_ctx(prompt, vocabulary, max_tokens, context_limit),
                text_input(prompt),
                PrefixRetention::Retain {
                    cache_points: cache_points.clone(),
                },
                max_tokens,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    host.drive(
        &mut streams.iter_mut().collect::<Vec<_>>(),
        Duration::from_secs(120),
    );
    streams
        .into_iter()
        .zip(prompts)
        .map(|(stream, prompt)| {
            assert_eq!(stream.output.len(), max_tokens);
            assert!(
                host.snapshot(stream.id).is_none(),
                "a completed request is retired"
            );
            Turn {
                prompt,
                cached: stream.usage().cached_tokens,
                output: stream.output,
            }
        })
        .collect()
}

fn path(turn: &Turn) -> Vec<TokenId> {
    turn.prompt.iter().chain(&turn.output).copied().collect()
}

/// A request the host abandons keeps what it computed: the same prompt sent
/// again resumes at least where the abandoned one stopped (a prompt group's
/// chain under way runs on, so it keeps that group whole), and a stopped
/// reply resumes where it stopped.
#[test]
#[ignore = "requires a device and MAGNITUDE_TEST_GGUF"]
fn an_abandoned_request_resumes_where_it_stopped() {
    let (mut host, vocabulary, tokenizer) = owner();
    let prompt = (7..10).flat_map(|seed| text(&tokenizer, seed * 10, 600)).collect::<Vec<_>>();
    let mut stream = host
        .admit(greedy(&prompt, vocabulary, 12), text_input(&prompt), PrefixRetention::Retain { cache_points: Vec::new() }, 12)
        .unwrap();
    let id = stream.id;
    host.step_until(&mut [&mut stream], Duration::from_secs(300), |host, _| {
        host.snapshot(id).is_some_and(|snapshot| snapshot.resident_position >= 512)
    });
    let stopped = host.snapshot(id).unwrap().resident_position;
    assert!(stopped < prompt.len() - 1, "stopped mid-prompt");
    host.owner_mut().cancel(id);
    host.release(stream, Duration::from_secs(300));
    let [resumed] = run(&mut host, vocabulary, &[prompt.clone()], 12).try_into().ok().unwrap();
    eprintln!("stopped at {stopped} of {}; resumed with {} cached", prompt.len(), resumed.cached);
    assert!(
        (stopped..prompt.len()).contains(&resumed.cached),
        "the prompt resumes where the abandoned request stopped or beyond"
    );

    let longer = text(&tokenizer, 8, 400);
    let mut stream = host
        .admit(greedy(&longer, vocabulary, 200), text_input(&longer), PrefixRetention::Retain { cache_points: Vec::new() }, 200)
        .unwrap();
    let id = stream.id;
    host.step_until(&mut [&mut stream], Duration::from_secs(300), |_, streams| streams[0].output.len() >= 40);
    let reached = host.snapshot(id).unwrap().resident_position;
    host.owner_mut().cancel(id);
    let mut seen = stream.output.clone();
    host.release(stream, Duration::from_secs(300));
    assert!(longer.len() + seen.len() >= reached, "the reply up to its state was seen");
    seen.truncate(reached - longer.len());
    let mut again = longer.clone();
    again.extend(&seen);
    again.extend(text(&tokenizer, 9, 20));
    let [continued] = run(&mut host, vocabulary, &[again], 4).try_into().ok().unwrap();
    eprintln!("stopped mid-reply at {reached}; the next request resumed with {} cached", continued.cached);
    assert_eq!(continued.cached, reached, "the reply seen so far stays cached");
}

/// Every turn of a conversation extends the previous prompt and reply, as a
/// chat template that re-renders prior replies exactly produces. Each turn
/// resumes at the end of the previous reply.
#[test]
#[ignore = "requires a device and MAGNITUDE_TEST_GGUF"]
fn conversation_turns_resume_at_the_previous_reply() {
    let (mut host, vocabulary, tokenizer) = owner();
    let mut prompt = text(&tokenizer, 1, 200);
    let mut previous: Option<usize> = None;
    for turn in 0..4 {
        let [done] = run(&mut host, vocabulary, &[prompt.clone()], 12)
            .try_into()
            .ok()
            .unwrap();
        eprintln!("turn {turn}: prompt={} cached={}", done.prompt.len(), done.cached);
        match previous {
            None => assert_eq!(done.cached, 0),
            Some(end) => assert_eq!(done.cached, end, "turn {turn} resumes at the last reply"),
        }
        let reply_end = done.prompt.len() + done.output.len();
        previous = Some(reply_end);
        prompt = path(&done);
        prompt.extend(text(&tokenizer, 100 + turn, 40));
    }
}

/// The prefixes a growing conversation leaves cached never take the state
/// stores past the workload state the plan charges (one conversation at the
/// fit depth with its three checkpoints), while every turn still resumes at
/// the previous reply. A recurrent model pins one bank per cached prefix, so
/// without the bound each turn would hold two more.
#[test]
#[ignore = "requires a device and MAGNITUDE_TEST_GGUF"]
fn a_growing_conversation_stays_within_the_charged_workload_state() {
    let resolved = resolved(false);
    let vocabulary = resolved.manifest.definition.decoder.vocabulary as usize;
    let tokenizer = resolved.host.shared_tokenizer();
    let (domain, bindings, plan) =
        build_native_domain(&resolved.manifest, resolved.host.shared_package()).unwrap();
    let workload = plan.capacity().workload_state_bytes;
    let mut host = Host::new(|wakes| {
        Owner::with_resource_plan(domain, bindings, limits(), &plan, wakes, None, None, 0)
    });
    let mut prompt = text(&tokenizer, 1, 300);
    let mut previous: Option<usize> = None;
    for turn in 0..12 {
        let [done] = run(&mut host, vocabulary, &[prompt.clone()], 12)
            .try_into()
            .ok()
            .unwrap();
        if let Some(end) = previous {
            assert_eq!(done.cached, end, "turn {turn} resumes at the last reply");
        }
        let occupied = host.owner().state_bytes().occupied();
        eprintln!(
            "turn {turn}: prompt={} cached={} prefixes={} occupied={occupied} workload={workload}",
            done.prompt.len(),
            done.cached,
            host.owner().cached_prefixes(),
        );
        assert!(occupied <= workload, "turn {turn} occupies {occupied} of {workload}");
        previous = Some(done.prompt.len() + done.output.len());
        prompt = path(&done);
        prompt.extend(text(&tokenizer, 100 + turn, 300));
    }
}

/// Distinct conversations, each retaining its own checkpoints (a recurrent
/// model pins a bank per cached prefix), never make the state stores commit
/// more than the workload state the plan charges: once retained prefixes
/// fill it, a new conversation takes the backing of the least recently used
/// ones instead of growing beside them. Holding only what holders occupy
/// within it is not enough: the slabs a new conversation grew while the
/// retained prefixes stood stay committed after its own caching releases
/// them.
#[test]
#[ignore = "requires a device and MAGNITUDE_TEST_GGUF"]
fn distinct_conversations_commit_no_state_beyond_the_charged_workload() {
    let resolved = resolved(false);
    let vocabulary = resolved.manifest.definition.decoder.vocabulary as usize;
    let tokenizer = resolved.host.shared_tokenizer();
    let (domain, bindings, plan) =
        build_native_domain(&resolved.manifest, resolved.host.shared_package()).unwrap();
    let workload = plan.capacity().workload_state_bytes;
    let mut host = Host::new(|wakes| {
        Owner::with_resource_plan(domain, bindings, limits(), &plan, wakes, None, None, 0)
    });
    let mut released = false;
    for conversation in 0..24 {
        let prompt = text(&tokenizer, 200 + conversation, 600);
        run(&mut host, vocabulary, &[prompt], 12);
        let committed = host.owner().state_bytes().committed();
        let prefixes = host.owner().cached_prefixes();
        released |= prefixes < 2 * (conversation as usize + 1);
        eprintln!("conversation {conversation}: prefixes={prefixes} committed={committed} workload={workload}");
        assert!(
            committed <= workload,
            "conversation {conversation} commits {committed} of {workload}"
        );
    }
    assert!(released, "the conversations filled the workload state");
}

/// A template that rewrites the previous reply (dropping reasoning, say)
/// diverges inside it; the next turn resumes one row before the previous
/// prompt's end.
#[test]
#[ignore = "requires a device and MAGNITUDE_TEST_GGUF"]
fn a_rewritten_reply_resumes_at_the_previous_prompt() {
    let (mut host, vocabulary, tokenizer) = owner();
    let first = text(&tokenizer, 2, 200);
    let [done] = run(&mut host, vocabulary, &[first.clone()], 12)
        .try_into()
        .ok()
        .unwrap();
    let mut rewritten = first.clone();
    rewritten.extend(text(&tokenizer, 3, 60));
    assert_ne!(rewritten[first.len()], done.output[0]);
    let [next] = run(&mut host, vocabulary, &[rewritten], 12)
        .try_into()
        .ok()
        .unwrap();
    assert_eq!(next.cached, first.len() - 1);
}

/// Short turns of a conversation whose template renders the previous reply
/// unlike the generation prompt it followed (an empty reasoning block the
/// history omits): each prompt declares where its last message opens and
/// where its next turn leaves it, as chat preparation does. Every turn is far
/// shorter than the minimum branch gain, yet each resumes where the previous
/// turn left its prompt, rather than all at the first turn's last message.
#[test]
#[ignore = "requires a device and MAGNITUDE_TEST_GGUF"]
fn short_rewritten_turns_resume_where_the_previous_turn_left_its_prompt() {
    let (mut host, vocabulary, tokenizer) = owner();
    let suffix = tokenizer
        .encode(" Reply in one word.", SpecialTokens::Literal)
        .unwrap();
    let mut history = text(&tokenizer, 41, 400);
    let mut previous: Option<usize> = None;
    for turn in 0..6 {
        let opened = history.len();
        history.extend(text(&tokenizer, 50 + turn, 20));
        let left = history.len();
        let mut prompt = history.clone();
        prompt.extend(&suffix);
        let [done] =
            run_with_cache_points(&mut host, vocabulary, &[(prompt, vec![opened, left])], 4, CONTEXT)
                .try_into()
                .ok()
                .unwrap();
        eprintln!(
            "turn {turn}: prompt={} cached={}",
            done.prompt.len(),
            done.cached
        );
        match previous {
            None => assert_eq!(done.cached, 0),
            Some(left) => assert_eq!(
                done.cached, left,
                "turn {turn} resumes where the last turn left"
            ),
        }
        previous = Some(left);
        let reply = text(&tokenizer, 60 + turn, 3);
        assert_ne!(reply[0], suffix[0]);
        history.extend(reply);
    }
}

/// A long system prompt, then a last message that changes on every request,
/// with its start declared as a cache point (as chat preparation declares
/// the boundary opening the last message). Every request after the first
/// resumes at that start, however its message begins: the second message
/// shares a leading phrase with the first, the third starts differently.
#[test]
#[ignore = "requires a device and MAGNITUDE_TEST_GGUF"]
fn a_changed_last_message_resumes_at_its_declared_start() {
    let (mut host, vocabulary, tokenizer) = owner();
    let system = text(&tokenizer, 20, 1000);
    let lead = text(&tokenizer, 21, 24);
    let message = |parts: &[Vec<TokenId>]| {
        let mut prompt = system.clone();
        for part in parts {
            prompt.extend(part);
        }
        (prompt, vec![system.len()])
    };
    let messages = [
        message(&[lead.clone(), text(&tokenizer, 22, 40)]),
        message(&[lead.clone(), text(&tokenizer, 23, 40)]),
        message(&[tokenizer
            .encode(
                "Summarize the harbour log in one line, then list every courier by name.",
                SpecialTokens::Literal,
            )
            .unwrap()]),
    ];
    assert_ne!(messages[2].0[system.len()], lead[0]);
    let mut cached = Vec::new();
    for request in &messages {
        let [done] = run_with_cache_points(&mut host, vocabulary, &[request.clone()], 8, CONTEXT)
            .try_into()
            .ok()
            .unwrap();
        eprintln!("prompt={} cached={}", done.prompt.len(), done.cached);
        cached.push(done.cached);
    }
    assert_eq!(cached, [0, system.len(), system.len()]);
}

/// Without a declared cache point, a changed last message diverges below
/// every retained state: the request recomputes its prompt and retains a
/// state at the divergence, where the next request diverging there resumes.
#[test]
#[ignore = "requires a device and MAGNITUDE_TEST_GGUF"]
fn an_undeclared_divergence_is_retained_by_the_request_that_finds_it() {
    let (mut host, vocabulary, tokenizer) = owner();
    let system = text(&tokenizer, 30, 1000);
    let message = |seed| {
        let mut prompt = system.clone();
        prompt.extend(text(&tokenizer, seed, 40));
        prompt
    };
    let turns = [31, 32, 33].map(|seed| {
        let [done] = run(&mut host, vocabulary, &[message(seed)], 8)
            .try_into()
            .ok()
            .unwrap();
        eprintln!("prompt={} cached={}", done.prompt.len(), done.cached);
        done.cached
    });
    assert_eq!(turns[..2], [0, 0]);
    assert!(turns[2] >= system.len());
}

/// A tiny deterministic generator, so a failing sequence reproduces.
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, bound: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % bound as u64) as usize
    }
}

/// Random sequences of conversations, branches from earlier paths, and
/// concurrent groups sharing prefixes. Every request resumes at least as
/// deeply as the deepest prefix it is guaranteed to find: the whole path of
/// an earlier finished request, or one row before an earlier prompt's end,
/// whichever ends deepest within its prompt.
#[test]
#[ignore = "requires a device and MAGNITUDE_TEST_GGUF"]
fn arbitrary_request_sequences_resume_from_their_deepest_cached_prefix() {
    let (mut host, vocabulary, tokenizer) = owner();
    let seed = std::env::var("PREFIX_CACHE_SEED")
        .ok()
        .and_then(|seed| seed.parse().ok())
        .unwrap_or(7);
    let mut random = Lcg(seed);
    let roots = [text(&tokenizer, 10, 160), text(&tokenizer, 11, 220)];
    let mut finished: Vec<Turn> = Vec::new();
    let mut fresh = 1000;
    for round in 0..12 {
        let group = 1 + random.below(3);
        let prompts = (0..group)
            .map(|_| {
                fresh += 1;
                let tail = text(&tokenizer, fresh, 30 + random.below(80));
                let mut prompt = match (finished.len(), random.below(3)) {
                    // A fresh conversation on a shared system prompt.
                    (0, _) | (_, 0) => roots[random.below(roots.len())].clone(),
                    // The next turn of an earlier conversation.
                    (count, 1) => path(&finished[random.below(count)]),
                    // A branch from inside an earlier path.
                    (count, _) => {
                        let source = path(&finished[random.below(count)]);
                        source[..MIN_PREFIX_HIT + random.below(source.len() - MIN_PREFIX_HIT)]
                            .to_vec()
                    }
                };
                prompt.extend(tail);
                prompt
            })
            .collect::<Vec<_>>();
        let turns = run(&mut host, vocabulary, &prompts, 8);
        for turn in &turns {
            let guaranteed = finished
                .iter()
                .flat_map(|earlier| [path(earlier), earlier.prompt[..earlier.prompt.len() - 1].to_vec()])
                .filter(|prefix| {
                    prefix.len() >= MIN_PREFIX_HIT
                        && prefix.len() < turn.prompt.len()
                        && turn.prompt.starts_with(prefix)
                })
                .map(|prefix| prefix.len())
                .max()
                .unwrap_or(0);
            eprintln!(
                "round {round}: prompt={} cached={} guaranteed={guaranteed}",
                turn.prompt.len(),
                turn.cached
            );
            assert!(turn.cached >= guaranteed, "round {round} resumed too shallow");
            assert!(turn.cached < turn.prompt.len());
        }
        finished.extend(turns);
    }
}

/// Run the encodes a residency returned and install their features. Each
/// flight carries the bindings and returns them.
fn encode(
    domain: &mut ExecutorDomain,
    mut bindings: StateBindings,
    operations: Vec<Operation>,
) -> StateBindings {
    for operation in operations {
        let ReservedResources::Vision(workspace, output) = domain
            .reserve(&mut bindings, std::slice::from_ref(&operation))
            .unwrap()
            .into_resources()
        else {
            panic!("an encode runs on the vision lane")
        };
        let flight = domain
            .submit_vision(bindings, &operation, workspace, output)
            .unwrap_or_else(|failure| panic!("{}", failure.error()));
        let (pending, next) = domain.finish_vision(flight).unwrap();
        bindings = next;
        domain
            .reconcile(pending, PhysicalDecision::rows(0))
            .unwrap();
    }
    bindings
}

/// Run one forward operation and return its last row's logits.
fn launch(
    domain: &mut ExecutorDomain,
    mut bindings: StateBindings,
    operation: Operation,
) -> (Vec<f32>, StateBindings) {
    let rows = operation.row_count();
    let operations = [operation];
    let ReservedResources::Target(reservation) = domain
        .reserve(&mut bindings, &operations)
        .unwrap()
        .into_resources()
    else {
        panic!("a forward runs on the target lane")
    };
    let flight = domain
        .submit_target(bindings, &operations, reservation)
        .unwrap_or_else(|failure| panic!("{}", failure.error()));
    let (pending, bindings) = domain.finish_target(flight).unwrap();
    let mut logits = Vec::new();
    for pending in pending {
        let Outcome::Forward { rows: results, .. } = pending.outcome().clone() else {
            panic!("a forward returns forward rows")
        };
        for row in results.iter().filter_map(|row| row.logits.as_ref()) {
            logits = row.read_to_host().unwrap();
        }
        domain
            .reconcile(pending, PhysicalDecision::rows(rows))
            .unwrap();
    }
    assert!(!logits.is_empty());
    (logits, bindings)
}

/// Forward the input rows `range` and return the last row's logits.
fn forward(
    domain: &mut ExecutorDomain,
    mut bindings: StateBindings,
    request: RequestId,
    input: &PreparedModelInput,
    range: std::ops::Range<usize>,
) -> (Vec<f32>, StateBindings) {
    let rows = range.len();
    let operation = Operation::Forward {
        request,
        kind: WorkKind::Prefill,
        tokens: input.tokens()[range.clone()].to_vec(),
        position: range.start,
        conditioning: None,
        demand: Demand::LOGITS,
        select: Vec::new(),
        committed: rows,
        prime: None,
        front: Vec::new(),
        chain: None,
    };
    launch(domain, bindings, operation)
}

/// Forward one already-sampled decode token at `position` and return its
/// logits: a one-row prefill over the current state.
fn forward_token(
    domain: &mut ExecutorDomain,
    bindings: StateBindings,
    request: RequestId,
    token: TokenId,
    position: usize,
) -> (Vec<f32>, StateBindings) {
    launch(
        domain,
        bindings,
        Operation::Forward {
            request,
            kind: WorkKind::Prefill,
            tokens: vec![token],
            position,
            conditioning: None,
            demand: Demand::LOGITS,
            select: Vec::new(),
            committed: 1,
            prime: None,
            front: Vec::new(),
            chain: None,
        },
    )
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Counts `restore_hit` tracing events (MUST 1); share the counter via the Arc.
#[derive(Clone, Default)]
struct RestoreHits(std::sync::Arc<std::sync::atomic::AtomicU64>);

impl RestoreHits {
    fn get(&self) -> u64 {
        self.0.load(std::sync::atomic::Ordering::SeqCst)
    }
}

struct HitVisitor {
    hit: bool,
}
impl Default for HitVisitor {
    fn default() -> Self {
        Self { hit: false }
    }
}
impl tracing::field::Visit for HitVisitor {
    fn record_debug(&mut self, _f: &tracing::field::Field, _v: &dyn std::fmt::Debug) {}
    fn record_bool(&mut self, f: &tracing::field::Field, v: bool) {
        if f.name() == "restore_hit" && v {
            self.hit = true;
        }
    }
}

impl tracing::Subscriber for RestoreHits {
    fn enabled(&self, _m: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _a: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn event(&self, event: &tracing::Event<'_>) {
        let mut v = HitVisitor::default();
        event.record(&mut v);
        if v.hit {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
    fn record(&self, _s: &tracing::span::Id, _v: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _s: &tracing::span::Id, _f: &tracing::span::Id) {}
    fn enter(&self, _s: &tracing::span::Id) {}
    fn exit(&self, _s: &tracing::span::Id) {}
}

fn argmax(logits: &[f32]) -> TokenId {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| TokenId(i as u32))
        .unwrap_or_else(|| TokenId(0))
}

/// Greedily decode `steps` tokens from a state at `position`: the first
/// token comes from `first_logits` (the last prefill row), each later one
/// is the argmax of the previous decode row's logits. Returns the tokens
/// and every row's logits, in step order (one logits row per token).
fn greedy_decode(
    domain: &mut ExecutorDomain,
    mut bindings: StateBindings,
    request: RequestId,
    first_logits: Vec<f32>,
    position: usize,
    steps: usize,
) -> (Vec<TokenId>, Vec<Vec<f32>>, StateBindings) {
    let mut logits_rows = vec![first_logits.clone()];
    let mut tokens = Vec::with_capacity(steps);
    let mut next = argmax(&first_logits);
    for _ in 0..steps {
        tokens.push(next);
        let (logits, b) = forward_token(domain, bindings, request, next, position + tokens.len() - 1);
        bindings = b;
        logits_rows.push(logits.clone());
        next = argmax(&logits);
    }
    (tokens, logits_rows, bindings)
}

/// A prompt with one image (`VISION_IMAGE`, or llama.cpp's `test-1.jpeg`)
/// between two sentences.
fn image_prompt(host: &HostArtifacts) -> PreparedModelInput {
    let placeholder = host.media_placeholder().expect("a model with a projector");
    let image = ImageInput {
        media_type: "image/jpeg".into(),
        bytes: std::fs::read(
            std::env::var_os("VISION_IMAGE")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    PathBuf::from(std::env::var("HOME").unwrap())
                        .join("repos/llama.cpp/tools/mtmd/test-1.jpeg")
                }),
        )
        .unwrap()
        .into(),
    };
    let tokens = host
        .tokenizer()
        .encode(
            &format!("Look at this picture:{placeholder}Now describe what you see in it, carefully."),
            SpecialTokens::Recognize,
        )
        .unwrap();
    host.prepare_input(tokens, &[image]).unwrap()
}

/// Vision holds no activation or output until an image arrives: an image
/// claims them, idle shrink releases them, and every image claims and
/// releases the same bytes.
#[test]
#[ignore = "requires a Metal or CUDA device, MAGNITUDE_TEST_GGUF and MAGNITUDE_TEST_MMPROJ"]
fn vision_claims_its_slots_on_an_image_and_releases_them_at_idle() {
    let resolved = resolved(true);
    let input = image_prompt(&resolved.host);
    let (mut domain, mut bindings, _) =
        build_native_domain(&resolved.manifest, resolved.host.shared_package()).unwrap();
    let charged = |domain: &ExecutorDomain| {
        let device = domain.resources().device();
        device.allocated()
    };
    let mut cycles = Vec::new();
    for request in [RequestId(1), RequestId(2)] {
        domain.install_input(request, input.clone()).unwrap();
        let encodes = domain.open_served(&mut bindings, request, None).unwrap();
        assert_eq!(encodes.len(), 1);
        let opened = charged(&domain);
        bindings = encode(&mut domain, bindings, encodes);
        let encoded = charged(&domain);
        assert!(encoded > opened, "the image claimed no vision storage");
        domain.close(request).unwrap();
        // Vision's free slots release on the load's release thread: the
        // charge falls by what it released once the boundary that finds it
        // ended records it.
        let released = magnitude_executor::fixtures::shrink_settled(&mut domain, &mut bindings);
        let idle = charged(&domain);
        assert_eq!(encoded - idle, released);
        assert!(released > 0, "idle released no vision storage");
        eprintln!(
            "vision cycle: opened={opened} encoded={encoded} idle={idle} released={released}"
        );
        cycles.push((encoded - opened, released, idle));
    }
    // The first image also loads the vision weights, which stay resident
    // until optional components are released; the second claims only the
    // slots, and idle releases exactly them both times.
    let [(_, first_released, first_idle), (second_claimed, second_released, second_idle)] =
        cycles[..]
    else {
        unreachable!("two cycles")
    };
    assert_eq!(second_claimed, second_released, "idle kept vision storage");
    assert_eq!(first_released, second_released);
    assert_eq!(first_idle, second_idle);
}

/// A request with an image keeps its input across eviction: replay
/// re-encodes the image and reaches the same logits as an uninterrupted run.
/// A request resuming from a cached prefix past the image needs no encode
/// and reaches them too.
#[test]
#[ignore = "requires a Metal or CUDA device, MAGNITUDE_TEST_GGUF and MAGNITUDE_TEST_MMPROJ"]
fn image_conditioning_survives_eviction_and_cached_resumption() {
    let resolved = resolved(true);
    let host = &resolved.host;
    let input = image_prompt(host);
    let [span] = input.layout().spans() else {
        panic!("one image span")
    };
    // Split after the image, so the first chunk carries the whole image.
    let split = span.end + 2;
    let end = input.tokens().len();
    assert!(split < end && split <= 512, "image span {}..{} of {end}", span.start, span.end);
    let (mut domain, mut bindings, _) =
        build_native_domain(&resolved.manifest, host.shared_package()).unwrap();

    // Uninterrupted: encode, the image chunk, then the rest.
    let reference = RequestId(1);
    domain.install_input(reference, input.clone()).unwrap();
    let encodes = domain.open_served(&mut bindings, reference, None).unwrap();
    assert_eq!(encodes.len(), 1);
    bindings = encode(&mut domain, bindings, encodes);
    bindings = forward(&mut domain, bindings, reference, &input, 0..split).1;
    let after_image = domain.resume_state(reference).unwrap();
    let (expected, next) = forward(&mut domain, bindings, reference, &input, split..end);
    bindings = next;
    domain.close(reference).unwrap();

    // Evicted after the image: its state and features go, its input stays.
    // Becoming resident again re-encodes the image for the replay.
    let evicted = RequestId(2);
    domain.install_input(evicted, input.clone()).unwrap();
    let encodes = domain.open_served(&mut bindings, evicted, None).unwrap();
    bindings = encode(&mut domain, bindings, encodes);
    bindings = forward(&mut domain, bindings, evicted, &input, 0..split).1;
    domain.release_state(&[evicted]).unwrap();
    let encodes = domain.open_served(&mut bindings, evicted, None).unwrap();
    assert_eq!(encodes.len(), 1, "replay re-encodes the released image");
    bindings = encode(&mut domain, bindings, encodes);
    bindings = forward(&mut domain, bindings, evicted, &input, 0..split).1;
    let (replayed, next) = forward(&mut domain, bindings, evicted, &input, split..end);
    bindings = next;
    assert_eq!(bits(&replayed), bits(&expected), "replay keeps the image conditioning");
    domain.close(evicted).unwrap();

    // Resumed from the cached state past the image: nothing to encode.
    let resumed = RequestId(3);
    domain.install_input(resumed, input.clone()).unwrap();
    let encodes = domain
        .open_served(&mut bindings, resumed, Some(&after_image))
        .unwrap();
    assert!(encodes.is_empty(), "an image before the resume position is not re-encoded");
    let (continued, _) = forward(&mut domain, bindings, resumed, &input, split..end);
    assert_eq!(bits(&continued), bits(&expected));
    domain.close(resumed).unwrap();

    drop(after_image);
}

/// The state export of a real checkpoint (cold tier unit 01C): two exports
/// of the same checkpoint agree, history and bank byte counts match their
/// row counts, and prefilling a different prompt leaves the exported bytes
/// untouched.
#[test]
#[ignore = "requires a Metal or CUDA device and MAGNITUDE_TEST_GGUF"]
fn cold_tier_export_roundtrip() {
    let resolved = resolved(false);
    let tokenizer = resolved.host.shared_tokenizer();
    let (mut domain, mut bindings, _) =
        build_native_domain(&resolved.manifest, resolved.host.shared_package()).unwrap();

    // Five distinct 512-token segments, one 2048-token prefill.
    let prompt_a: Vec<TokenId> = (0..5)
        .flat_map(|i| text(&tokenizer, 100u32 + i as u32, 512))
        .take(2048)
        .collect();
    let input_a = text_input(&prompt_a);
    let request_a = RequestId(1);
    domain.install_input(request_a, input_a.clone()).unwrap();
    let encodes = domain.open_served(&mut bindings, request_a, None).unwrap();
    assert!(encodes.is_empty(), "plain text has nothing to encode");
    for chunk in 0..4 {
        let start = chunk * 512;
        bindings = forward(&mut domain, bindings, request_a, &input_a, start..start + 512).1;
    }
    let held = domain.resume_state(request_a).unwrap();
    domain.close(request_a).unwrap();
    let _ = domain.shrink_state(&mut bindings, CompletionWake::new(|| {})).unwrap();

    let first = held.target().export().unwrap();
    assert_eq!(first.position, 2048);
    let second = held.target().export().unwrap();
    assert_eq!(first, second, "two exports of the same checkpoint agree");

    let mut history_bytes = 0;
    for region in &first.history {
        let rows: usize = region.ranges.iter().map(|(s, e)| e - s).sum();
        assert_eq!(
            region.bytes.len(),
            region.row_bytes * rows,
            "domain {} region {} exports its rows",
            region.domain,
            region.region
        );
        history_bytes += region.bytes.len();
    }
    assert!(history_bytes > 0, "a 2K prefill exports history rows");

    let arenas = held.target().store().recurrent_arenas();
    assert_eq!(first.bank.len(), arenas.len(), "one bank row per recurrent arena");
    for (row, arena) in first.bank.iter().zip(arenas.iter()) {
        assert_eq!(
            row.len() as u64,
            arena.byte_len() / arena.extents()[0],
            "one bank row of the arena's row size"
        );
    }

    // A different prefill must not disturb the first checkpoint's rows.
    let prompt_b = text(&tokenizer, 999, 512);
    let input_b = text_input(&prompt_b);
    let request_b = RequestId(2);
    domain.install_input(request_b, input_b.clone()).unwrap();
    let encodes = domain.open_served(&mut bindings, request_b, None).unwrap();
    assert!(encodes.is_empty());
    bindings = forward(&mut domain, bindings, request_b, &input_b, 0..512).1;
    domain.close(request_b).unwrap();
    let again = held.target().export().unwrap();
    assert_eq!(first, again, "the first export is stable across other prefills");
}

/// What an export saves, without where its rows lay: an import places the
/// rows afresh, so only their position, recurrent version, starts, bank and
/// bytes in logical order are its to reproduce.
fn saved_content(saved: &ExportedState) -> impl PartialEq + std::fmt::Debug + '_ {
    (
        saved.position,
        saved.tape,
        &saved.starts,
        &saved.bank,
        saved
            .history
            .iter()
            .map(|region| (region.domain, region.region, region.row_bytes, &region.bytes))
            .collect::<Vec<_>>(),
    )
}

/// The state import of an exported checkpoint (cold tier unit 02): after the
/// original state is evicted and its rows are reused by a different prefill,
/// importing the export back into the store reproduces it byte-for-byte.
#[test]
#[ignore = "requires a Metal or CUDA device and MAGNITUDE_TEST_GGUF"]
fn cold_tier_import_roundtrip() {
    let resolved = resolved(false);
    let tokenizer = resolved.host.shared_tokenizer();
    let (mut domain, mut bindings, _) =
        build_native_domain(&resolved.manifest, resolved.host.shared_package()).unwrap();

    // Five distinct 512-token segments, one 2048-token prefill.
    let prompt_a: Vec<TokenId> = (0..5)
        .flat_map(|i| text(&tokenizer, 100u32 + i as u32, 512))
        .take(2048)
        .collect();
    let input_a = text_input(&prompt_a);
    let request_a = RequestId(1);
    domain.install_input(request_a, input_a.clone()).unwrap();
    let encodes = domain.open_served(&mut bindings, request_a, None).unwrap();
    assert!(encodes.is_empty(), "plain text has nothing to encode");
    for chunk in 0..4 {
        let start = chunk * 512;
        bindings = forward(&mut domain, bindings, request_a, &input_a, start..start + 512).1;
    }
    let held = domain.resume_state(request_a).unwrap();
    let original = held.target().export().unwrap();
    assert_eq!(original.position, 2048);
    let store = std::rc::Rc::clone(held.target().store());
    drop(held);
    domain.close(request_a).unwrap();
    let _ = domain.shrink_state(&mut bindings, CompletionWake::new(|| {})).unwrap();

    // A different prefill reuses the freed rows, then frees them again.
    let prompt_b = text(&tokenizer, 999, 512);
    let input_b = text_input(&prompt_b);
    let request_b = RequestId(2);
    domain.install_input(request_b, input_b.clone()).unwrap();
    let encodes = domain.open_served(&mut bindings, request_b, None).unwrap();
    assert!(encodes.is_empty());
    bindings = forward(&mut domain, bindings, request_b, &input_b, 0..512).1;
    domain.close(request_b).unwrap();
    let _ = domain.shrink_state(&mut bindings, CompletionWake::new(|| {})).unwrap();

    let imported = store.import(&original).unwrap();
    let restored = imported.checkpoint().export().unwrap();
    assert_eq!(
        saved_content(&original),
        saved_content(&restored),
        "import reproduces the exported state byte-for-byte"
    );
}

/// The SSD cold tier keeps evicted prefixes on disk: A's retained prefix is
/// pushed out of the RAM cache by larger distinct prefixes, spilled to the
/// state-cache directory, and a later request extending A's path resumes from
/// the disk entry instead of re-prefilling.
#[test]
#[ignore = "requires a Metal or CUDA device and MAGNITUDE_TEST_GGUF"]
fn cold_tier_spill_restore() {
    let resolved = resolved(false);
    let vocabulary = resolved.manifest.definition.decoder.vocabulary as usize;
    let tokenizer = resolved.host.shared_tokenizer();
    let (domain, bindings, plan) =
        build_native_domain(&resolved.manifest, resolved.host.shared_package()).unwrap();

    let identity = [7u8; 32];
    let dir = std::env::temp_dir()
        .join(format!("magnitude-cold-tier-spill-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut host = Host::new(|wakes| {
        Owner::with_resource_plan(
            domain,
            bindings,
            limits(),
            &plan,
            wakes,
            Some(identity),
            Some(dir.clone()),
            8 * 1024 * 1024 * 1024,
        )
    });

    let prompt_a: Vec<TokenId> = (0..5)
        .flat_map(|i| text(&tokenizer, 100u32 + i as u32, 512))
        .take(2048)
        .collect();
    let [first] = run(&mut host, vocabulary, &[prompt_a], 12)
        .try_into()
        .ok()
        .unwrap();
    assert_eq!(first.cached, 0);
    let path_a = path(&first);
    eprintln!(
        "A: prompt={} path={} prefixes={}",
        path_a.len(),
        path_a.len(),
        host.owner().cached_prefixes()
    );

    // The tier dir holds one identity subdir; a spill is any entry file.
    let spilled = || -> bool {
        std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .find_map(|identity_dir| {
                std::fs::read_dir(identity_dir.join("entries"))
                    .ok()
                    .and_then(|entries| entries.flatten().next().map(|_| true))
            })
            .unwrap_or(false)
    };

    // Distinct larger prompts until A's LRU prefix is evicted and spilled.
    for i in 0..6u32 {
        let prompt_b: Vec<TokenId> = (0..8)
            .flat_map(|j| text(&tokenizer, 300u32 + i * 100 + j, 512))
            .take(3800)
            .collect();
        let [turn] = run(&mut host, vocabulary, &[prompt_b], 12)
            .try_into()
            .ok()
            .unwrap();
        eprintln!(
            "B{i}: prompt={} cached={} prefixes={} spilled={}",
            turn.prompt.len(),
            turn.cached,
            host.owner().cached_prefixes(),
            spilled()
        );
        if spilled() {
            break;
        }
    }
    assert!(spilled(), "A's retained prefix was evicted and spilled to disk");

    // A' extends A's full path; the RAM cache no longer holds it, so the
    // request must resume from the spilled disk entry.
    let prompt_a2: Vec<TokenId> = path_a
        .iter()
        .copied()
        .chain(text(&tokenizer, 990, 40))
        .collect();
    let a2_prompt_len = prompt_a2.len();
    let [second] = run(&mut host, vocabulary, &[prompt_a2], 8)
        .try_into()
        .ok()
        .unwrap();
    eprintln!(
        "A': prompt={} cached={} prefixes={}",
        a2_prompt_len,
        second.cached,
        host.owner().cached_prefixes()
    );
    assert_eq!(
        second.cached,
        path_a.len(),
        "A' resumes at A's spilled position"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// The SSD cold tier's restore is numerically exact: a prefix spilled to
/// disk and restored through the owner resumes the same continuation a
/// fresh engine produces (same tokens), and at the domain level the
/// restored state yields bitwise-identical logits to an uninterrupted
/// prefill at every decode step (cold tier unit 06).
const CONTEXT_06: usize = 16384;
const PROMPT_06: usize = 8192;
const DECODE_STEPS: usize = 32;

/// The SSD cold tier's restore is numerically exact: a prefix spilled to
/// disk and restored through the owner resumes the same continuation a
/// fresh engine produces (same tokens), and at the domain level the
/// restored state yields bitwise-identical logits to an uninterrupted
/// prefill at every decode step (cold tier unit 06).
#[test]
#[ignore = "requires a Metal or CUDA device and MAGNITUDE_TEST_GGUF"]
fn cold_tier_restore_matches_fresh_prefill() {
    cold_tier_restore_matches_fresh_prefill_impl(false);
}

/// The restart variant: the engine (domain + owner) is dropped and rebuilt
/// between the spill and the restore; the disk entry must survive the
/// restart and restore identically.
#[test]
#[ignore = "requires a Metal or CUDA device and MAGNITUDE_TEST_GGUF"]
fn cold_tier_restore_matches_fresh_prefill_after_restart() {
    cold_tier_restore_matches_fresh_prefill_impl(true);
}

fn cold_tier_restore_matches_fresh_prefill_impl(restart: bool) {
    let resolved = resolved_ctx(false, CONTEXT_06);
    let vocabulary = resolved.manifest.definition.decoder.vocabulary as usize;
    let tokenizer = resolved.host.shared_tokenizer();

    // Eight thousand tokens: sixteen distinct 512-token segments.
    let prompt: Vec<TokenId> = (0..16)
        .flat_map(|i| text(&tokenizer, 400u32 + i as u32, 512))
        .take(PROMPT_06)
        .collect();

    // -- Part A: the owner path, end to end.
    let identity = [7u8; 32];
    let dir = std::env::temp_dir()
        .join(format!("magnitude-cold-tier-06-{}-{}", restart as u8, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let spilled = || -> bool {
        std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .find_map(|identity_dir| {
                std::fs::read_dir(identity_dir.join("entries"))
                    .ok()
                    .and_then(|entries| entries.flatten().next().map(|_| true))
            })
            .unwrap_or(false)
    };

    let build_owner = || -> Host {
        let (domain, bindings, plan) =
            build_native_domain(&resolved.manifest, resolved.host.shared_package()).unwrap();
        Host::new(|wakes| {
            Owner::with_resource_plan(
                domain,
                bindings,
                limits(),
                &plan,
                wakes,
                Some(identity),
                Some(dir.clone()),
                8 * 1024 * 1024 * 1024,
            )
        })
    };

    // P: a full 8K prefill, thirty-two greedy tokens.
    let mut host = build_owner();
    let [first] = run_with_cache_points(
        &mut host,
        vocabulary,
        &[(prompt.clone(), vec![])],
        DECODE_STEPS,
        CONTEXT_06,
    )
    .try_into()
    .ok()
    .unwrap();
    assert_eq!(first.cached, 0);
    let path_p = path(&first);
    assert_eq!(path_p.len(), PROMPT_06 + DECODE_STEPS);
    eprintln!(
        "P: prompt={} path={} prefixes={}",
        path_p.len(),
        path_p.len(),
        host.owner().cached_prefixes()
    );

    // Distinct longer prompts until P's LRU prefix is evicted and spilled.
    for i in 0..6u32 {
        let prompt_e: Vec<TokenId> = (0..24)
            .flat_map(|j| text(&tokenizer, 600u32 + i * 100 + j, 512))
            .take(12288)
            .collect();
        let [turn] = run_with_cache_points(
            &mut host,
            vocabulary,
            &[(prompt_e, vec![])],
            DECODE_STEPS,
            CONTEXT_06,
        )
        .try_into()
        .ok()
        .unwrap();
        eprintln!(
            "E{i}: prompt={} cached={} prefixes={} spilled={}",
            turn.prompt.len(),
            turn.cached,
            host.owner().cached_prefixes(),
            spilled()
        );
        if spilled() {
            break;
        }
    }
    assert!(spilled(), "P's retained prefix was evicted and spilled to disk");

    // A' extends P's full path; the RAM cache no longer holds it, so the
    // request must resume from the spilled disk entry.
    let prompt_a2: Vec<TokenId> = path_p
        .iter()
        .copied()
        .chain(text(&tokenizer, 990, 40))
        .collect();

    // If the restart variant, drop and rebuild the engine between spill and
    // restore — the disk entries must survive.
    if restart {
        drop(host);
        host = build_owner();
    }

    let hits = RestoreHits::default();
    let [second] = with_default(hits.clone(), || {
        run_with_cache_points(
            &mut host,
            vocabulary,
            &[(prompt_a2.clone(), vec![])],
            DECODE_STEPS,
            CONTEXT_06,
        )
    })
    .try_into()
    .ok()
    .unwrap();
    assert_eq!(
        second.cached,
        path_p.len(),
        "A' resumes at P's spilled position"
    );
    assert_eq!(
        hits.get(),
        1,
        "A' restored exactly one disk entry (MUST 1)"
    );

    // Fresh engine: no cold tier (no identity/dir), must prefill from zero.
    drop(host);
    let (domain, bindings, plan) =
        build_native_domain(&resolved.manifest, resolved.host.shared_package()).unwrap();
    let mut fresh = Host::new(|wakes| {
        Owner::with_resource_plan(
            domain,
            bindings,
            limits(),
            &plan,
            wakes,
            None,
            None,
            0,
        )
    });
    let fresh_hits = RestoreHits::default();
    let [third] = with_default(fresh_hits.clone(), || {
        run_with_cache_points(
            &mut fresh,
            vocabulary,
            &[(prompt_a2.clone(), vec![])],
            DECODE_STEPS,
            CONTEXT_06,
        )
    })
    .try_into()
    .ok()
    .unwrap();
    assert_eq!(
        third.cached,
        0,
        "the fresh engine prefills from the start"
    );
    assert_eq!(
        fresh_hits.get(),
        0,
        "the fresh engine never restores"
    );
    assert_eq!(
        second.output, third.output,
        "the cold restore continues identically to a fresh prefill"
    );

    // -- Part B: domain-level logit equivalence.
    // Fresh reference: uninterrupted 8K prefill + greedy decode, every logits
    // row captured.
    drop(fresh);
    let (mut domain, mut bindings, _) =
        build_native_domain(&resolved.manifest, resolved.host.shared_package()).unwrap();
    let input_p = text_input(&prompt);

    let fresh_req = RequestId(1);
    domain.install_input(fresh_req, input_p.clone()).unwrap();
    assert!(domain.open_served(&mut bindings, fresh_req, None).unwrap().is_empty());
    let mut fresh_logits: Vec<Vec<f32>> = Vec::with_capacity(16);
    for chunk in 0..16 {
        let start = chunk * 512;
        let (logits, b) = forward(&mut domain, bindings, fresh_req, &input_p, start..start + 512);
        bindings = b;
        fresh_logits.push(logits);
    }
    let (fresh_tokens, fresh_rows, b) = greedy_decode(
        &mut domain,
        bindings,
        fresh_req,
        fresh_logits.last().unwrap().clone(),
        PROMPT_06,
        DECODE_STEPS - 1,
    );
    bindings = b;
    assert_eq!(
        fresh_rows.len(),
        DECODE_STEPS,
        "one logits row per token"
    );
    domain.close(fresh_req).unwrap();
    let _ = domain
        .shrink_state(&mut bindings, CompletionWake::new(|| {}))
        .unwrap();

    // Restored side: same prefill, exported at 8192, state released and rows
    // freed, then restored into a new request.
    let export_req = RequestId(2);
    domain.install_input(export_req, input_p.clone()).unwrap();
    assert!(domain.open_served(&mut bindings, export_req, None).unwrap().is_empty());
    let mut export_logits: Vec<Vec<f32>> = Vec::with_capacity(16);
    for chunk in 0..16 {
        let start = chunk * 512;
        let (logits, b) = forward(&mut domain, bindings, export_req, &input_p, start..start + 512);
        bindings = b;
        export_logits.push(logits);
    }
    // The chunk logits must match the fresh side: the prefill is deterministic.
    for (i, (a, b)) in fresh_logits.iter().zip(&export_logits).enumerate() {
        assert_eq!(bits(a), bits(b), "chunk {} logit rows", i);
    }
    let held = domain.resume_state(export_req).unwrap();
    let exported = held.target().export().unwrap();
    assert_eq!(exported.position, PROMPT_06);
    drop(held);
    if restart {
        drop(domain);
        drop(bindings);
        let (d, b, _) =
            build_native_domain(&resolved.manifest, resolved.host.shared_package()).unwrap();
        domain = d;
        bindings = b;
    } else {
        domain.release_state(&[export_req]).unwrap();
        domain.close(export_req).unwrap();
        let _ = domain
            .shrink_state(&mut bindings, CompletionWake::new(|| {}))
            .unwrap();
    }

    // A new request restores the export and continues the same continuation.
    let restore_req = RequestId(3);
    domain.install_input(restore_req, input_p.clone()).unwrap();
    // The rows it opens with: the history it restores and the steps it decodes.
    let rows = (PROMPT_06 + DECODE_STEPS) as u64;
    let unchained = domain.unchained(&mut bindings).expect("no chain is under way");
    let encodes = domain
        .restore_state(unchained, restore_req, &exported, None, rows)
        .unwrap();
    assert!(
        encodes.is_empty(),
        "a restored text state encodes nothing"
    );
    let (restored_tokens, restored_rows, _) = greedy_decode(
        &mut domain,
        bindings,
        restore_req,
        export_logits.last().unwrap().clone(),
        PROMPT_06,
        DECODE_STEPS - 1,
    );
    assert_eq!(
        restored_tokens, fresh_tokens,
        "the restored state continues the same token continuation"
    );
    for (i, (a, b)) in restored_rows.iter().zip(&fresh_rows).enumerate() {
        assert_eq!(
            bits(a),
            bits(b),
            "step {} decode logits",
            i
        );
    }
    domain.close(restore_req).unwrap();

    let _ = std::fs::remove_dir_all(&dir);
}
