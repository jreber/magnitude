//! Real-device check of one load serving both generation and scoring: a
//! continuation the load wrote greedily is, token by token, the same load's
//! most likely token, a score is reproducible, scores walked together (as
//! the service's concurrent requests are, a single-row decode beside wider
//! ones in one scoring flight) match the same scores walked alone within
//! rounding, and the load writes the same continuation after scoring as
//! before.
//!
//! `MAGNITUDE_TEST_GGUF=M.gguf [MAGNITUDE_TEST_KERNEL_CACHE=DIR] cargo test
//! --release -p magnitude-engine --test scoring -- --ignored --nocapture`,
//! on any supported model. The load runs its worker on a thread of this
//! process over the channel transport, serving canonical defaults for
//! tuning units the kernel cache holds no record of, and loads as a load in
//! its own process does ([`model_load`]): the prices its plan waits on are
//! measured into the kernel cache first, once per machine.

use magnitude_engine::{
    composition::{scoring_rows, EngineConfiguration, ReadyEngine},
    error::RequestError,
    inputs::SpecialTokens,
    options::{
        standard_service_limits, ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection,
    },
    worker::{
        connect_worker, run_worker, transport::channel_pair, RequestEvent, RequestOptions,
        ScoreOutcome, WorkerExit,
    },
    LoadAttempt,
};
use magnitude_executor::{
    platform::{DeviceRequest, ReleaseRate},
    ExecutionPath, UnpreparedUnits,
};
use magnitude_generation::{EndOfGeneration, MethodChoice, Options, Sampling, Shaping, TokenId};
use magnitude_scheduler::{
    prefix_cache::PrefixRetention,
    scoring::{SCORING_ROWS, SCORING_WIDTH},
};
use std::{
    future::Future,
    path::PathBuf,
    pin::pin,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
    thread::JoinHandle,
};

mod model_load;

const PROMPT: &str = "The following is a short, factual account of how the first transatlantic \
telegraph cable was laid in the nineteenth century:";

/// The tokens the generation load continues the prompt with.
const CONTINUATION: usize = 16;

/// Text the scores walked together are cut from: long enough for the longest.
const PASSAGE: &str = "The lighthouse of Alexandria stood on the island of Pharos at the \
entrance to the city's harbour. It was built in the third century before the common era, during \
the reigns of the first two Ptolemaic kings, and it rose in three stages: a square base, an \
octagonal middle and a round top, where a fire burned at night. Ancient writers said that its \
light could be seen from far out at sea, and travellers described a statue at its summit, \
perhaps of Zeus or of Poseidon. The tower survived for well over a thousand years. Earthquakes \
in the tenth and fourteenth centuries damaged it badly, and by the fifteenth century its stones \
had been used to build a fortress on the same site. Divers working in the harbour in the late \
twentieth century found large blocks, columns and statues on the sea floor nearby, and some of \
them are thought to have come from the lighthouse itself. Its name survives in the word for a \
lighthouse in several languages, including French, Italian, Spanish and Portuguese.";

/// How far, on average over a walk's positions, a log-probability scored
/// together may be from the same one scored alone, in nats. Walks together
/// launch more rows a step than a walk alone, and a launch's rows choose its
/// kernels' forms and tiling (past `VERIFY_ROWS` the matrix forms stage
/// activations as 16-bit floats, and where a row falls in a tile follows
/// its companions), so the arithmetic rounds otherwise, each step's state
/// carries that into the next, and a model routing its rows to experts may
/// route a near-tied row elsewhere: no single position is bounded, as such
/// a row may move by a nat or more. Measured on Metal with this test's
/// walks, the largest walk's mean and the largest single difference:
/// Qwen3.8 27B Q4 (dense) 0.015 and 0.10; Qwen3.8 Flash Next UD-Q2_K_XL
/// (routed experts) 0.14 and, over loads, 0.84 to 1.5. A walk scored
/// against another walk's state or positions loses its context: its
/// log-probabilities, which average 1.3 to 1.9 nats here, move by about
/// their own size.
const TOGETHER_TOLERANCE: f64 = 0.5;

/// A plain load of the test's model, exporting the logits rows its service
/// scores with: the one load that writes the continuation and scores it.
fn generation() -> EngineConfiguration {
    let model = std::env::var_os("MAGNITUDE_TEST_GGUF").expect("set MAGNITUDE_TEST_GGUF");
    let service = standard_service_limits();
    EngineConfiguration {
        package: PackageOptions {
            target_shards: vec![PathBuf::from(model)],
            projector: ProjectorSelection::Disabled,
            draft: None,
        },
        model: ModelPolicy {
            method: ModelMethod::Plain,
            exported_logits_rows: scoring_rows(&service),
            ..ModelPolicy::default()
        },
        context_tokens: Some(4096),
        service,
        path: ExecutionPath::Native,
        device: DeviceRequest::Automatic,
        kernel_cache: Some(model_load::kernel_cache()),
        release: ReleaseRate::of_machine(None),
        model_sha256: None,
    }
}

/// Load `configuration` on a worker thread over the channel transport. The
/// worker ends once the engine is dropped.
fn start(configuration: EngineConfiguration) -> (ReadyEngine, JoinHandle<WorkerExit>) {
    let resolved = model_load::resolve(configuration);
    let manifest = resolved.manifest.clone();
    let (host_end, worker_end) = channel_pair();
    let worker = seismic::threads::spawn("magnitude-worker", move || {
        run_worker(manifest, UnpreparedUnits::ServeDefaults, worker_end)
    })
    .expect("spawn the worker thread");
    let LoadAttempt::Loaded(connection) =
        connect_worker(host_end, None, |progress| eprintln!("load: {progress:?}")).unwrap()
    else {
        panic!("a load serving canonical defaults does not stop unprepared");
    };
    let engine = ReadyEngine::new(resolved.host, *connection).unwrap();
    (engine, worker)
}

/// Drive a future on this thread; the engine client is executor-agnostic.
fn block_on<T>(future: impl Future<Output = T>) -> T {
    struct Unpark(std::thread::Thread);
    impl Wake for Unpark {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

/// Score `prompts` together on `engine`, with the top-1 token at each
/// position: every request is sent before any is awaited, so the walks share
/// the load's scoring flights.
fn score_together(engine: &ReadyEngine, prompts: &[Vec<TokenId>]) -> Vec<ScoreOutcome> {
    let mut scores: Vec<_> = prompts
        .iter()
        .map(|tokens| Box::pin(engine.client().score(tokens.clone(), true, false)))
        .collect();
    let mut outcomes: Vec<Option<ScoreOutcome>> = prompts.iter().map(|_| None).collect();
    block_on(std::future::poll_fn(|context| {
        for (score, outcome) in scores.iter_mut().zip(outcomes.iter_mut()) {
            if outcome.is_some() {
                continue;
            }
            if let Poll::Ready(result) = score.as_mut().poll(context) {
                *outcome = Some(result.expect("a score walked together"));
            }
        }
        if outcomes.iter().all(Option::is_some) {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }));
    outcomes.into_iter().map(Option::unwrap).collect()
}

/// Each of `prompts` scored together on `engine` matches its score alone
/// within [`TOGETHER_TOLERANCE`] on average over its positions. Single
/// positions and ranks are not compared: rounding may route a row
/// otherwise, and a token within rounding of its neighbours trades places
/// with them.
fn assert_together_matches_alone(engine: &ReadyEngine, prompts: &[Vec<TokenId>]) {
    let alone: Vec<_> = prompts
        .iter()
        .map(|tokens| block_on(engine.client().score(tokens.clone(), true, false)).unwrap())
        .collect();
    let together = score_together(engine, prompts);
    for (prompt, (alone, together)) in alone.iter().zip(&together).enumerate() {
        assert_eq!(alone.positions.len(), prompts[prompt].len() - 1);
        assert_eq!(together.positions.len(), alone.positions.len());
        let differences: Vec<f64> = alone
            .positions
            .iter()
            .zip(&together.positions)
            .map(|(solo, joint)| (solo.logprob - joint.logprob).abs())
            .collect();
        let mean = differences.iter().sum::<f64>() / differences.len() as f64;
        let largest = differences.iter().copied().fold(0.0, f64::max);
        let size = alone.positions.iter().map(|position| -position.logprob).sum::<f64>() / differences.len() as f64;
        eprintln!(
            "prompt {prompt} scored together: log-probability difference from alone: mean {mean:e}, \
             largest {largest:e}; mean log-probability alone -{size:.3}"
        );
        assert!(
            mean <= TOGETHER_TOLERANCE,
            "prompt {prompt}: log-probabilities {mean} from alone on average"
        );
    }
}

/// `CONTINUATION` greedy tokens after `prompt` on `engine`.
fn continue_greedily(engine: &ReadyEngine, prompt: &[TokenId]) -> Result<Vec<TokenId>, String> {
    let host = engine.host();
    let input = host
        .prepare_input(prompt.to_vec(), &[])
        .map_err(|error| error.to_string())?;
    let definition = host.definition();
    let options = Options {
        max_tokens: CONTINUATION,
        output_capacity: 16,
        context_limit: usize::try_from(definition.decoder.context_limit).unwrap(),
        vocabulary: usize::try_from(definition.decoder.vocabulary).unwrap(),
        stop_tokens: host.tokenizer().stop_tokens().clone(),
        suppressed_tokens: host.tokenizer().suppressed_tokens().clone(),
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
    };
    block_on(async {
        let mut request = engine
            .client()
            .admit(RequestOptions {
                input,
                options,
                constraint: None,
                retention: PrefixRetention::Transient,
                output_capacity: 16,
            })
            .await
            .map_err(|error| error.to_string())?;
        let mut output = Vec::new();
        loop {
            match request.receive().await {
                Some(RequestEvent::Output(batch)) => {
                    output.extend(batch.into_iter().map(|token| token.token))
                }
                Some(RequestEvent::Completed { .. }) => return Ok(output),
                Some(RequestEvent::Failed(error)) => return Err(error.to_string()),
                None => return Err("the request ended without a terminal event".into()),
            }
        }
    })
}

#[test]
#[ignore = "needs a model on a device: set MAGNITUDE_TEST_GGUF"]
fn one_load_scores_its_greedy_continuation_rank_one_and_scores_repeat_alone_and_together() {
    let (engine, worker) = start(generation());
    assert_eq!(engine.ready_info().model.exported_logits_rows, SCORING_ROWS);
    let prompt = engine
        .host()
        .tokenizer()
        .encode(PROMPT, SpecialTokens::Recognize)
        .unwrap();
    let continuation = continue_greedily(&engine, &prompt).unwrap();
    assert_eq!(continuation.len(), CONTINUATION);

    let tokens = [prompt.as_slice(), continuation.as_slice()].concat();
    let first = block_on(engine.client().score(tokens.clone(), true, true)).unwrap();
    assert_eq!(first.positions.len(), tokens.len() - 1);
    assert!(first.next.is_some(), "the next token was asked for");
    for (index, position) in first.positions.iter().enumerate() {
        assert!(
            position.logprob <= 0.0,
            "position {}: {position:?}",
            index + 1
        );
    }
    for (offset, position) in first.positions[prompt.len() - 1..].iter().enumerate() {
        assert_eq!(
            (position.rank, position.top),
            (1, None),
            "continuation token {offset} ({:?}) is not the load's greedy choice",
            continuation[offset]
        );
    }
    let second = block_on(engine.client().score(tokens.clone(), true, true)).unwrap();
    let logprobs = |outcome: &ScoreOutcome| {
        outcome
            .positions
            .iter()
            .map(|position| position.logprob.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        logprobs(&first),
        logprobs(&second),
        "a score repeats exactly"
    );
    assert_eq!(first, second);
    eprintln!(
        "scored {} positions; continuation log-probability {:.4}",
        first.positions.len(),
        first.positions[prompt.len() - 1..]
            .iter()
            .map(|position| position.logprob)
            .sum::<f64>()
    );

    // Three walks in the load's scoring flights. One is 2W + 1 tokens long,
    // so its third flight forwards a single row; the longest then still
    // forwards W rows, in the same flight.
    let passage = engine
        .host()
        .tokenizer()
        .encode(PASSAGE, SpecialTokens::Recognize)
        .unwrap();
    let (offset, longest) = (SCORING_WIDTH / 2, 3 * SCORING_WIDTH + 4);
    assert!(
        passage.len() >= offset + longest,
        "the passage is {} tokens, fewer than the {} the scores walked together are cut from",
        passage.len(),
        offset + longest
    );
    let decoding = passage[..2 * SCORING_WIDTH + 1].to_vec();
    assert_eq!(decoding.len() % SCORING_WIDTH, 1);
    let prompts = [
        tokens.clone(),
        decoding,
        passage[offset..offset + longest].to_vec(),
    ];
    assert_together_matches_alone(&engine, &prompts);

    let empty = block_on(engine.client().score(Vec::new(), false, false));
    assert!(
        matches!(empty, Err(RequestError::InvalidRequest { .. })),
        "a score of no tokens is refused: {empty:?}"
    );
    // The load that scored still generates, as it did before it scored: one
    // instance served both, and no other was loaded.
    assert_eq!(
        continue_greedily(&engine, &prompt).unwrap(),
        continuation,
        "the load writes the same continuation after scoring"
    );
    assert!(!worker.is_finished(), "the one worker served every request");
    drop(engine);
    eprintln!("worker: {:?}", worker.join().unwrap());
}
