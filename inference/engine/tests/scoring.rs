//! Real-device check of a scoring load: a continuation a generation load of
//! the same model wrote greedily is, token by token, the scoring load's most
//! likely token, a score is reproducible, scores walked together (as the
//! service's concurrent requests are, a single-row decode among verification
//! forwards) match the same scores walked alone, and each load refuses the
//! other purpose's requests.
//!
//! `MAGNITUDE_TEST_GGUF=M.gguf [MAGNITUDE_TEST_KERNEL_CACHE=DIR] cargo test
//! --release -p magnitude-engine --test scoring -- --ignored --nocapture`.
//! Both loads run their worker on a thread of this process over the channel
//! transport, serving canonical defaults for tuning units the kernel cache
//! holds no record of.

use magnitude_engine::{
    composition::{scoring_configuration, EngineConfiguration, ReadyEngine},
    error::RequestError,
    inputs::SpecialTokens,
    options::{
        standard_service_limits, ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection,
        Serve, SCORING_WIDTH,
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
use magnitude_scheduler::prefix_cache::PrefixRetention;
use std::{
    future::Future,
    path::PathBuf,
    pin::pin,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
    thread::JoinHandle,
};

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

/// How far a log-probability scored together may be from the same one scored
/// alone, in nats. A forward's other rows may change the kernel shapes and so
/// the order of its arithmetic; a row scored against another walk's state or
/// position is off by far more.
const TOGETHER_TOLERANCE: f64 = 1e-2;

/// A plain generation load of the test's model: the base a scoring load is
/// made from, and the load that writes the continuation.
fn generation() -> EngineConfiguration {
    let model = std::env::var_os("MAGNITUDE_TEST_GGUF").expect("set MAGNITUDE_TEST_GGUF");
    EngineConfiguration {
        package: PackageOptions {
            target_shards: vec![PathBuf::from(model)],
            projector: ProjectorSelection::Disabled,
            draft: None,
        },
        model: ModelPolicy {
            method: ModelMethod::Plain,
            ..ModelPolicy::default()
        },
        context_tokens: Some(4096),
        service: standard_service_limits(),
        path: ExecutionPath::Native,
        device: DeviceRequest::Automatic,
        kernel_cache: std::env::var_os("MAGNITUDE_TEST_KERNEL_CACHE").map(PathBuf::from),
        release: ReleaseRate::of_machine(None),
        model_sha256: None,
        serve: Serve::Generation,
    }
}

/// Load `configuration` on a worker thread over the channel transport. The
/// worker ends once the engine is dropped.
fn start(configuration: EngineConfiguration) -> (ReadyEngine, JoinHandle<WorkerExit>) {
    let resolved = configuration.resolve().unwrap();
    let manifest = resolved.manifest.clone();
    let (host_end, worker_end) = channel_pair();
    let worker = std::thread::spawn(move || {
        run_worker(manifest, UnpreparedUnits::ServeDefaults, worker_end)
    });
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
/// the scoring load's steps.
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

/// Each of `prompts` scored together on `scoring` matches its score alone:
/// the same rank at every position, and log-probabilities within
/// [`TOGETHER_TOLERANCE`].
fn assert_together_matches_alone(scoring: &ReadyEngine, prompts: &[Vec<TokenId>]) {
    let alone: Vec<_> = prompts
        .iter()
        .map(|tokens| block_on(scoring.client().score(tokens.clone(), true, false)).unwrap())
        .collect();
    let together = score_together(scoring, prompts);
    let mut largest = 0.0f64;
    for (prompt, (alone, together)) in alone.iter().zip(&together).enumerate() {
        assert_eq!(alone.positions.len(), prompts[prompt].len() - 1);
        assert_eq!(together.positions.len(), alone.positions.len());
        for (index, (solo, joint)) in alone.positions.iter().zip(&together.positions).enumerate() {
            let difference = (solo.logprob - joint.logprob).abs();
            largest = largest.max(difference);
            assert_eq!(
                solo.rank,
                joint.rank,
                "prompt {prompt}, position {}: rank {} alone, {} together",
                index + 1,
                solo.rank,
                joint.rank
            );
            assert!(
                difference <= TOGETHER_TOLERANCE,
                "prompt {prompt}, position {}: log-probability {} alone, {} together",
                index + 1,
                solo.logprob,
                joint.logprob
            );
        }
    }
    eprintln!(
        "scored {} prompts together; largest log-probability difference from alone {largest:e}",
        prompts.len()
    );
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
fn a_greedy_continuation_scores_rank_one_and_scores_repeat_alone_and_together() {
    let (generating, worker) = start(generation());
    let prompt = generating
        .host()
        .tokenizer()
        .encode(PROMPT, SpecialTokens::Recognize)
        .unwrap();
    let continuation = continue_greedily(&generating, &prompt).unwrap();
    assert_eq!(continuation.len(), CONTINUATION);
    let refused = block_on(generating.client().score(prompt.clone(), false, false));
    assert!(
        matches!(refused, Err(RequestError::InvalidRequest { .. })),
        "a generation load refuses a score: {refused:?}"
    );
    drop(generating);
    eprintln!("generation worker: {:?}", worker.join().unwrap());

    let (scoring, worker) = start(scoring_configuration(generation()));
    let tokens = [prompt.as_slice(), continuation.as_slice()].concat();
    let first = block_on(scoring.client().score(tokens.clone(), true, true)).unwrap();
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
            "continuation token {offset} ({:?}) is not the scoring load's greedy choice",
            continuation[offset]
        );
    }
    let second = block_on(scoring.client().score(tokens.clone(), true, true)).unwrap();
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

    // Three walks in one scoring load's steps. One is 2W + 1 tokens long, so
    // its third step is a single-row decode; the longest is then still
    // forwarding W rows, a verification forward in the same step.
    let passage = scoring
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
    assert_together_matches_alone(&scoring, &prompts);

    let empty = block_on(scoring.client().score(Vec::new(), false, false));
    assert!(
        matches!(empty, Err(RequestError::InvalidRequest { .. })),
        "a score of no tokens is refused: {empty:?}"
    );
    let admitted = continue_greedily(&scoring, &prompt);
    assert!(
        admitted
            .as_ref()
            .is_err_and(|error| error.contains("scoring")),
        "a scoring load refuses generation: {admitted:?}"
    );
    drop(scoring);
    eprintln!("scoring worker: {:?}", worker.join().unwrap());
}
