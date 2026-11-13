// CLI runner for P3 component composition.
//
// This component imports the text:processor and compute:concurrent
// interfaces and exports wasi:cli/run. It is composed with the P3
// library components via wac to produce a single runnable binary.
//
// The imported functions are `async func`s, so each call returns a future.
// `wasi:cli/run` is a sync export, so `run` drives the command to completion
// with `wit_bindgen::block_on`.
//
// Usage (after composition):
//   wasmtime run p3_cli.wasm analyze "the quick brown fox"
//   wasmtime run p3_cli.wasm fibonacci 10
//   wasmtime run p3_cli.wasm transform uppercase "hello world"

use p3_runner_bindings::text::processor::analyzer;
use p3_runner_bindings::text::processor::transformer::{self, TransformMode};
use p3_runner_bindings::compute::concurrent::tasks;
use p3_runner_bindings::exports::wasi::cli::run::Guest;

struct Component;

impl Guest for Component {
    fn run() -> Result<(), ()> {
        let args: Vec<String> = std::env::args().collect();

        if args.len() < 2 {
            print_usage();
            return Err(());
        }

        wit_bindgen::block_on(dispatch(args))
    }
}

async fn dispatch(args: Vec<String>) -> Result<(), ()> {
    match args[1].as_str() {
        "analyze" => cmd_analyze(&args[2..]).await,
        "frequencies" => cmd_frequencies(&args[2..]).await,
        "search" => cmd_search(&args[2..]).await,
        "transform" => cmd_transform(&args[2..]).await,
        "caesar" => cmd_caesar(&args[2..]).await,
        "replace" => cmd_replace(&args[2..]).await,
        "fibonacci" | "fib" => cmd_fibonacci(&args[2..]).await,
        "factorial" | "fact" => cmd_factorial(&args[2..]).await,
        "prime" => cmd_prime(&args[2..]).await,
        "collatz" => cmd_collatz(&args[2..]).await,
        "batch" => cmd_batch(&args[2..]).await,
        other => {
            eprintln!("Unknown command: {other}");
            print_usage();
            Err(())
        }
    }
}

async fn cmd_analyze(args: &[String]) -> Result<(), ()> {
    let text = args.join(" ");
    let s = analyzer::analyze(text).await;
    println!(
        "chars={} words={} lines={} unique={} avg_len={:.1}",
        s.char_count, s.word_count, s.line_count, s.unique_word_count, s.avg_word_length
    );
    Ok(())
}

async fn cmd_frequencies(args: &[String]) -> Result<(), ()> {
    let text = args.join(" ");
    for e in &analyzer::word_frequencies(text).await {
        println!("{}: {}", e.word, e.count);
    }
    Ok(())
}

async fn cmd_search(args: &[String]) -> Result<(), ()> {
    if args.len() < 2 {
        eprintln!("Usage: search <pattern> <text...>");
        return Err(());
    }
    let positions = analyzer::search_positions(args[1..].join(" "), args[0].clone()).await;
    println!("{:?}", positions);
    Ok(())
}

async fn cmd_transform(args: &[String]) -> Result<(), ()> {
    if args.len() < 2 {
        eprintln!("Usage: transform <mode> <text...>");
        return Err(());
    }
    let mode = match args[0].as_str() {
        "upper" | "uppercase" => TransformMode::Uppercase,
        "lower" | "lowercase" => TransformMode::Lowercase,
        "reverse" => TransformMode::Reverse,
        "title" | "title-case" => TransformMode::TitleCase,
        other => {
            eprintln!("Unknown mode: {other}. Use: uppercase, lowercase, reverse, title-case");
            return Err(());
        }
    };
    println!("{}", transformer::transform(args[1..].join(" "), mode).await);
    Ok(())
}

async fn cmd_caesar(args: &[String]) -> Result<(), ()> {
    if args.len() < 2 {
        eprintln!("Usage: caesar <shift> <text...>");
        return Err(());
    }
    let shift: i32 = args[0].parse().unwrap_or(0);
    println!("{}", transformer::caesar_cipher(args[1..].join(" "), shift).await);
    Ok(())
}

async fn cmd_replace(args: &[String]) -> Result<(), ()> {
    if args.len() < 3 {
        eprintln!("Usage: replace <pattern> <replacement> <text...>");
        return Err(());
    }
    println!(
        "{}",
        transformer::replace_all(args[2..].join(" "), args[0].clone(), args[1].clone()).await
    );
    Ok(())
}

async fn cmd_fibonacci(args: &[String]) -> Result<(), ()> {
    let n: u32 = args.first().and_then(|s| s.parse().ok()).unwrap_or(10);
    println!("fibonacci({}) = {}", n, tasks::fibonacci(n).await);
    Ok(())
}

async fn cmd_factorial(args: &[String]) -> Result<(), ()> {
    let n: u32 = args.first().and_then(|s| s.parse().ok()).unwrap_or(10);
    println!("factorial({}) = {}", n, tasks::factorial(n).await);
    Ok(())
}

async fn cmd_prime(args: &[String]) -> Result<(), ()> {
    let n: u64 = args.first().and_then(|s| s.parse().ok()).unwrap_or(17);
    let result = if tasks::is_prime(n).await { "prime" } else { "not prime" };
    println!("{n} is {result}");
    Ok(())
}

async fn cmd_collatz(args: &[String]) -> Result<(), ()> {
    let n: u64 = args.first().and_then(|s| s.parse().ok()).unwrap_or(27);
    println!("collatz({}) = {} steps", n, tasks::collatz_steps(n).await);
    Ok(())
}

async fn cmd_batch(args: &[String]) -> Result<(), ()> {
    let numbers: Vec<u64> = args.iter().filter_map(|s| s.parse().ok()).collect();
    if numbers.is_empty() {
        eprintln!("Usage: batch <n1> <n2> ...");
        return Err(());
    }
    for r in &tasks::compute_batch(numbers).await {
        println!("{}: input={} output={}", r.task_name, r.input, r.output);
    }
    Ok(())
}

fn print_usage() {
    eprintln!("P3 Component CLI Runner");
    eprintln!();
    eprintln!("Text commands:");
    eprintln!("  analyze <text>                    Text statistics");
    eprintln!("  frequencies <text>                Word frequency table");
    eprintln!("  search <pattern> <text>           Find pattern positions");
    eprintln!("  transform <mode> <text>           uppercase/lowercase/reverse/title-case");
    eprintln!("  caesar <shift> <text>             Caesar cipher");
    eprintln!("  replace <pat> <repl> <text>       Replace pattern");
    eprintln!();
    eprintln!("Compute commands:");
    eprintln!("  fibonacci <n>                     Fibonacci number");
    eprintln!("  factorial <n>                     Factorial");
    eprintln!("  prime <n>                         Primality test");
    eprintln!("  collatz <n>                       Collatz steps");
    eprintln!("  batch <n1> <n2> ...               Batch computation");
}

p3_runner_bindings::export!(Component with_types_in p3_runner_bindings);
