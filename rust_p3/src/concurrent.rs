// Async concurrent-tasks P3 component.
//
// Exports async functions for mathematical computations that can be
// dispatched concurrently when composed with other async components.
//
//   fibonacci(n)       — iterative Fibonacci (saturating)
//   factorial(n)       — n! (saturating)
//   is_prime(n)        — trial-division primality test
//   collatz_steps(n)   — steps to reach 1 under the Collatz map
//   compute_batch(ns)  — runs all three on every element of ns

use concurrent_tasks_bindings::exports::compute::concurrent::tasks::{
    ComputationResult, Guest,
};

struct Component;

// ---------------------------------------------------------------------------
// Core logic
// ---------------------------------------------------------------------------

impl Component {
    fn do_fibonacci(n: u32) -> u64 {
        if n <= 1 {
            return n as u64;
        }
        let (mut a, mut b) = (0u64, 1u64);
        for _ in 2..=n {
            let next = a.saturating_add(b);
            a = b;
            b = next;
        }
        b
    }

    fn do_factorial(n: u32) -> u64 {
        (1..=n as u64).fold(1u64, |acc, x| acc.saturating_mul(x))
    }

    fn do_is_prime(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n < 4 {
            return true;
        }
        if n % 2 == 0 || n % 3 == 0 {
            return false;
        }
        let mut i = 5u64;
        while i.saturating_mul(i) <= n {
            if n % i == 0 || n % (i + 2) == 0 {
                return false;
            }
            i += 6;
        }
        true
    }

    fn do_collatz_steps(n: u64) -> u32 {
        if n <= 1 {
            return 0;
        }
        let mut steps = 0u32;
        let mut current = n;
        while current != 1 {
            current = if current % 2 == 0 {
                current / 2
            } else {
                current.saturating_mul(3).saturating_add(1)
            };
            steps += 1;
        }
        steps
    }

    fn do_compute_batch(numbers: &[u64]) -> Vec<ComputationResult> {
        numbers
            .iter()
            .flat_map(|&n| {
                vec![
                    ComputationResult {
                        task_name: "fibonacci".into(),
                        input: n,
                        output: Self::do_fibonacci(n as u32),
                    },
                    ComputationResult {
                        task_name: "is_prime".into(),
                        input: n,
                        output: u64::from(Self::do_is_prime(n)),
                    },
                    ComputationResult {
                        task_name: "collatz_steps".into(),
                        input: n,
                        output: Self::do_collatz_steps(n) as u64,
                    },
                ]
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Guest trait (async on wasm32, sync on host)
// ---------------------------------------------------------------------------

impl Guest for Component {
    #[cfg(target_arch = "wasm32")]
    async fn fibonacci(n: u32) -> u64 {
        Self::do_fibonacci(n)
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn fibonacci(n: u32) -> u64 {
        Self::do_fibonacci(n)
    }

    #[cfg(target_arch = "wasm32")]
    async fn factorial(n: u32) -> u64 {
        Self::do_factorial(n)
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn factorial(n: u32) -> u64 {
        Self::do_factorial(n)
    }

    #[cfg(target_arch = "wasm32")]
    async fn is_prime(n: u64) -> bool {
        Self::do_is_prime(n)
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn is_prime(n: u64) -> bool {
        Self::do_is_prime(n)
    }

    #[cfg(target_arch = "wasm32")]
    async fn collatz_steps(n: u64) -> u32 {
        Self::do_collatz_steps(n)
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn collatz_steps(n: u64) -> u32 {
        Self::do_collatz_steps(n)
    }

    #[cfg(target_arch = "wasm32")]
    async fn compute_batch(numbers: Vec<u64>) -> Vec<ComputationResult> {
        Self::do_compute_batch(&numbers)
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn compute_batch(numbers: Vec<u64>) -> Vec<ComputationResult> {
        Self::do_compute_batch(&numbers)
    }
}

concurrent_tasks_bindings::export!(Component with_types_in concurrent_tasks_bindings);
