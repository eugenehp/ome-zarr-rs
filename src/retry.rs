//! Asking a server again. A read of a large image is thousands of requests, and
//! one refused for a moment should not end it: a server pushing back (429) or
//! failing for now (500, 502, 503, 504) is asked again after its `Retry-After`,
//! else after a jittered wait doubling up to 16 times `base`; a request that
//! got no whole response, a couple of times. Anything else is an answer.

use std::time::Duration;

/// Why an attempt failed, and so whether another is worth making.
pub(crate) enum Failure<E> {
    /// The server pushed back, naming a wait or not.
    Refused(Option<Duration>, E),
    /// No whole response: a dropped connection, a timeout.
    Lost(E),
    /// Asking again would not help.
    Fatal(E),
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Policy {
    pub refused: u32,
    pub lost: u32,
    pub base: Duration,
}

impl Default for Policy {
    fn default() -> Self {
        Policy {
            refused: 6,
            lost: 2,
            base: Duration::from_millis(500),
        }
    }
}

/// In [0.5, 1): so that many requests refused together do not return together.
fn jitter() -> f64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0x9E37_79B9_7F4A_7C15);
    let mut z = COUNTER.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    0.5 + 0.5 * ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}

/// `once`, until it succeeds, fails for good, or `policy` runs out.
pub(crate) fn retry<T, E>(
    policy: Policy,
    mut once: impl FnMut() -> Result<T, Failure<E>>,
) -> Result<T, E> {
    let (mut refused, mut lost) = (0, 0);
    loop {
        let wait = match once() {
            Ok(value) => return Ok(value),
            Err(Failure::Fatal(e)) => return Err(e),
            Err(Failure::Refused(_, e)) if refused >= policy.refused => return Err(e),
            Err(Failure::Lost(e)) if lost >= policy.lost => return Err(e),
            Err(Failure::Refused(after, _)) => {
                let doubled = policy.base.saturating_mul(1 << refused.min(4));
                refused += 1;
                after.unwrap_or_else(|| doubled.mul_f64(jitter()))
            }
            Err(Failure::Lost(_)) => {
                lost += 1;
                policy.base.saturating_mul(lost).mul_f64(jitter())
            }
        };
        std::thread::sleep(wait);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    const QUICK: Policy = Policy {
        refused: 6,
        lost: 2,
        base: Duration::from_millis(1),
    };

    /// Fails `failures` times as `how`, then answers; counts the attempts.
    fn flaky(
        failures: u32,
        how: fn() -> Failure<&'static str>,
        asked: &Cell<u32>,
    ) -> Result<u32, &'static str> {
        retry(QUICK, || {
            asked.set(asked.get() + 1);
            if asked.get() <= failures {
                Err(how())
            } else {
                Ok(asked.get())
            }
        })
    }

    #[test]
    fn failures_for_now_are_asked_again() {
        let asked = Cell::new(0);
        assert_eq!(flaky(3, || Failure::Refused(None, "503"), &asked), Ok(4));
        let asked = Cell::new(0);
        let after = || Failure::Refused(Some(Duration::from_millis(1)), "429");
        assert_eq!(flaky(2, after, &asked), Ok(3));
        let asked = Cell::new(0);
        assert_eq!(flaky(2, || Failure::Lost("reset"), &asked), Ok(3));
    }

    #[test]
    fn other_answers_are_not_and_retries_end() {
        let asked = Cell::new(0);
        assert_eq!(flaky(9, || Failure::Fatal("403"), &asked), Err("403"));
        assert_eq!(asked.get(), 1);
        let asked = Cell::new(0);
        assert_eq!(
            flaky(99, || Failure::Refused(None, "503"), &asked),
            Err("503")
        );
        assert_eq!(asked.get(), 7);
        let asked = Cell::new(0);
        assert_eq!(flaky(99, || Failure::Lost("reset"), &asked), Err("reset"));
        assert_eq!(asked.get(), 3);
    }
}
