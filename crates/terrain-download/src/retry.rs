//! Trying again when the network, rather than the data, is what failed.
//!
//! A download of any size is hundreds of thousands of range requests spread
//! over tens of minutes, and the tool has no resume within a block: one failed
//! request used to abort the run and throw away everything fetched since it
//! started. That is not a hypothetical. A run over the Howe Sound box died at
//! block 66 of 672, ninety seconds in, on `error sending request` -- the
//! endpoint was healthy again immediately afterwards, answering five range
//! requests in under 0.2 s each. Nothing was wrong except the moment.
//!
//! What is retried matters as much as that it is. `error_for_status` is applied
//! inside `async-tiff`'s reader, so a request that reached the server and was
//! refused arrives here as an error carrying a status, and one that never
//! completed arrives with none. Retrying the first kind is mostly pointless and
//! occasionally harmful -- a 404 for an asset that does not exist would be
//! asked for five times before failing with the same message, five times slower
//! -- so only the second kind, plus the two statuses that explicitly mean "come
//! back later", are tried again.

use std::future::Future;
use std::time::Duration;

/// How many times a request is made before its failure is taken as real.
///
/// Five attempts with the backoff below spans about four seconds. That is
/// chosen against the observed failure -- a blip lasting less than one second
/// between two healthy periods -- and not against a sustained outage, which no
/// number of retries fixes and which should fail the run promptly.
pub const ATTEMPTS: usize = 5;

/// How long to wait before the second attempt. Each wait after that doubles,
/// so the sequence is 250 ms, 500 ms, 1 s, 2 s.
const FIRST_BACKOFF: Duration = Duration::from_millis(250);

/// Whether a failed request is worth making again.
///
/// A status means the server answered, and its answer will not change for the
/// same request -- except for 5xx, which is the server saying it is having
/// trouble, and 429, which is it asking to be asked less often. No status means
/// the request never completed: a reset connection, a DNS failure, a body that
/// stopped arriving. That is the case this whole module exists for.
pub fn is_transient(error: &reqwest::Error) -> bool {
    match error.status() {
        Some(status) => {
            status.is_server_error() || status == reqwest::StatusCode::TOO_MANY_REQUESTS
        }
        None => true,
    }
}

/// Whether a failure that surfaced through `async-tiff` was a network failure.
///
/// Everything else it can return -- a short read, a bad tag, a tile index out
/// of range, a decode failure -- is a fact about the file that will hold on
/// every attempt.
pub fn is_transient_tiff(error: &async_tiff::error::AsyncTiffError) -> bool {
    matches!(
        error,
        async_tiff::error::AsyncTiffError::ReqwestError(error) if is_transient(error)
    )
}

/// Calls `attempt` until it succeeds, fails in a way retrying will not fix, or
/// runs out of attempts.
///
/// `what` names the thing being fetched and appears in the warning logged
/// before each wait. A run that retries silently looks identical to a run that
/// is merely slow, and the difference is worth seeing in a log that may be the
/// only record of a download nobody watched.
pub async fn retrying<T, E, F, Fut>(
    what: impl std::fmt::Display,
    transient: impl Fn(&E) -> bool,
    mut attempt: F,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let mut wait = FIRST_BACKOFF;
    let mut tries = 1;
    loop {
        let error = match attempt().await {
            Ok(value) => return Ok(value),
            Err(error) => error,
        };
        if tries >= ATTEMPTS || !transient(&error) {
            return Err(error);
        }
        log::warn!(
            "{what} failed on attempt {tries} of {ATTEMPTS} ({error}); \
             retrying in {} ms",
            wait.as_millis()
        );
        tokio::time::sleep(wait).await;
        wait *= 2;
        tries += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// A stand-in for a request that fails a given number of times first.
    async fn flaky(failures: &Cell<usize>, error: &'static str) -> Result<u32, TestError> {
        if failures.get() > 0 {
            failures.set(failures.get() - 1);
            return Err(TestError(error));
        }
        Ok(7)
    }

    #[derive(Debug)]
    struct TestError(&'static str);

    impl std::fmt::Display for TestError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    fn transient(error: &TestError) -> bool {
        error.0 == "transient"
    }

    #[tokio::test(start_paused = true)]
    async fn a_request_that_recovers_returns_its_value() {
        let failures = Cell::new(ATTEMPTS - 1);
        let got = retrying("thing", transient, || flaky(&failures, "transient")).await;
        assert_eq!(got.expect("should have recovered"), 7);
        assert_eq!(failures.get(), 0, "every attempt should have been used");
    }

    #[tokio::test(start_paused = true)]
    async fn a_request_that_never_recovers_gives_up() {
        let failures = Cell::new(ATTEMPTS + 10);
        let got = retrying("thing", transient, || flaky(&failures, "transient")).await;
        assert!(got.is_err(), "should have given up");
        assert_eq!(
            failures.get(),
            10,
            "should have stopped after exactly {ATTEMPTS} attempts"
        );
    }

    /// The point of classifying rather than retrying everything: a permanent
    /// failure costs one attempt, not five, and fails with its own message.
    #[tokio::test(start_paused = true)]
    async fn a_permanent_failure_is_not_retried() {
        let failures = Cell::new(ATTEMPTS + 10);
        let got = retrying("thing", transient, || flaky(&failures, "permanent")).await;
        assert!(got.is_err(), "should have failed");
        assert_eq!(
            failures.get(),
            ATTEMPTS + 9,
            "should have tried exactly once"
        );
    }

    #[test]
    fn statuses_that_mean_come_back_later_are_the_only_ones_retried() {
        // Built through a real response so the classification is exercised on
        // the error type the tool will actually see, not on a mock of it.
        let error = |code: u16| {
            http::Response::builder()
                .status(code)
                .body("")
                .map(reqwest::Response::from)
                .expect("failed to build a response")
                .error_for_status()
                .expect_err("should be an error status")
        };
        assert!(is_transient(&error(500)), "500 should be retried");
        assert!(is_transient(&error(503)), "503 should be retried");
        assert!(is_transient(&error(429)), "429 should be retried");
        assert!(!is_transient(&error(404)), "404 should not be retried");
        assert!(!is_transient(&error(403)), "403 should not be retried");
    }
}
