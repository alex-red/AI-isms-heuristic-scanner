# Retry behavior in the upload client

The upload client retries a request when the server returns 429, 502, 503 or 504, or when the connection resets before any response bytes arrive. It does not retry 4xx errors other than 429, because those indicate a problem with the request itself.

Delays follow exponential backoff with full jitter. The first retry waits a random time between 0 and 500 ms, the second between 0 and 1 s, and so on, capped at 30 s. If the server sends a `Retry-After` header, the client waits at least that long.

The client gives up after five attempts or 90 seconds in total, whichever comes first, and raises `UploadError` with the last response attached. Multipart uploads retry per part, so a failure in part 7 does not resend parts 1 to 6.

Retries are only safe because each part carries an `Idempotency-Key` derived from the file hash and part number. If you replace the transport layer, keep that header, or a retried part can be stored twice.

To turn retries off for debugging, pass `max_attempts=1`. To change the cap, set `max_delay`. Both are arguments to `UploadClient`, not environment variables.
