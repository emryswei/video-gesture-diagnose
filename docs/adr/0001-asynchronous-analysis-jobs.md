# Use asynchronous in-memory analysis jobs for V1

Video inference and first-time model loading can take several minutes, making a synchronous upload request vulnerable to client and proxy timeouts. V1 therefore creates an asynchronous analysis job, returns its ID immediately, and exposes status polling while keeping job state in memory; the job-oriented API will move to local persistence in V1.1 without making persistence or history part of V1.

## Considered options

- A synchronous request was rejected because its lifetime is coupled to a long-running client connection.
- A persistent job queue was deferred because restart recovery and history are outside the V1 scope.

## Consequences

- Clients create an analysis and poll its status until it succeeds or fails.
- V1 permits only one queued or running analysis; concurrent creation returns HTTP `409 Conflict` rather than building a multi-job queue.
- Successful and failed job results remain available in memory for 30 minutes, then expire with subsequent queries returning HTTP `410 Gone`.
- All job state is lost when the server restarts.
- Temporary video data is deleted when the job succeeds or fails.
