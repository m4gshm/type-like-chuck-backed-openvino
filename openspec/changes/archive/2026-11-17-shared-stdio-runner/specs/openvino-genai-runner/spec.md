# Spec Delta

## MODIFIED Requirements

### Requirement: NDJSON stdio protocol
The runner SHALL read one completion request per stdin line as a JSON object and SHALL write newline-delimited JSON events to stdout: one `token` event per generated piece (with the text and the request `id`), one `done` event at the end of a generation, and `error` events on failures. Human-readable logs SHALL go to stderr only. stdout SHALL carry no non-JSON output. When started with `--listen <endpoint>` the same protocol SHALL be served over a local named pipe (Windows) or unix domain socket (Linux) per the `runner-sharing` capability, with no TCP port and no API key.

#### Scenario: Streaming generation
- **WHEN** a completion request is written to the runner's stdin
- **THEN** the extension receives zero or more `token` events followed by a `done` event on stdout, in order

#### Scenario: Logs never corrupt the stream
- **WHEN** the runner logs diagnostics during generation
- **THEN** the log lines appear on stderr and stdout remains valid NDJSON

#### Scenario: Queued requests
- **WHEN** a `generate` request arrives while a generation is in progress
- **THEN** the runner queues it and processes requests strictly in arrival order, each ending with a `done` or `error` event carrying its request `id`
- **AND** `token` events carry the `id` of the request being generated, so a multi-client (listen-mode) session never routes tokens to the wrong client

#### Scenario: Cancel by request id
- **WHEN** a `cancel` request with an integer `id` arrives and that request is queued
- **THEN** the runner removes it from the queue and emits a `done` event with `finishReason:"cancel"` for that id immediately
- **WHEN** the cancelled request is the one currently generating
- **THEN** generation stops and a `done` event with `finishReason:"cancel"` follows
- **WHEN** a `cancel` request without an id arrives
- **THEN** the runner cancels the currently active generation

#### Scenario: Shutdown drains the queue
- **WHEN** a `shutdown` request arrives while requests are still queued
- **THEN** every queued request is answered with a `done` event with `finishReason:"cancel"`, the active generation stops with `finishReason:"cancel"`, and the runner exits
