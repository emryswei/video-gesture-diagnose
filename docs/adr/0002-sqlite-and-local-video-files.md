# Store audit data in SQLite and retained videos as local files

V1.1 uses SQLite for Audit Records, job state, and version metadata, while opt-in source videos remain ordinary files in the operating system's application data directory. This keeps the single-user demo self-contained and queryable without putting large media BLOBs in the database, writing user data into the project directory, or introducing a production storage service.

## Consequences

- The demo remains local to one machine and does not provide multi-user access or remote durability.
- SQLite stores relative video paths rather than machine-specific absolute paths.
- Moving to shared or cloud storage later will require a storage migration while preserving the Audit Record contract.

