We moved our build cache from S3 to a local NVMe volume on the CI runners in March. Cold builds went from 14 minutes to just under 9, mostly because the dependency layer no longer has to come over the network on every job.

The catch was eviction. Our first version used a plain LRU with a 200 GB cap, and the monorepo's Rust targets alone are about 140 GB, so any branch that touched the protobuf definitions flushed almost everything else. We switched to a two-tier policy: anything built from `main` in the last 48 hours is pinned, and feature branches compete for the remaining 60 GB. That cut cache misses on `main` from roughly one in four to one in twenty.

Two things I'd do differently. I would measure hit rate per target before choosing a cap, rather than guessing from total disk usage; and I would not have put the eviction logic in a bash cron job. It worked, but when Priya tried to add metrics she had to rewrite it in Go anyway.

If you run something similar, watch the runner's disk-full alerts for the first week. Ours fired twice, both times because a runaway test wrote 30 GB of logs into the cache directory.
