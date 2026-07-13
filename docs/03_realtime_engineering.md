# Realtime Engineering

## What To Build

The realtime stage simulates a producer-consumer pipeline:

```text
audio producer -> queue -> mouth inference/control consumer -> output log
```

Command:

```powershell
python scripts/07_realtime_rule_demo.py --config configs/realtime/realtime-rule.yaml
```

## Why It Matters

Realtime systems care about latency and stability, not just accuracy. A mouth animation system must avoid blocking audio capture, inference, rendering, and actuator control in one long serial path.

## Concepts

- Queue: decouples producer and consumer.
- FPS: output frame rate, usually 25 or 30 for face animation.
- Latency: time from audio frame creation to mouth parameter output.
- Backpressure: queues should be bounded so latency does not grow without limit.
- Smoothing: reduces jitter but adds delay.
- Overflow policy: `block`, `drop_oldest`, or `drop_newest` controls latency versus frame retention.
- Stop propagation: blocked producers and consumers must wake when the pipeline closes.
- Error propagation: a worker exception must reach the output consumer instead of silently hanging a thread.

## Expected Output

```text
outputs/logs/realtime_rule_queue.csv
outputs/reports/realtime_rule_report.json
```

## Interview Explanation

I would describe the system as a low-latency streaming pipeline. Audio capture, feature extraction, model inference, and output control should be separate stages connected by bounded queues. The system should monitor FPS, mean latency, P95 latency, and dropped frames.

The realtime report also records input/output queue accepted and dropped counts. Unit tests cover full queues, both drop policies, stop wakeup, timeout, and worker exception propagation.
