# Realtime Engineering

## What To Build

The realtime stage simulates a producer-consumer pipeline:

```text
audio producer -> queue -> mouth inference/control consumer -> output log
```

Command:

```powershell
python scripts/07_realtime_rule_demo.py --config configs/realtime_rule.yaml
```

## Why It Matters

Realtime systems care about latency and stability, not just accuracy. A mouth animation system must avoid blocking audio capture, inference, rendering, and actuator control in one long serial path.

## Concepts

- Queue: decouples producer and consumer.
- FPS: output frame rate, usually 25 or 30 for face animation.
- Latency: time from audio frame creation to mouth parameter output.
- Backpressure: queues should be bounded so latency does not grow without limit.
- Smoothing: reduces jitter but adds delay.

## Expected Output

```text
outputs/logs/realtime_rule_queue.csv
outputs/reports/realtime_rule_report.json
```

## Interview Explanation

I would describe the system as a low-latency streaming pipeline. Audio capture, feature extraction, model inference, and output control should be separate stages connected by bounded queues. The system should monitor FPS, mean latency, P95 latency, and dropped frames.
