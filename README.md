# sag-tracker

Measure your mountain bike fork's setup from a video. Film yourself compressing and
releasing the suspension, and get back the numbers you would otherwise estimate with
an O-ring and a guess: **sag** in mm and as a percentage of travel, and how the fork
**rebounds** — peak extension speed, time back to sag, overshoot, and an effective
damping ratio.

## How it works

Stick a piece of bright tape on the stanchion and another on the lower leg. The tool
tracks both and measures the **distance between them**, frame by frame:

```
      ╔═════════╗   crown
      ║         ║
  ▓▓▓▓█         █▓▓▓▓  ← pink tape (stanchion)
      █         █
      █         █    this distance shrinks
      █         █    as the fork compresses
  ────┴─────────┴────  dust seal
  ▓▓▓▓█         █▓▓▓▓  ← green tape (lower leg)
      █         █
      ▀▀▀▀▀▀▀▀▀▀▀
```

Measuring a distance *between two markers on the fork* rather than either marker's
position in the frame is what makes this work handheld: if the camera shifts or rolls,
both markers move together and the distance is unchanged. No stabilisation, no
tripod-only rule, no reference markers in the background.

From that one signal the tool finds the moments you were holding still (the plateaus),
works out which is the unweighted fork and which is sag, spots each compress-and-release
cycle, and fits the free vibration that follows.

## Install

```sh
uv sync
```

## Filming protocol

Accuracy comes mostly from the filming, so it is worth two minutes of care.

**Setup**

- Bright tape, wrapped all the way round so it cannot rotate out of view. Pink on the
  stanchion, green on the lower leg (any of `pink`, `red`, `orange`, `yellow`, `green`,
  `blue` works — just tell the tool which).
- Camera roughly square to the fork, at about fork mid-height, ~2 m back and zoomed in
  so the fork fills the frame. Both bits of tape must stay visible the whole time —
  film the side away from the rider's leg.
- **Shoot at 120 or 240 fps.** A rebound stroke lasts 150–400 ms, so 30 fps gives only
  5–12 samples across the thing you are measuring. Sag is fine at any frame rate; the
  rebound numbers are not. The tool warns you below 60 fps.

**The sequence**

| | Duration | What to do | What it gives |
|---|---|---|---|
| 1 | ~2 s | Bike upright, nobody on it, fork fully extended | the topped-out reference |
| 2 | ~3 s | Rider on, neutral stance, hold still | **sag** |
| 3 | ×3 | Push down hard, let go cleanly, hold still until it settles | **rebound** |
| 4 | ~2 s | Push the fork all the way to bottom-out and hold | calibration |

Step 4 is what lets the tool convert pixels to millimetres from your known travel, and
it is the more reliable of the two calibration routes — see below.

## Use

```sh
uv run sag-tracker analyze fork.mp4 --travel 160
```

Writes `results/metrics.json` and `results/trace.csv`.

Common options:

```sh
--upper-color pink --lower-color green   # whichever tape you used
--travel 160                             # your fork's travel in mm
--fps 240                                # if the file's own frame rate is wrong
--start 2 --end 14                       # analyse only part of the clip
--out-dir results/
```

### Calibration

Pixels become millimetres one of two ways. **Prefer `--travel`**: because the reference
stroke and the measured stroke are foreshortened identically, a camera that is not
perfectly square to the fork cancels out of the ratio. It needs a bottom-out in the clip.

The other route measures the stanchion's known diameter across the image. It needs no
bottom-out but *is* sensitive to camera angle, since a stanchion viewed off-axis is
foreshortened differently from the fork's stroke:

```sh
uv run sag-tracker calibrate fork.mp4 --stanchion 35   # click the two edges
# prints: --stanchion 35 --stanchion-pt-a 154,318 --stanchion-pt-b 224,318
uv run sag-tracker analyze fork.mp4 --stanchion 35 --stanchion-pt-a 154,318 --stanchion-pt-b 224,318
```

Pass both and the tool reports each scale and warns if they disagree by more than 5 %,
which is a good sign the camera was off-square or the fork never fully bottomed.

## Output

`metrics.json`:

- **`sag`** — `sag_mm`, `sag_pct`, `travel_mm`, `max_compression_mm`
- **`rebound`** — medians across the clean releases, plus every event individually:
  `peak_rebound_velocity_mm_s`, `time_to_sag_s`, `t90_s`, `overshoot_mm`,
  `settling_time_s`, `oscillation_count`, `damping_ratio`, `natural_freq_hz`
- **`segments`** — every still stretch found, labelled `topped-out` / `sag` /
  `compressed`, so you can check the tool split the clip the way you intended
- **`quality`** — what fraction of frames each marker was found in, how much was
  interpolated, the calibration used
- **`warnings`** — everything the tool wants you to know: low frame rate, lost markers,
  disagreeing calibrations, too few clean releases

`trace.csv` is the per-frame signal: `frame, t_s, separation_px, separation_mm,
compression_mm, velocity_mm_s, upper_ok, lower_ok, valid`. Plot `compression_mm`
against `t_s` if you want to see the strokes.

The deliberate bottom-out is much deeper than the pushes, so it is detected, labelled
`"role": "bottom-out"`, and kept out of the rebound medians — letting the fork back up
under control is not the same experiment as releasing it.

### Reading the numbers

`damping_ratio` is fitted from the log decrement of successive turning points after you
let go. A real fork is nonlinear — progressive spring, position-sensitive damping,
stiction — so treat it as an *effective* value for that stroke depth, useful for
comparing clicks on the rebound knob rather than as a physical constant. When the fork
comes back without overshooting at all, the tool reports `null` and says why instead of
inventing a number.

## Checking it without a real video

`synth` renders a fake fork driven by a displacement programme whose sag, travel,
damping ratio and natural frequency are known exactly, and writes the ground truth
alongside the clip:

```sh
uv run sag-tracker synth /tmp/fork.mp4 --fps 120 --zeta 0.35
uv run sag-tracker analyze /tmp/fork.mp4 --travel 160 --out-dir /tmp/out
```

Against that fixture the pipeline currently recovers sag to 0.2 mm, the pixel scale to
0.04 %, peak rebound velocity to 0.2 % and the damping ratio to 0.007. `--shake` adds
handheld camera wobble, `--occlude T0,T1` hides a marker, `--noise` adds sensor grain.

```sh
uv run pytest
```

## Limitations

- Frame rate is the dominant limit on rebound velocity. Sag is unaffected.
- Both markers must stay visible. Short dropouts are bridged and flagged; long ones are
  interpolated but marked invalid so they cannot quietly pass as data.
- Segmentation needs you to actually hold still. If no settled rider position is found,
  the tool says so and reports the trace rather than guessing at sag.
- Front fork only for now. The measurement model is just "distance between two markers
  on a telescoping pair", so a rear shock is a configuration change rather than a
  rewrite.
