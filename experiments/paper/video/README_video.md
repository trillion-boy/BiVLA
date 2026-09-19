# ICRA 2027 accompanying video: storyboard draft and production notes

Deck: `ICRA27_video_storyboard.pptx` (16 slides, 16:9). Each slide's speaker notes hold the
English narration, its word count, the target duration, and a production note.
Total planned length: 165 s (limit 180 s).

## Rules that apply (ICRA 2027 call for papers, checked 2026-09-18)

| Item | Requirement |
|---|---|
| Length | at most 180 s |
| File size | at most 20 MB |
| Format | mp4 (mpeg / mpg also accepted), 16:9, height at least 480 px |
| Upload window | second window 17 to 22 September 2026 (a video not uploaded in the window is not accepted later) |
| Anonymity | double-anonymous: no author names, affiliations, logos, GitHub handles, or project-page URL in the video |

## Slide list and timing

| # | s | Slide | Visual | Needs recording |
|---|---|---|---|---|
| 1 | 7 | Title | dimmed Fig. 2 | no |
| 2 | 10 | No trick is plug-and-play | three big deltas (depth pruning, WidowX) | no |
| 3 | 10 | VLA models are strong but slow | one original rollout + latency counter | yes (1 clip) |
| 4 | 12 | Five tricks, three axes | Fig. 2 with axis chips | no |
| 5 | 8 | What to see: foveation | Fig. 2 crop + raw / keep 20% / keep 50% frames | no |
| 6 | 10 | When to act: repeat vs guarded reuse | policy-call timeline dots | no |
| 7 | 10 | How much to compute | Fig. 2 crops (pruning, fusion) | no |
| 8 | 12 | Matched evaluation protocol | backbone × environment grid, stats | no |
| 9 | 12 | RQ-1 success | Fig. 1 + delta chips | no |
| 10 | 12 | Depth pruning helps CogACT | Fig. 3(a) strips + paired clip | yes (1 clip) |
| 11 | 10 | Same trick, different backbone | MiniVLA original vs pruned clips | yes (2 clips) |
| 12 | 12 | RQ-2 speed | Fig. 4 + latency table | no |
| 13 | 14 | Open loop vs gating | CogACT original / repeat k=4 / guarded reuse | yes (3 clips) |
| 14 | 12 | RQ-3 consistency | consistency chart + chips | no |
| 15 | 10 | Takeaways | four verdict cards | no |
| 16 | 4 | End card | dimmed Fig. 1 | no |

Clips to record (7 in total, 6 to 10 s each, 720p, same task instance and initial state
inside each pair):

1. Slide 3: any original policy on WidowX with a burned-in "ms per step" counter.
2. Slide 10: CogACT WidowX, original (fails) vs depth pruning (succeeds).
3. Slide 11: MiniVLA WidowX, original (succeeds) vs depth pruning (fails), same task family.
4. Slide 13: one CogACT WidowX episode as original / action repeat k=4 / guarded reuse,
   with a "policy calls used" counter under each panel.

Label every clip "Selected episode, same initial state" (already on the slides) so no clip
reads as representative of the aggregate numbers.

## How to turn the deck into the mp4

1. Fill the clip placeholders: Insert > Video > This Device, then set playback to start
   automatically, or just leave a still frame if recording is not possible.
2. Narration: either record in PowerPoint (Slide Show > Record) or generate a TTS track from
   the speaker notes and insert per slide (Insert > Audio). Speaking pace is at most
   2.5 words per second, already checked in the notes.
3. Set slide timings to the values in the notes (Transitions > After: N s), or let the
   recorded narration set them.
4. File > Export > Create a Video > Full HD (1080p) or HD (720p), "Use recorded timings and
   narrations". Save as `ICRA27_<paperID>_raw.mp4`.
5. Compress to the 20 MB cap. For 165 s, a total bitrate around 900 kb/s is safe:

```
ffmpeg -i ICRA27_raw.mp4 -vf scale=1280:720 -c:v libx264 -preset slow -b:v 800k -maxrate 900k -bufsize 1800k -c:a aac -b:a 64k -movflags +faststart ICRA27_<paperID>.mp4
```

   Check: `ls -l` under 20,000,000 bytes and `ffprobe` duration under 180 s.
6. Anonymity pass: scrub the mp4 metadata (`ffmpeg -i in.mp4 -map_metadata -1 -c copy out.mp4`),
   watch the whole video once for any name, logo, path, or URL in the clips.
7. Upload on PaperPlaza as the video attachment for the paper before 22 September 2026 23:59 PST.

## Numbers used on the slides (all from Tables I and II, best setting per trick)

- Depth pruning, WidowX success: CogACT 50.0 → 59.5 (+9.5), MiniVLA 36.0 → 18.5 (−17.5),
  SpatialVLA 45.0 → 38.5 (−6.5).
- Foveation, WidowX: CogACT +2.5, SpatialVLA +5.0, CronusVLA +2.5. LIBERO Goal: UniVLA +7.0,
  SmolVLA −13.0.
- Action repeat, WidowX: CogACT −38.0, UniVLA −75.0. Fractal: CogACT −1.2, SpatialVLA 0.0.
- Latency ms/step: SpatialVLA WidowX 423.5 → 235.1 (repeat), OpenVLA LIBERO Long
  171.6 → 93.2 (repeat), OpenVLA WidowX 210.2 → 199.4 (pruning), CogACT WidowX
  141.4 → 137.5 (pruning), SmolVLA LIBERO Long 289.5 → 268.9 with success 42.0 unchanged
  (guarded reuse).
- Scale: 7 backbones, 3 environments, 22 backbone × environment pairs, 13 trick settings +
  original per pair, 286 trick settings.
