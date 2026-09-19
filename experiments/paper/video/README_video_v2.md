# ICRA 2027 accompanying video, v2 (mentor's structure)

Deck: `ICRA27_video_v2.pptx`, 14 slides, 16:9, about 155 s planned (limit 180 s, target under 3 min).
Structure as agreed with the mentor: no title slide, one motivation slide, five trick slides
(one per trick), then results. Each slide's speaker notes hold the narration, the word count,
the target duration, and a production note.

## Rules (ICRA 2027)

| Item | Requirement |
|---|---|
| Length | at most 180 s |
| File size | at most 20 MB |
| Format | mp4 (mpeg / mpg accepted), 16:9, height at least 480 px |
| Upload window | second window 17 to 22 September 2026 on PaperPlaza; not accepted later |
| Anonymity | no author names, affiliations, logos, GitHub handles, or project-page URL |

## Slides

| # | s | Slide | Clip needed |
|---|---|---|---|
| 1 | 15 | Motivation: VLA models are strong but slow | optional background rollout |
| 2 | 10 | Trick 1 Visual foveation | original vs foveation |
| 3 | 10 | Trick 2 Action repeat | original vs action repeat k=4 |
| 4 | 10 | Trick 3 Depth pruning | original vs depth pruning (CogACT WidowX) |
| 5 | 10 | Trick 4 Guarded action reuse | original vs guarded reuse |
| 6 | 10 | Trick 5 Temporal fusion | original vs temporal fusion |
| 7 | 10 | Results: matched evaluation protocol | none |
| 8 | 12 | Which tricks preserve success? (Fig. 1) | none |
| 9 | 12 | Depth pruning helps CogACT on WidowX (Fig. 3) | one paired episode |
| 10 | 10 | Same trick, same benchmark, different backbone | MiniVLA original vs pruned |
| 11 | 12 | Which tricks are actually faster? (Fig. 4) | none |
| 12 | 12 | Open loop has a cost, gating keeps it safe | one CogACT episode, three variants |
| 13 | 12 | Does the effect transfer across models? | none |
| 14 | 10 | Takeaways | none |

Every clip pair must use the same task instance and initial state, and carry the label
"selected episode, same initial state" (already on the slides). Slides 2 to 6 can share
one episode per trick with the results slides if recording time is short. If a clip is not
available in time, keep the still frame from Fig. 3 in its place; the video still works.

## Voice with Speechma (mentor's choice)

1. Open https://speechma.com/english. Pick one English voice and keep it for every block
   (the free version takes up to 2,000 characters per run; the longest block here is 243).
2. Paste one block from `speechma_script_v2.txt` at a time, generate, download the MP3.
   Keep speed at the default. If a block runs longer than its target seconds, do not speed
   up the voice; extend that slide's duration instead.
3. In PowerPoint, on each slide: Insert > Audio > Audio on My PC, pick the MP3, then in
   Playback set Start: Automatically, tick Hide During Show, and Play Across Slides off.
4. Transitions > Advance Slide > After: set to the MP3 length plus 1 s (Timings are then
   fixed; no manual recording needed).
5. Insert the rollout clips on slides 2 to 6, 9, 10, 12 (Insert > Video > This Device,
   Start: Automatically, and trim to at most the slide duration).
6. File > Export > Create a Video > HD (720p) or Full HD, "Use Recorded Timings and
   Narrations". Save as `ICRA27_raw.mp4`.

## Compress and check

```
ffmpeg -i ICRA27_raw.mp4 -vf scale=1280:720 -c:v libx264 -preset slow -b:v 800k -maxrate 900k -bufsize 1800k -c:a aac -b:a 64k -movflags +faststart -map_metadata -1 ICRA27_<paperID>.mp4
ffprobe -v error -show_entries format=duration,size -of default=nw=1 ICRA27_<paperID>.mp4
```

Duration must be under 180 s and size under 20,000,000 bytes. At 800 kb/s, 155 s is
about 16 MB. Watch the whole file once for any name, logo, path, or URL inside the clips
before uploading on PaperPlaza (deadline 22 September 2026, 23:59 PST).

## Numbers on the slides (Tables I and II, best setting per trick)

- Depth pruning, WidowX success: CogACT 50.0 → 59.5 (+9.5), MiniVLA 36.0 → 18.5 (−17.5),
  SpatialVLA 45.0 → 38.5 (−6.5).
- Foveation: WidowX CogACT +2.5, SpatialVLA +5.0, CronusVLA +2.5; LIBERO Goal UniVLA +7.0,
  SmolVLA −13.0.
- Action repeat: WidowX CogACT −38.0, UniVLA −75.0; Fractal CogACT −1.2, SpatialVLA 0.0.
- Latency ms/step: SpatialVLA WidowX 423.5 → 235.1 (repeat), OpenVLA LIBERO Long 171.6 → 93.2
  (repeat), OpenVLA WidowX 210.2 → 199.4 (pruning), CogACT WidowX 141.4 → 137.5 (pruning),
  SmolVLA LIBERO Long 289.5 → 268.9 with success 42.0 unchanged (guarded reuse).
- Scale: 7 backbones, 3 environments, 22 pairs, 13 trick settings + original per pair,
  286 trick settings.
