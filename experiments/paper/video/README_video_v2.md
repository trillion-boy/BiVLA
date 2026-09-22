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
ffmpeg -i ICRA27_raw.mp4 -vf scale=1280:720 -r 30 -c:v libx264 -preset slow -b:v 720k -maxrate 800k -bufsize 1600k -c:a aac -b:a 64k -movflags +faststart -map_metadata -1 ICRA27_<paperID>.mp4
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

## v3 wall deck (final candidate): `ICRA27_video_v3_wall_subs.pptx`, 19 slides, 178 s

Two builds of the same deck. `ICRA27_video_v3_wall_subs.pptx` has the narration as a subtitle
band at the bottom of every slide (the mentor's choice when it fits, and it does: every element
ends above the band). `ICRA27_video_v3_wall.pptx` is the same deck without the band. Both are
post-processed by `autoplay.py`: unique shape ids, every clip starts automatically when its
slide appears, and every slide advances by itself after its target seconds with a plain cut
(no transition effect, so the exported video length equals the sum of the slide times).

Build: `WALL=1 SUBS=1 node build_v2.js` (or without `SUBS=1`), then
`python3 autoplay.py in.pptx out.pptx "$(cat ICRA27_video_v3_wall_subs.secs.json)"`,
then `python3 qa_deck.py out.pptx deck_render --subs` (renders, bounds and overlap check,
identifying-string scan, narration files).

| Slides | Content | s |
|---|---|---|
| 1 | Motivation (Fig. 2 pipeline row) | 13 |
| 2 to 6 | One slide per trick, method panel from Fig. 2 plus a paired CogACT WidowX clip where the original fails and the trick succeeds (foveation 20%, action repeat 4, depth pruning 2, reuse moderate, fusion motion-entropy) | 11, 11, 12, 11, 12 |
| 7 to 9 | CogACT, WidowX walls, 14 tiles: eggplant in basket (rollout 3, 3×), stack cube (rollout 5, 2×), spoon on towel (rollout 4, 2×) | 8, 6, 6 |
| 10 to 12 | OpenVLA, Fractal walls, 14 tiles: move near (rollout 4), pick coke can (rollout 4), open drawer (rollout 5), all 2× | 8, 8, 8 |
| 13 to 16 | UniVLA, LIBERO walls, 14 tiles: Spatial task 3 (rollout 1, 3×), Object task 9 (rollout 1, 4×), Goal task 8 (rollout 2, 3×), Long task 1 (rollout 2, 8×) | 6, 6, 6, 7 |
| 17 | Q1 success, Fig. 1 | 13 |
| 18 | Q2 speed, Fig. 4 and latency table | 13 |
| 19 | Takeaways | 13 |

Each wall shows every configuration on one episode with the same initial state (rollout_k is
episode_index k-1 in `results_corrected`). Episodes were chosen by agreement with the paper's
per-setting deltas: WidowX and Fractal against Table I (OpenVLA on Fractal: foveation fails,
the other tricks match or beat the original), LIBERO against Table II (only action repeat
fails). The success flag of every tile is read from the SUCCESS/FAILURE badge of the rollout
video and compared with `episodes.jsonl` in `deck_render/clip_manifest.txt`: 145 of 150 agree;
the 5 that differ are WidowX CogACT clips (that set of videos is a re-run of the original
experiment; the mentor decided to show the videos as they are, so every border, count and
narration follows the video badge). LIBERO runs longer than the
slide are cut at the slide end (the badge is visible from the first frame). Sources:
`simulation_rollouts/`, indexes `rollout_index.json` and `rollout_index_libero.json`, clip list
`clips/manifest.json`, selection `select_clips.py` and `select_libero.py`.

### Voice: 19 Speechma blocks

`speechma_script_v3_wall.txt` holds one block per slide (longest 234 characters, far below Speechma's 2,000). Keep one
English voice and the default speed. After downloading each MP3:

1. Insert > Audio > Audio on My PC on that slide; Playback: Start Automatically, Hide During
   Show, Play Across Slides off.
2. The slide's Advance After is already set (Transitions tab). If an MP3 is longer than the
   slide's target seconds, set that slide's Advance After to the MP3 length plus 0.3 s.
   Every block is written at 2.6 words per second or slower, so a default-speed voice should fit.
3. Add up the Advance After values over the 19 slides (Slide Sorter shows them under each
   slide). The sum must stay at or below 179 s. If it goes over, regenerate the densest
   blocks (slides 1, 4, 6, 19, then 3 and 5) at Speechma speed +10% rather than cutting content.

### Export, compress, check

File > Export > Create a Video > Full HD (1080p), "Use Recorded Timings and Narrations",
save as `ICRA27_raw.mp4`. Then:

```
ffmpeg -i ICRA27_raw.mp4 -vf scale=1280:720 -r 30 -c:v libx264 -preset slow -b:v 720k -maxrate 800k -bufsize 1600k -c:a aac -b:a 64k -movflags +faststart -map_metadata -1 ICRA27_<paperID>.mp4
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -show_entries format=duration,size -of default=nw=1 ICRA27_<paperID>.mp4
```

| Rule | Target | Check |
|---|---|---|
| Length | at most 180 s | `duration` at or below 179 |
| Size | at most 20 MB | `size` below 20000000 bytes (178 s at 720 + 64 kb/s is about 17.5 MB, a 12% margin; use 650k if over) |
| Resolution and rate | 16:9, height at least 480, at least 20 fps | 1280x720, 30/1 |
| Format | mp4 | H.264 video, AAC audio |
| Anonymity | no names, affiliations, logos, URLs | `qa_deck.py` scans the slide text; watch the exported file once end to end |
| Upload | PaperPlaza, second window, by 22 Sept 2026 23:59 Pacific | 23 Sept 15:59 KST at the latest, upload on the 22nd |
