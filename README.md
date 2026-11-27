# Driver Drowsiness Detection

A real-time system built with OpenCV and Dlib. It tracks the driver's eyes from live webcam video and sounds an alarm when the eyes stay closed for more than a set number of frames. The alarm stops as soon as the eyes reopen.

![Demo: eyes open, eyes closed past the threshold, alert banner, eyes open again](docs/demo.gif)

*The overlay shows the face box, the 12 eye landmarks, the live Eye Aspect Ratio (EAR), the closed-frame counter, and the alert banner. The demo GIF has no sound; the real program also sounds an alarm.*

## How it works

```
webcam frame → face detection → 68 facial landmarks → eye aspect ratio → closed-frame counter → alarm
```

1. **Face detection.** Dlib's HOG + linear SVM detector finds the face in each frame. It runs on a reduced 400 px copy of the frame because detection is the slowest step.
2. **68 landmarks.** Dlib's shape predictor places 68 points on the face. Six of them outline each eye.
3. **Eye aspect ratio (EAR).** For each eye, with corners p1 and p4, upper-lid points p2 and p3, and lower-lid points p6 and p5:

   $$\text{EAR} = \frac{\lVert p_2 - p_6 \rVert + \lVert p_3 - p_5 \rVert}{2\,\lVert p_1 - p_4 \rVert}$$

   This is eye height divided by eye width. The original paper measured about 0.4 ± 0.1 for open eyes and 0.05 ± 0.05 for closed ones, with other landmark detectors. It drops when the eyelids close because the height shrinks while the width stays the same. Since it is a ratio, it doesn't change when you move closer to or further from the camera. The two eyes are averaged.
4. **Closed-frame counter.** A frame counts as "closed" when the EAR is below `EAR_THRESHOLD` (0.25). The counter goes up by 1 for every closed frame in a row, and any open frame resets it to 0.
5. **Alarm.** When the counter reaches `CONSEC_FRAMES_LIMIT` (48 frames, about 2 seconds), a looping alarm sounds and a red banner appears. Both stop on the first open-eye frame.

**Why a frame threshold instead of reacting to a single closed frame?** A blink lasts about 100–400 ms (Soukupová and Čech, 2016), which is only 3–10 frames on a webcam. Reacting to every low-EAR frame would sound the alarm on every blink. With a 48-frame threshold, a blink resets the counter long before it reaches the limit. Only a sustained closure, the sign of drowsiness, gets there. The same rule ignores one-off landmark glitches from a single bad frame.

**Relation to the paper.** The EAR formula comes from Soukupová and Čech (2016). The decision rule does not. The paper detects blinks with a linear SVM trained on a 13-frame window of EAR values, and uses a plain 0.2 threshold only as a baseline to compare against. This project uses the simpler threshold-plus-counter rule because it needs no labelled training data, and its goal is sustained closure rather than individual blinks.

Full derivation, testing and calibration notes: [docs/report_notes.md](docs/report_notes.md).

## Requirements

- Python 3.11
- A webcam
- Python packages, pinned in [requirements.txt](requirements.txt): `opencv-python 4.10.0.84`, `dlib 19.24.6`, `imutils 0.5.4`, `pygame 2.6.1`, `scipy 1.14.1`, `numpy 1.26.4`
- Optional, only for making the demo GIF: [ffmpeg](https://ffmpeg.org/)

## Setup

### 1. Install the dependencies

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

**About dlib:** pip compiles dlib from source, which needs [CMake](https://cmake.org/download/) and a C++ compiler. On Windows that compiler is "Visual Studio Build Tools" with the "Desktop development with C++" workload. If that's a hassle, Anaconda provides a precompiled dlib:

```bash
conda create -n drowsy -c conda-forge --override-channels python=3.11 dlib=19.24.6
conda activate drowsy
pip install opencv-python==4.10.0.84 imutils==0.5.4 pygame==2.6.1 scipy==1.14.1 numpy==1.26.4
```

### 2. Download the Dlib landmark model

```bash
python scripts/download_model.py
```

This downloads `shape_predictor_68_face_landmarks.dat.bz2` (about 64 MB) from [dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), decompresses it, and saves it as `models/shape_predictor_68_face_landmarks.dat` (about 100 MB). Running it again does nothing if the file is already there. To do it by hand instead, download the link above, extract it with 7-Zip or `bunzip2`, and put the `.dat` file in `models/`.

**The model is not committed to Git.** It is right at GitHub's 100 MB per-file limit and freely downloadable. It was also trained on the iBUG 300-W dataset, which does not allow commercial use. `.gitignore` excludes it.

## Usage

```bash
python drowsiness_detector.py                    # run live
python drowsiness_detector.py --record demo.mp4  # run live and also save the overlaid video (no audio)
```

Press **`q`** in the video window to quit. The camera is released, the alarm stops, any recording is saved, and windows close.

### Configuration

The constants at the top of `drowsiness_detector.py`:

| Constant | Default | Effect |
|---|---|---|
| `EAR_THRESHOLD` | `0.25` | EAR below this counts as closed |
| `CONSEC_FRAMES_LIMIT` | `48` | Closed frames in a row before the alarm (about 2 s at 24 fps) |
| `CAMERA_INDEX` | `0` | Which webcam to open |
| `ALARM_SOUND_FILE` | `sounds/alarm.wav` | Sound looped during the alarm |

To calibrate `EAR_THRESHOLD` for your own face and lighting, see [docs/report_notes.md](docs/report_notes.md#4-testing-and-calibration).

### Error handling

| Situation | Behaviour |
|---|---|
| Model file missing | Exits and tells you to run `scripts/download_model.py` |
| Webcam missing or busy | Exits with troubleshooting tips |
| Sound file missing / no audio device | Warns and continues with the on-screen alert only |
| Webcam disconnected mid-run | Stops cleanly and saves any recording |
| No face in frame | Shows "No face detected" and resets the counter |

## Recording the demo

1. **Record** about 12–15 seconds of yourself:

   ```bash
   python drowsiness_detector.py --record demo.mp4
   ```

   The first 30 frames (about 1–2 seconds, depending on your computer) are used to measure the real frame rate, so the video plays back at true speed.
2. **Convert** the best 8–10 seconds to a GIF:

   ```bash
   python scripts/make_gif.py demo.mp4 --start 1 --duration 9
   ```

   This writes `docs/demo.gif`: at most 10 s long, about 12 fps, 640 px wide and 8 MB or less. If the GIF comes out too big, the script retries with fewer colours and then a smaller width.

`.mp4` files are git-ignored, so the raw recording never gets committed by accident. Only `docs/demo.gif` is.

## Limitations

- **Lighting.** Dim or uneven light makes landmarks jitter. The EAR then gets noisy and can cross the threshold while the eyes are open. Strong backlight (a window behind you) can stop the face from being found at all.
- **Glasses glare.** Reflections on lenses hide the eyelids, so the eye landmarks may be placed badly. Thick frames sitting close to the eyes can have the same effect.
- **Head turned away.** The HOG detector is trained on near-frontal faces. If the head turns more than about 30° or drops out of view, no face is found, and the counter resets instead of raising an alarm. A driver nodding off with their head slumped may therefore be missed.
- **Frame-rate dependence.** The limit is counted in frames, so on a slower computer 48 frames takes longer in real time.
- **One fixed threshold.** EAR differs between people. 0.25 is a starting point; calibrate it for your own face.

## Ideas for improvement

- **Per-user calibration:** measure the user's open-eye EAR for a few seconds at start-up and set the threshold relative to it.
- **Time-based limit:** trigger after a number of seconds rather than frames, so behaviour is the same on any computer.
- **Smoothing:** apply a moving average to the EAR to reduce landmark jitter.
- **Head pose and yawning:** estimate head pitch from the landmarks to catch nodding, and add a mouth aspect ratio to detect yawns.
- **Robustness in the dark:** use an infrared camera for night driving. A CNN face detector (Dlib's MMOD, or MediaPipe Face Mesh) handles turned heads and poor light better, at extra compute cost.
- **Learned classifier:** replace the fixed threshold with a small model trained on a sequence of EAR values. Blink-pattern features such as PERCLOS (the share of time the eyes are closed) are standard in drowsiness research.

## Project structure

```
├── drowsiness_detector.py     # main script: detection, EAR, counter, alarm, recording
├── requirements.txt           # pinned dependencies
├── scripts/
│   ├── download_model.py      # fetches the Dlib landmark model into models/
│   └── make_gif.py            # turns a recording into docs/demo.gif (needs ffmpeg)
├── models/                    # landmark model goes here (git-ignored)
├── sounds/alarm.wav           # alarm sound
└── docs/
    ├── demo.gif               # README demo
    └── report_notes.md        # method, maths, testing and calibration notes
```

## Reference

Soukupová, T. and Čech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks.* 21st Computer Vision Winter Workshop.
