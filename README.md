# Driver Drowsiness Detection

A real-time system built with OpenCV and Dlib. It tracks the driver's eyes from live webcam video and sounds an alarm when the eyes stay closed for more than a set number of frames. The alarm stops as soon as the eyes reopen.

## How it works

1. **Capture:** a webcam frame is read and resized so detection is fast.
2. **Detect:** Dlib finds the face and places 68 facial landmarks on it.
3. **Measure:** the Eye Aspect Ratio (EAR) is computed for each eye from its 6 landmarks.
4. **Classify:** if the average EAR is below `EAR_THRESHOLD`, the eyes are treated as closed.
5. **Count:** a counter tracks how many frames in a row the eyes have been closed.
6. **Alert:** once the counter reaches `CONSEC_FRAMES_LIMIT`, the alarm loops until the eyes open.

The math behind EAR, the blink-versus-drowsiness logic, and testing guidance are in [docs/report_notes.md](docs/report_notes.md).

## Project structure

```
Driver Drowsiness Detection/
├── drowsiness_detector.py   # main script (all detection and alarm logic)
├── requirements.txt         # exact Python dependencies
├── models/                  # put shape_predictor_68_face_landmarks.dat here (not committed)
├── sounds/alarm.wav         # alarm sound
├── docs/report_notes.md     # explanation for the report and testing guidance
└── README.md
```

## Setup

You need Python 3.11 and a webcam.

### 1. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

**About dlib:** pip compiles dlib from source, which needs [CMake](https://cmake.org/download/) and a C++ compiler (on Windows, "Visual Studio Build Tools" with the "Desktop development with C++" workload). If the install fails, the easiest route on Windows is Anaconda:

```bash
conda create -n drowsy python=3.11
conda activate drowsy
conda install -c conda-forge dlib=19.24
pip install opencv-python==4.10.0.84 imutils==0.5.4 pygame==2.6.1 scipy==1.14.1 numpy==1.26.4
```

### 2. Download the landmark model

1. Download `shape_predictor_68_face_landmarks.dat.bz2` from <http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2> (about 64 MB).
2. Extract it with 7-Zip on Windows or `bunzip2` on macOS/Linux. You get `shape_predictor_68_face_landmarks.dat` (about 100 MB).
3. Place it here:

   ```
   models/shape_predictor_68_face_landmarks.dat
   ```

The script builds this path relative to its own location (`MODEL_FILE` at the top of `drowsiness_detector.py`), so it works whichever folder you run it from. If the file is missing, the program stops with a message telling you where to put it.

**Why the model is not committed to Git:** the file is about 100 MB, which is right at GitHub's 100 MB hard limit for a single file and would bloat every clone. It is also freely downloadable from a stable URL. Its license is restrictive too: it was trained on the iBUG 300-W dataset, which does not allow commercial use. For these reasons `.gitignore` excludes `models/*.dat`, and each user downloads the model once.

## Usage

```bash
python drowsiness_detector.py
```

A window shows the live video with:
- the current **EAR** value,
- the **eye state** (OPEN in green, CLOSED in red),
- the **closed-frame counter** (for example `12/48`),
- **DROWSINESS ALERT!** when the limit is reached.

Press **`q`** in the video window to quit. The camera is released, the alarm is stopped, and windows are closed.

## Configuration

Edit the constants at the top of `drowsiness_detector.py`:

| Constant | Default | Effect |
|---|---|---|
| `EAR_THRESHOLD` | `0.25` | EAR below this counts as closed |
| `CONSEC_FRAMES_LIMIT` | `48` | Closed frames in a row before the alarm (about 2 s at 24 fps) |
| `CAMERA_INDEX` | `0` | Which webcam to open |
| `ALARM_SOUND_FILE` | `sounds/alarm.wav` | Sound looped during the alarm |

See [docs/report_notes.md](docs/report_notes.md#4-testing-and-calibration) for how to calibrate the threshold for your own face.

## Error handling

| Situation | Behaviour |
|---|---|
| Model file missing | Exits with download and placement instructions |
| Webcam missing or busy | Exits with troubleshooting tips |
| Sound file missing / no audio device | Prints a warning and continues with the on-screen alert only |
| Webcam disconnected mid-run | Stops cleanly |
| No face in frame | Shows "No face detected" and resets the counter |

## Reference

Soukupová, T. and Čech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks.* 21st Computer Vision Winter Workshop.
