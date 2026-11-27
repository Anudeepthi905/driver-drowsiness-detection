# Driver Drowsiness Detection

A real-time drowsiness detector built with OpenCV and Dlib. It watches your eyes through a webcam and sounds an alarm when they stay closed for about 2 seconds. Normal blinks are ignored.

## How it works

```
webcam frame → face detection → 68 facial landmarks → eye aspect ratio → closed-frame counter → alarm
```

For each eye, the **eye aspect ratio (EAR)** compares eye height with eye width, using six landmarks:

$$\text{EAR} = \frac{\lVert p_2 - p_6 \rVert + \lVert p_3 - p_5 \rVert}{2\,\lVert p_1 - p_4 \rVert}$$

The EAR is high when the eye is open and falls towards 0 as the lids close. A frame counts as **closed** when the average EAR of both eyes is below 0.25.

A blink lasts only 100–400 ms, which is 3–10 frames, so a single closed frame proves nothing. The alarm sounds only after **48 closed frames in a row** (about 2 s), and it stops on the first open-eye frame.

The EAR formula comes from [Soukupová and Čech (2016)](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf). The threshold-plus-counter rule is this project's own; the paper uses a trained classifier instead. See [docs/report_notes.md](docs/report_notes.md) for the full method, calibration and tests.

## Setup

You need Python 3.11 and a webcam.

The easiest route on Windows is conda, because pip has to compile dlib from source (CMake plus a C++ compiler):

```bash
conda create -n drowsy -c conda-forge --override-channels python=3.11 dlib=19.24.6
conda activate drowsy
pip install opencv-python==4.10.0.84 imutils==0.5.4 pygame==2.6.1 scipy==1.14.1 numpy==1.26.4
```

If you already have CMake and a compiler, `pip install -r requirements.txt` works instead.

Then download Dlib's landmark model (about 100 MB, kept out of Git):

```bash
python scripts/download_model.py
```

## Usage

```bash
python drowsiness_detector.py                    # run
python drowsiness_detector.py --record out.mp4   # run and save the overlaid video (no audio)
```

Press **`q`** to quit. The window shows the live EAR, the eye state, the closed-frame counter, and a red alert banner when drowsiness is detected.

To turn a recording into a GIF, run `python scripts/make_gif.py out.mp4` (needs [ffmpeg](https://ffmpeg.org/)).

## Configuration

These constants are at the top of `drowsiness_detector.py`:

| Constant | Default | Effect |
|---|---|---|
| `EAR_THRESHOLD` | `0.25` | EAR below this counts as closed. Calibrate for your face ([how](docs/report_notes.md#4-testing-and-calibration)) |
| `CONSEC_FRAMES_LIMIT` | `48` | Closed frames in a row before the alarm |
| `CAMERA_INDEX` | `0` | Which webcam to use |
| `ALARM_SOUND_FILE` | `sounds/alarm.wav` | Alarm sound: a siren, then a spoken "take a break and get some fresh air" warning, looped |

## Limitations

- **Poor lighting and glasses glare:** both make the eye landmarks unreliable.
- **Faces turned away:** faces turned more than about 30° are not detected, and the counter resets instead of raising the alarm.
- **Frame rate:** the limit is counted in frames, so on a slower computer the alarm takes longer.
- **Fixed threshold:** one threshold does not suit every face.
