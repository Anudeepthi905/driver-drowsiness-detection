# Report Notes: Method, Logic and Testing

These notes explain how the system works, at a level you can adapt for an academic report. Each section points to the function in `drowsiness_detector.py` that implements it.

## 1. Face and landmark detection

*Functions: `capture_frame`, `detect_landmarks`*

Each webcam frame is resized to 640 px wide and converted to grayscale, because the detector only needs brightness patterns, not colour. Face detection is the slowest step and its cost grows with the number of pixels, so it runs on a smaller 400 px copy of the frame. The face box it finds is scaled back up, and the fast landmark predictor runs on the full 640 px image so the eye points stay accurate.

Two pre-trained Dlib models are then applied:

- **Face detector:** a HOG (Histogram of Oriented Gradients) feature extractor with a linear SVM classifier. It returns a bounding box around each face and runs fast on an ordinary CPU.
- **Shape predictor:** an ensemble of regression trees (Kazemi and Sullivan, 2014) trained on the iBUG 300-W dataset. Given the face box, it places 68 landmarks on the eyebrows, eyes, nose, lips and jawline in a few milliseconds.

Only 12 of the 68 points are used: points 36–41 for one eye and 42–47 for the other.

## 2. The Eye Aspect Ratio (EAR)

*Functions: `eye_aspect_ratio`, `compute_ear`*

Each eye is described by six landmarks, p1 to p6:

```
          p2    p3
    p1                p4
          p6    p5
```

p1 and p4 are the eye corners. p2 and p3 sit on the upper eyelid, and p6 and p5 on the lower eyelid. The EAR is:

$$
\text{EAR} = \frac{\lVert p_2 - p_6 \rVert + \lVert p_3 - p_5 \rVert}{2\,\lVert p_1 - p_4 \rVert}
$$

where ‖·‖ is the Euclidean (straight-line) distance between two points.

**What the formula measures:**
- The **numerator** adds the two **vertical** distances between the upper and lower eyelids, which is how open the eye is.
- The **denominator** is the **horizontal** width of the eye. It is multiplied by 2 because the numerator contains two vertical distances, so the ratio compares one average height with one width.

**Why it drops when the eye closes:** as the eyelid comes down, p2 moves towards p6 and p3 towards p5, so the numerator shrinks towards zero. The eye corners do not move, so the denominator stays roughly constant. EAR therefore falls from a typical open value of about 0.25–0.35 to a closed value of about 0.05–0.15.

**Why a ratio instead of raw distances:** if the driver leans towards the camera, every distance in pixels gets larger, but the ratio stays the same. Dividing by the eye width makes EAR independent of the face's distance from the camera and of image resolution. It is also fairly robust to small head rotations.

The EAR of the two eyes is averaged, because both eyes close together and averaging smooths out small landmark errors.

## 3. Separating blinks from drowsiness: the frame counter

*Functions: `classify_eye_state`, `update_closed_counter`, `is_drowsy`, `update_alarm`*

A single frame with a low EAR does not mean the driver is drowsy. People blink 15–20 times a minute, and each blink lasts only about 100–400 ms. At a typical webcam rate of 24–30 fps, a blink covers only about **3–10 frames**.

The system therefore uses the **duration** of the closure, not just its presence:

1. **Classify** every frame: `CLOSED` if EAR < `EAR_THRESHOLD`, otherwise `OPEN`.
2. **Count** consecutive closed frames. Each `CLOSED` frame adds 1 to the counter, and any `OPEN` frame resets it to 0.
3. **Alarm** when the counter reaches `CONSEC_FRAMES_LIMIT` (default 48, about 2 seconds).
4. **Reset:** the first `OPEN` frame sets the counter back to 0, `is_drowsy` becomes false, and the alarm stops.

A normal blink pushes the counter to at most about 10 before an open frame resets it, so it never comes close to 48 and never triggers the alarm. A drowsy driver whose eyes stay shut keeps the counter climbing until it crosses the limit. In effect, this is a simple **temporal filter**: the alarm needs evidence that persists over time, which suppresses the short-lived false positives caused by blinks and by occasional single-frame landmark noise.

**The trade-off:** a larger limit gives fewer false alarms but a slower response, and a smaller limit gives the opposite. Because the limit is counted in frames, the time it represents depends on the frame rate:

$$
\text{time to alarm (s)} = \frac{\text{CONSEC\_FRAMES\_LIMIT}}{\text{frames per second}}
$$

`update_alarm` only starts the sound when the alarm *becomes* active and only stops it when it *becomes* inactive. This means the sound is not restarted on every frame.

## 4. Testing and calibration

### 4.1 Calibrating the EAR threshold

EAR depends on eye shape, glasses, camera angle and lighting, so the default 0.25 may not suit everyone.

1. Run `python drowsiness_detector.py` in the lighting and seating position you will test in.
2. **Eyes open:** look naturally at the screen for about 10 seconds and note the typical EAR shown, for example 0.30.
3. **Eyes closed:** close your eyes and have a friend read the value, or take a screenshot of the window. Note the typical EAR, for example 0.12.
4. Set the threshold between the two values, a little closer to the closed value so that squinting is not counted as closed:

   $$
   \text{EAR\_THRESHOLD} \approx \text{closed} + 0.4 \times (\text{open} - \text{closed})
   $$

   With the example values: 0.12 + 0.4 × (0.30 − 0.12) ≈ **0.19**.

5. Repeat in a second lighting condition, such as a dim room. If the values shift, choose a threshold that separates open and closed eyes in both.

Record the open and closed values in a small table for your report (subject, glasses yes/no, lighting, open EAR, closed EAR, chosen threshold).

### 4.2 Verifying the alarm triggers and resets

| Test | Action | Expected result |
|---|---|---|
| Blink rejection | Blink normally several times | Counter rises briefly (under ~10) and returns to 0; no alarm |
| Trigger | Close your eyes and keep them closed | Counter climbs to the limit, then the red "ALERT: DROWSINESS DETECTED" banner appears and the alarm sounds |
| Timing | Repeat the trigger test with a stopwatch | Delay ≈ `CONSEC_FRAMES_LIMIT` ÷ fps (about 2 s at 24 fps) |
| Reset | Open your eyes while the alarm is sounding | Alarm stops immediately; counter returns to 0; alert disappears |
| Re-trigger | Close your eyes again | Alarm sounds again after the full limit, not instantly |
| Limit change | Set `CONSEC_FRAMES_LIMIT = 15` and repeat | Alarm triggers noticeably sooner |
| No face | Move out of the camera's view | "No face detected"; no alarm |
| Missing model | Temporarily rename the `.dat` file | Program exits with download instructions |
| Missing sound | Temporarily rename `alarm.wav` | Warning printed; visual alert still works |
| Clean exit | Press `q` while the alarm is sounding | Sound stops, window closes, "Shut down cleanly." is printed |

## 5. Known limitations (useful for a report's discussion section)

- The HOG face detector works best on near-frontal faces. Strong head turns or a head drooping out of view mean no face is found, and the counter is then reset rather than flagged.
- Very low light reduces landmark accuracy. An infrared camera would help in night driving.
- The frame-based limit is tied to frame rate. On a slow machine, 48 frames takes longer in real seconds.
- EAR only captures eye closure. Other drowsiness signs such as yawning or head nodding are not modelled.

## References

- Soukupová, T. and Čech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks.* 21st Computer Vision Winter Workshop.
- Kazemi, V. and Sullivan, J. (2014). *One Millisecond Face Alignment with an Ensemble of Regression Trees.* CVPR.
- Dalal, N. and Triggs, B. (2005). *Histograms of Oriented Gradients for Human Detection.* CVPR.
- King, D. E. (2009). *Dlib-ml: A Machine Learning Toolkit.* Journal of Machine Learning Research, 10, 1755–1758.
