# bikefitting

Estimate your bike fit from a side-view video. Uses MediaPipe pose detection to measure knee, hip, torso and elbow angles over all pedal strokes and suggests a saddle height change.

## Install

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

The pose model (~30 MB) is downloaded on first run.

## Usage

```sh
uv run bikefit.py ride.mov
uv run bikefit.py ride.mov --height 180
```

| Option | Description |
| --- | --- |
| `--height CM` | Body height, gives the saddle change in mm |
| `--side left\|right` | Side facing the camera (default: auto) |
| `--no-preview` | Skip the live window while analyzing (`q` closes it) |
| `--no-video` | Skip the annotated video |

Output goes to `<video>_bikefit/`: `report.txt`, `frames.csv`, `top.png`, `bottom.png`, `annotated.mp4`.

## Filming

Camera exactly from the side at hip height, whole body and bike in frame, good light, a few steady pedal revolutions.

## Limits

2D pose estimation is typically off by 3-9°. This does not replace a professional fit.

## License

MIT
