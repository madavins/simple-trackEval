
# Simplified Multiple Object Tracking evaluation
A simplified version of [TrackEval](https://github.com/JonathonLuiten/TrackEval) focused on Multiple Object Tracking (MOT) evaluation in [MOTChallenge](https://motchallenge.net/) format. This tool provides a streamlined interface for evaluating object tracking results against ground truth data and supports multiple evaluation metrics including HOTA (recommended tracking metric), CLEAR, Identity, Count, and VACE.

> Note: This project is currently under active development.

## Installation
```bash
# Clone the repository
git clone https://github.com/madavins/simple-TrackEval.git
cd simple-TrackEval

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Usage
The evaluation can be run using the provided script:
```bash
python simple_eval.py <ground_truth_path> <tracker_results_path> [options]
```

### Example:
```bash
python simple_eval.py \
    data/gt.txt \
    data/tracker_results.txt \
    --metrics HOTA CLEAR Identity \
    --output_dir results \
    --plots \
    --tracker_name StrongSort
```

## Input format
### Tracker results file
The tracker results should be provided as a CSV text-file where each line represents a detection of an object instance in a specific frame. Each line must contain 10 comma-separated values in the following order:

```bash
<frame_number>, <track_id>, <bounding_box_left>, <bounding_box_top>, <bounding_box_width>, <bounding_box_height>, <confidence_score>, <x>, <y>, <z>
```

- **frame_number**: The frame index in which the object is detected. Frame numbers are 1-based.
- **track_id**: The ID of the tracked object. Track IDs are 1-based.
- **bounding_box_left**: The x-coordinate of the top-left corner of the bounding box. 1-based.
- **bounding_box_top**: The y-coordinate of the top-left corner of the bounding box. 1-based.
- **bounding_box_width**: The width of the bounding box.
- **bounding_box_height**: The height of the bounding box.
- **confidence_score**: The detection confidence score. Use `-1` if not available.
- **x, y, z**: These coordinates are ignored for the 2D MOT challenge and can be filled with `-1`.

Example:
```
1, 3, 794.27, 247.59, 71.245, 174.88, -1, -1, -1, -1
1, 6, 1648.1, 119.61, 66.504, 163.24, -1, -1, -1, -1
1, 8, 875.49, 399.98, 95.303, 233.93, -1, -1, -1, -1
```

### Ground truth file
The ground truth data should be provided as a text-file with each line representing a ground truth object instance. Each line must contain 9 comma-separated values in the following order:

```bash
<frame_id>, <track_id>, <x>, <y>, <width>, <height>, <valid_flag>, <class_id>, <visibility>
```

- **frame_id**: The frame index of the ground truth object. Frame IDs are 1-based.
- **track_id**: The ground truth object ID. Track IDs are 1-based.
- **x**: The x-coordinate of the top-left corner of the bounding box.
- **y**: The y-coordinate of the top-left corner of the bounding box.
- **width**: The width of the bounding box.
- **height**: The height of the bounding box.
- **valid_flag**:  Indicates whether the object is of interest for tracking evaluation. Set to `1` for objects to be considered, and `0` to ignore the object.
- **class_id**: The class ID of the object. While required, it is not considered in this version of the code as the evaluation is class-agnostic.
- **visibility**: The visibility of the object as a percentage value from 0 to 1 (0 for completely occluded, 1 for fully visible).

Example:
```
1, 1, 794, 247, 71, 174, 1, 1, 1
1, 2, 1648, 119, 66, 163, 1, 1, 1
1, 3, 875, 399, 95, 233, 1, 1, 1
```
