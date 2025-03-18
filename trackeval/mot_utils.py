import numpy as np
from scipy.optimize import linear_sum_assignment
from trackeval.utils import _load_simple_text_file


def load_and_preprocess_data(gt_path, tracker_path):
    """Loads and preprocesses MOTChallenge 2D data (class-agnostic)."""

    raw_gt_data = _load_raw_file(gt_path)
    raw_tracker_data = _load_raw_file(tracker_path)

    num_timesteps = _combine_gt_and_tracker(raw_gt_data, raw_tracker_data)
    preprocessed_data = _preprocess_data(raw_gt_data, raw_tracker_data, num_timesteps)

    return preprocessed_data
    
def _load_raw_file(file_path):
    """Loads a single MOTChallenge file (GT or tracker)."""
    read_data = _load_simple_text_file(file_path, is_zipped=False, zip_file=None)

    data_by_timestep = {}
    for row in read_data:
        timestep = int(row[0])  # Frame number is always the first element
        if timestep not in data_by_timestep:
            data_by_timestep[timestep] = []
        data_by_timestep[timestep].append(row)

    return data_by_timestep


def _combine_gt_and_tracker(raw_gt_data, raw_tracker_data):
    """Combines GT and tracker data to determine the number of timesteps."""
    # Find the maximum timestep across both GT and tracker data.
    max_gt_timestep = 0
    for k in raw_gt_data.keys():
        try:
            timestep = int(k)
            max_gt_timestep = max(max_gt_timestep, timestep)
        except ValueError:
            pass

    max_tracker_timestep = 0
    for k in raw_tracker_data.keys():
        try:
            timestep = int(k)
            max_tracker_timestep = max(max_tracker_timestep, timestep)
        except ValueError:
            pass

    return max(max_gt_timestep, max_tracker_timestep)


def _preprocess_data(raw_gt_data, raw_tracker_data, num_timesteps):
    """Preprocessing steps (MODIFIED from MotChallenge2DBox.get_preprocessed_seq_data).
        - Class-agnostic.
        - Keeps matching and zero_marked removal.
        - Correctly handles variable number of columns in GT.
    """
    data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'similarity_scores']
    data = {
        'gt_ids': [None] * num_timesteps,
        'tracker_ids': [None] * num_timesteps,
        'gt_dets': [None] * num_timesteps,
        'tracker_dets': [None] * num_timesteps,
        'similarity_scores': [None] * num_timesteps,
        'num_timesteps': num_timesteps
    }
    
    for t in range(num_timesteps):
        time_key = t + 1

        # Get and check gt data.
        if time_key in raw_gt_data.keys():
            gt_time_data = np.array(raw_gt_data[time_key], dtype=np.float64)
        else:
            gt_time_data = None

        if gt_time_data is not None:
            data['gt_dets'][t] = np.atleast_2d(gt_time_data[:, 2:6])
            data['gt_ids'][t] = np.atleast_1d(gt_time_data[:, 1]).astype(int)
            if gt_time_data.shape[1] >= 7:
                gt_zero_marked = np.atleast_1d(gt_time_data[:, 6].astype(int))
            else:
                gt_zero_marked = np.ones_like(data['gt_ids'][t])
        else:
            data['gt_dets'][t] = np.empty((0, 4))
            data['gt_ids'][t] = np.empty(0).astype(int)
            gt_zero_marked = np.empty(0)

        if time_key in raw_tracker_data.keys():
            tracker_time_data = np.array(raw_tracker_data[time_key], dtype=np.float64)
        else:
            tracker_time_data = None

        if tracker_time_data is not None:
            data['tracker_dets'][t] = np.atleast_2d(tracker_time_data[:, 2:6])
            data['tracker_ids'][t] = np.atleast_1d(tracker_time_data[:, 1]).astype(int)
            tracker_confidences = np.atleast_1d(tracker_time_data[:, 6])
        else:
            data['tracker_dets'][t] = np.empty((0, 4))
            data['tracker_ids'][t] = np.empty(0).astype(int)
            tracker_confidences = np.empty(0)
            
        # IoU matching
        if data['gt_ids'][t].shape[0] > 0 and data['tracker_ids'][t].shape[0] > 0:
            ious = _calculate_box_ious(data['gt_dets'][t], data['tracker_dets'][t])
            match_rows, match_cols = linear_sum_assignment(-ious)
            actually_matched_mask = ious[match_rows, match_cols] > 0.5 - np.finfo('float').eps
            match_rows = match_rows[actually_matched_mask]
            match_cols = match_cols[actually_matched_mask]
            data['similarity_scores'][t] = ious

        else:
             data['similarity_scores'][t] = np.zeros(
                (data['gt_ids'][t].shape[0] if data['gt_ids'][t] is not None else 0,
                 data['tracker_ids'][t].shape[0] if data['tracker_ids'][t] is not None else 0))

        # Remove gt detections marked as to remove (zero marked)
        gt_to_keep_mask = np.not_equal(gt_zero_marked, 0)
        data['gt_ids'][t] = data['gt_ids'][t][gt_to_keep_mask]
        data['gt_dets'][t] = data['gt_dets'][t][gt_to_keep_mask]
        if data['similarity_scores'][t].shape[0] > 0:
            data['similarity_scores'][t] = data['similarity_scores'][t][gt_to_keep_mask]

    data['num_tracker_dets'] = 0
    data['num_gt_dets'] = 0
    data['num_tracker_ids'] = 0
    data['num_gt_ids'] = 0

    for t in range(num_timesteps):
        data['num_tracker_dets'] += len(data['tracker_ids'][t])
        data['num_gt_dets'] += len(data['gt_ids'][t])
        # Ensure that tracker and gt ids are consecutive across timesteps
        if len(data['tracker_ids'][t]) > 0:
            data['num_tracker_ids'] = max(data['num_tracker_ids'], np.max(data['tracker_ids'][t]) + 1)
        if len(data['gt_ids'][t]) > 0:
            data['num_gt_ids'] = max(data['num_gt_ids'], np.max(data['gt_ids'][t]) + 1)

    return data


def _calculate_box_ious(gt_dets, tracker_dets, box_format='xywh'):
    """Calculate IOU between two sets of bounding box detections.
    
    Args:
        gt_dets: Ground truth detections
        tracker_dets: Tracker detections
        box_format: Format of input boxes. Either 'xywh' (x,y,width,height) or 
                   'x0y0x1y1' (top-left x,y and bottom-right x,y)
    """

    if box_format == 'xywh':
        # Convert 'xywh' to 'x0y0x1y1' for calculation
        gt_dets = gt_dets.copy()
        tracker_dets = tracker_dets.copy()
        gt_dets[:, 2] = gt_dets[:, 0] + gt_dets[:, 2]  # x1 = x0 + w
        gt_dets[:, 3] = gt_dets[:, 1] + gt_dets[:, 3]  # y1 = y0 + h
        tracker_dets[:, 2] = tracker_dets[:, 0] + tracker_dets[:, 2]
        tracker_dets[:, 3] = tracker_dets[:, 1] + tracker_dets[:, 3]
    elif box_format != 'x0y0x1y1':
        raise ValueError("Invalid box_format. Supported formats are 'xywh' and 'x0y0x1y1'")

    min_ = np.minimum(gt_dets[:, np.newaxis, :], tracker_dets[np.newaxis, :, :])
    max_ = np.maximum(gt_dets[:, np.newaxis, :], tracker_dets[np.newaxis, :, :])

    # Intersection width and height
    iw = np.maximum(min_[..., 2] - max_[..., 0], 0)  # Clip at 0
    ih = np.maximum(min_[..., 3] - max_[..., 1], 0)
    ia = iw * ih  # Intersection area

    # Areas of GT and tracker boxes
    area1 = (gt_dets[..., 2] - gt_dets[..., 0]) * (gt_dets[..., 3] - gt_dets[..., 1])
    area2 = (tracker_dets[..., 2] - tracker_dets[..., 0]) * (tracker_dets[..., 3] - tracker_dets[..., 1])

    # Union area (broadcasting for pairwise calculation)
    union = area1[:, np.newaxis] + area2[np.newaxis, :] - ia

    # Handle cases where area1, area2, or union are zero
    intersection = np.copy(ia)
    intersection[area1 <= 0 + np.finfo('float').eps, :] = 0
    intersection[:, area2 <= 0 + np.finfo('float').eps] = 0
    intersection[union <= 0 + np.finfo('float').eps] = 0
    union[union <= 0 + np.finfo('float').eps] = 1

    ious = intersection / union
    return ious