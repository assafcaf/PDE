import numpy as np

def smooth(y, box_pts=9):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth[box_pts-1:-box_pts+1]

def smooth_2d(arr, box_pts=9):
    # Create the box filter
    if type(arr) == list:
        arr = np.array(arr)
    
    arr = arr.T    
    box = np.ones(box_pts)/box_pts
    
    # Initialize an empty array to hold the smoothed results
    smoothed_arr = np.empty_like(arr)

    # Apply smoothing for each row independently
    for i in range(arr.shape[0]):
        smoothed_row = np.convolve(arr[i, :], box, mode='same')
        smoothed_arr[i, :] = smoothed_row

    return smoothed_arr[:, box_pts-1:-box_pts+1].T