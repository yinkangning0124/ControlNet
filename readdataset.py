import numpy as np

body_pose_path = '/home/aida/kangning/ControlNet/annotator/ckpts/body_pose_model.pth'
hand_pose_file = '/home/aida/kangning/ControlNet/annotator/ckpts/hand_pose_model.pth'

body_pose = np.load(body_pose_path, allow_pickle=True)
hand_pose = np.load(hand_pose_file, allow_pickle=True)
