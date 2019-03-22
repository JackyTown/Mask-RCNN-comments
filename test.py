import torch
import numpy as np
import math
# boxes = torch.Tensor([[-1,2,3,4]])
# window = torch.Tensor([0,1,2,3])

# boxes = torch.stack( \
#         [boxes[:, 0].clamp(float(window[0]), float(window[2])),
#          boxes[:, 1].clamp(float(window[1]), float(window[3])),
#          boxes[:, 2].clamp(float(window[0]), float(window[2])),
#          boxes[:, 3].clamp(float(window[1]), float(window[3]))], 1)
# print(boxes)


# a,b = np.meshgrid(np.array([0,1]), np.array([0.5,1,1.5]))
# print(a.flatten())
# print(b)
roi_level = torch.Tensor([2,3,4])
for i, level in enumerate(range(2, 6)): # (0,2),(1,3)……
    ix  = roi_level==level
    # print(ix)
    if not ix.any():
        continue
    print(torch.nonzero(ix))
    ix = torch.nonzero(ix)[:,0]
    print (ix)
    # level_boxes = boxes[ix.data, :]

# print(math.log(4,2))
# a = torch.Tensor([0,1,1])
# print(torch.nonzero(a))