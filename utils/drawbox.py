import numpy as np
import cv2
import torch.nn.functional as F

def xyz_to_xy(P, xyz):
    projected_xy = np.array(list(xyz)+[1])@P.T
    projected_xy = projected_xy/projected_xy[2:]
    return projected_xy[:2]
    
def draw_line(image, point1, point2, color=(0, 255, 0), thickness=1):
    # Convert points to integers
    point1 = tuple(map(int, point1))
    point2 = tuple(map(int, point2))

    cv2.line(image, point1, point2, color, thickness) # Draw the line on the image

def draw_boxes(rgb_input, cam_input, annotation_boxes, annotation_labels):

    random_scene_image_idx = np.random.choice(rgb_input.shape[0])
    im = (((rgb_input[random_scene_image_idx].permute(1,2,0)))*255).cpu().numpy().astype('uint8').copy()
    P = cam_input[random_scene_image_idx].cpu().numpy().copy()
    target_box, target_cls = annotation_boxes.detach().cpu().numpy(), annotation_labels.detach().cpu().numpy()
    stride = 1
    P[:2,:] = P[:2,:] / stride

    for i in range(len(target_box)):

        if target_cls[i] == 2:
            continue

        color = (0,255,0) if target_cls[i] == 0 else (255,0,0)
        

        xyz, hwl = target_box[i][:3], target_box[i][3:]

        cx,cy,cz = xyz
        half_length = hwl[0] / 2
        half_width = hwl[1] / 2
        half_height = hwl[2] / 2

        vertices_3d = [
            (cx - half_length, cy - half_width, cz - half_height),  # 0: bottom-left-back
            (cx + half_length, cy - half_width, cz - half_height),  # 1: bottom-right-back
            (cx + half_length, cy + half_width, cz - half_height),  # 2: top-right-back
            (cx - half_length, cy + half_width, cz - half_height),  # 3: top-left-back
            (cx - half_length, cy - half_width, cz + half_height),  # 4: bottom-left-front
            (cx + half_length, cy - half_width, cz + half_height),  # 5: bottom-right-front
            (cx + half_length, cy + half_width, cz + half_height),  # 6: top-right-front
            (cx - half_length, cy + half_width, cz + half_height)   # 7: top-left-front
        ]

        vertices_2d = [xyz_to_xy(P, verts_) for verts_ in vertices_3d]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Back face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Front face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
        ]

        for edge in edges:
            st_point = (vertices_2d[edge[0]]).astype('int')
            end_point = (vertices_2d[edge[1]]).astype('int')
            draw_line(im, st_point, end_point, color = color)

    return im

def create_box_visualization(rgb_images, cam_p, targets, outputs):
    ims = []
    for batch_idx in range(len(rgb_images)):
        rgb_input = rgb_images[batch_idx]
        cam_input = cam_p[batch_idx]
        annotation_boxes = targets[batch_idx]['boxes']
        annotation_labels = targets[batch_idx]['labels']

        prob = F.softmax(outputs['pred_logits'], -1)
        _, labels = prob.max(-1)
        output_boxes = outputs['pred_boxes'][batch_idx]
        output_labels = labels[batch_idx]

        im_gt = draw_boxes(rgb_input, cam_input, annotation_boxes, annotation_labels)
        im_pred = draw_boxes(rgb_input, cam_input, output_boxes, output_labels)

        im = np.concatenate([im_gt, im_pred], 0)
        ims.append(im)

    return np.concatenate(ims, 1)
