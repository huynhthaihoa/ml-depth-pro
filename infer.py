# from PIL import Image
import numpy as np
import argparse
import matplotlib
import cv2
import depth_pro

'''new utility functions'''
def depth_value_to_depth_image(depth_values, min_depth=None, max_depth=None, cmap='magma'):# applyReverse=False):#, applyColor=False):
    """Convert depth value to depth image for visualization (in magma colormap/grayscale)

    Args:
        depth_values: input depth value
        min_depth (default is None): maximum depth value in visualization range (if max_depth is None, it is set as max value of depth_values)
        max_depth (default is None): maximum depth value in visualization range (if max_depth is None, it is set as max value of depth_values)
        cmap: cmap to display, recommended value is 'magma' for short-range depth (like NYUv2) and 'magma_r' for long-range depth (like KITTI)
    Returns:
        depth visualization
    """
    try:    
        cmapper = matplotlib.cm.get_cmap(cmap)
    except:
        raise TypeError
    
    if min_depth is None:
        min_depth = depth_values.min()  #else min_depth
    if max_depth is None:
        max_depth = depth_values.max() #else max_depth

    depth_values_clip = np.clip(depth_values, min_depth, max_depth)
    
    if max_depth != min_depth:
        depth_values_vis = (depth_values_clip - min_depth) / (max_depth - min_depth)
    else:
        depth_values_vis = depth_values_clip * 0.
        
    depth_values_vis = cmapper(depth_values_vis, bytes=True)

    img = depth_values_vis[:, :, :3]
    
    return img[:, :, ::-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DepthPro inference (video/webcam)', fromfile_prefix_chars='@')

    # Input
    parser.add_argument('-i', '--input', help='Input path (image/video/camera index)', type=str, default='0')
    
    # Output
    parser.add_argument('-o', '--output', help='Output path (image(s)/video)', type=str, default='0')

    args = parser.parse_args()

    # Load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms(device="cuda:0")
    model.eval()

    # Read input
    input = args.input
    
    if isinstance(input, str) and (input.endswith('png') or input.endswith('jpg') or input.endswith('jpeg') or input.endswith('png')): # input is image
        # Load and preprocess an image.
        image, _, f_px = depth_pro.load_rgb(input)
        image = transform(image)

        # Run inference.
        prediction = model.infer(image, f_px=f_px)
        depth = prediction["depth"]  # Depth in [m].
        focallength_px = prediction["focallength_px"]  # Focal length in pixels.
        
        depth_vis = depth_value_to_depth_image(depth)
        if args.output != '':
            cv2.imwrite(args.output, depth_vis)
    else:
        cam = cv2.VideoCapture(input)
        
        writer = None
        
        #Inference loop
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            
            # Load and preprocess an image.
            image, _, f_px = depth_pro.load_opencvimage(frame)
            image = transform(image)
            
            # Run inference.
            prediction = model.infer(image, f_px=f_px)
            depth = prediction["depth"].detach().cpu().numpy()  # Depth in [m].
            print(depth.min(), depth.max())
            focallength_px = prediction["focallength_px"]  # Focal length in pixels.
            
            depth_vis = depth_value_to_depth_image(depth)
            if args.output != '':
                if writer is None:
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), 30, (depth_vis.shape[1], depth_vis.shape[0]))
                writer.write(depth_vis)
        
        if writer is not None:
            writer.release()
            