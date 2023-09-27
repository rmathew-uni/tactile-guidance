import detect_script_run
import box_to_FourDirection

if __name__ == '__main__':
    weights = 'yolov5s.pt'  # Model weights path
    source = '0'  # Input image path
    # Add other parameters as needed

    detect_script_run.main(weights, source)