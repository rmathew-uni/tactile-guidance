import detect_script_run
import box_to_FourDirection

if __name__ == '__main__':
    weights_obj = 'aibox/yolov5s.pt'  # Model weights path
    weights_hand = 'aibox/hand.pt'
    source = '1'  # Input image path
    # Add other parameters as needed

    detect_script_run.main(weights_obj, weights_hand, source)
    
