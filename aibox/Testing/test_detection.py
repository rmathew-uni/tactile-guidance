import subprocess

# Command to run in the terminal
command = "python ultralytics/detect.py --weights ultralytics/runs/train/exp8/weights/best.pt --source battery/"

# Run the command and capture the output
try:
    output = subprocess.check_output(command, shell=True, universal_newlines=True)
    print("Command output:")
    print(output)
except subprocess.CalledProcessError as e:
    print(f"Error running command: {e}") 