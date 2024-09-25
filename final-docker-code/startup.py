import os
import configparser
import sys

def create_config_and_dirs(base_dir):
    config = configparser.ConfigParser()
    config_path = os.path.join(base_dir, 'config.ini')

    if os.path.exists(config_path):
        print(f"Config file already exists at {config_path}. Using existing config.")
        config.read(config_path)
    else:
        config['Cameras'] = {
            'RTSPLink1': 'rtsp://dummy:dummy@0.0.0.0:554/dummy1',
            'RTSPLink2': 'rtsp://dummy:dummy@0.0.0.0:554/dummy2'
        }
        config['Interval'] = {
            'Detection_Interval': '5'
        }
        config['Paths'] = {
            'SavedPlatesDir': os.path.join(base_dir, 'saved_plates'),
            'CleanPlatesDir': os.path.join(base_dir, 'clean_plates')
        }

        with open(config_path, 'w') as configfile:
            config.write(configfile)
        print(f"Created new config file at {config_path}")

    os.makedirs(config['Paths']['SavedPlatesDir'], exist_ok=True)
    os.makedirs(config['Paths']['CleanPlatesDir'], exist_ok=True)
    print(f"Ensured directories exist: {config['Paths']['SavedPlatesDir']}, {config['Paths']['CleanPlatesDir']}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the base directory as an argument.")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    create_config_and_dirs(base_dir)
    os.system(f"python3 /app/anpr_script.py {base_dir}")
