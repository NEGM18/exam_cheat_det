import urllib.request
import os
import ssl
import certifi

def download_file(url, filename):
    print(f"Downloading {filename}...")
    try:
        # Create SSL context with verified certificates
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=context) as response, open(filename, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"✅ {filename} downloaded successfully!")
    except Exception as e:
        print(f"❌ Error downloading {filename}: {str(e)}")
        return False
    return True

def main():
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # YOLO model files
    files = {
        "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
        "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    }
    
    # Download each file
    success = True
    for filename, url in files.items():
        if not os.path.exists(filename):
            if not download_file(url, filename):
                success = False
        else:
            print(f"✅ {filename} already exists!")
    
    if success:
        print("\n✅ All model files downloaded successfully!")
    else:
        print("\n⚠️ Some files failed to download. Please try again.")

if __name__ == "__main__":
    main() 