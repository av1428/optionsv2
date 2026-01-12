import zipfile
import os

def zip_directory(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            # Exclude directories
            dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git', '.idea', 'tmp']]
            
            for file in files:
                if file == os.path.basename(output_path):
                    continue
                if file.endswith('.pyc') or file.endswith('.DS_Store'):
                    continue
                    
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
    print(f"Created {output_path}")

if __name__ == "__main__":
    folder = r"c:\Users\vardh\Downloads\stockwebapp\Options"
    output = r"c:\Users\vardh\Downloads\stockwebapp\Options\stockwebapp_v1.zip"
    zip_directory(folder, output)
