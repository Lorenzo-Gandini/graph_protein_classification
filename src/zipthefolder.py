import tarfile
import os

def gzip_folder(folder_path, output_file):
    """
    Compresses an entire folder into a single .tar.gz file.
    
    Args:
        folder_path (str): Path to the folder to compress.
        output_file (str): Path to the output .tar.gz file.
    """
    print(os.getcwd())
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(folder_path, arcname=os.path.basename(folder_path))
    print(f"Folder '{folder_path}' has been compressed into '{output_file}'")



if __name__=="__main__":
    # Example usage
    folder_path = "./sub_gcod"            # Path to the folder you want to compress
    output_file = "./sub_gcod/solution.gz"         # Output .gz file name
    gzip_folder(folder_path, output_file)