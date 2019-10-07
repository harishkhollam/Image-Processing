# README

# Codes are Arranged on the basis of the task and operations performed on the images.
## Merging of all the outputs are required. Please rename the dataset with version in order to streamline the merging process.

1. **Find-QR :** Find QR codes and extract data from files inside a folder
2. **Image-Metrics :**  Get Image metrics like 
    - Image Name
    - Image Size
    - Width
    - Height
    - Blur Score
    - Noise Score
    - Sharpness
    - Nima Score
1. **rename-images :** Rename images (Note : Add Dataset version) Please save the mapping of changed using 
    >> python rename.py -i \<pathname> >> rename_mapping.csv

## Installing
1.  Create a Virtual Environment

    `$ pip install virtualenv`

    `$ virtualenv .VENV && source .VENV/bin/activate && pip install -r requirements.txt`

