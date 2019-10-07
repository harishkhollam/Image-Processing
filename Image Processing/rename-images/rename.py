# Pythono3 code to rename multiple 
# files in a directory or folder 

# importing os module 
import os 
from PIL import Image 

# Function to rename multiple files 
def main():
    print("Orignal_name,New_name")
    i = 1
    path = 'input/'
    destpath = 'output/'
    for filename in os.listdir(path):
        
        newname ="V2image" + str(i) + ".jpg"
        print('{},{}'.format(filename,newname))
        
        src = path+filename 
        dst = destpath+newname
        # rename() function will
        # rename all the files
        os.rename(src, dst)
        i += 1

# Driver Code 
if __name__ == '__main__': 
	
	# Calling main() function 
	main() 

