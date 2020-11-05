from PIL import Image
import os
import PIL

## image가 class별 폴더에 존재할 경우 image resize
## resize를 진행하고 돌릴 시 병목현상 x 학습속도 빨라짐

folder_path = './datasets/alopecia/'
save_folder_path = './datasets/alopecia_resize/'
foldernames = os.listdir(folder_path) ## 폴더명 담긴 목록
for foldername in foldernames:
    files_path = os.path.join(folder_path,foldername) ## files_path = 폴더경로+폴더명(파일이 있는 폴더 경로)
    filesname = os.listdir(files_path) ## filesname =  파일이름 목록
    for filename in filesname: ## filename = 파일이름
        readfile_path = os.path.join(files_path,filename) ## readfile_path = 폴더경로 + 폴더명(files_path) + 파일이름(filename)
        savefile_path = os.path.join(save_folder_path, foldername) ## savefile_path = 저장폴더경로(save_folder_path) + 폴더명(files_path) => 저장디렉토리
        if not os.path.exists(savefile_path): ## 저장디렉토리 없으면 만들어라~
            os.makedirs(savefile_path)
        im = PIL.Image.open(readfile_path)
        width, height = im.size
        im = im.resize((width // 16, height // 16))
        im.save(savefile_path+'/'+filename)  ## 저장디렉토리 + / + 파일이름