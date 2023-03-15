# Preprocessing license plate LP data for training/testing
# Step 1: unzip external data
# Step 2 (optional): super-resolution 
# Step 3: resize images, optimal LP image input: [100 x 32]
# Step 4: organize LP label information in dataframe 
# Step 5: prepare lmd dataset files (train/validation/test)



# Step 1: unzip external data
BASEDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
unzip ../../data/external/[라벨]자동차번호판OCR_train -d ../../data/raw/Label/
unzip ../../data/external/[원천]자동차번호판OCR데이터 -d ../../data/raw/Image/


# Step 2 (optional): super-resolution 
# python superresol_images.py \
#     --input_loc "/home/jovyan/sod/sod-license-plate/src/data/test/" \
#     --output_loc "/home/jovyan/sod/sod-license-plate/src/data/test_out/" \
#     --iternum 10 \



# Step 3: resize images, optimal LP image input: [100 x 32]
python resize_images.py \
    --input_loc "/home/jovyan/sod/sod-license-plate/src/data/test_out/" \
    --output_loc "/home/jovyan/sod/sod-license-plate/src/data/test_out_resize/" \
    --width 100 \
    --height 32 \
              

# Step 4: organize LP contents in a pickle file from json files 
python get_info.py \
    --input_loc "/home/jovyan/sod/sod-license-plate/src/data/raw/Label" \
    --output_loc "/home/jovyan/sod/sod-license-plate/src/data/test_out/" \
              
