import os
import random
import numpy as np
import uuid
import scipy.ndimage
import sys 

PATH_TO_LIGHT_BACKGROUNDS = 'data/light_backgrounds/'
PATH_TO_DARK_BACKGROUNDS = 'data/dark_backgrounds/'
PATH_TO_FONT_FILES = 'data/fonts/'

# Get all files from directory
def get_files_from_dir(dirname):
  list_files = (os.listdir(dirname))
  list_files = [dirname + x for x in list_files]
  return list_files

# Random perspective distortion created by randomly moving the for corners of the image.
def get_distort_arg():
  amount = 5
  hundred_minus_amount = 100 - amount
  return '\'0,0 ' + str(np.random.randint(0,amount)) + ',' + str(np.random.randint(0,amount)) + ' 100,0 '  + str(np.random.randint(hundred_minus_amount,100)) + ',' + str(np.random.randint(0,amount)) + ' 0,100 '  + str(np.random.randint(0,amount)) + ',' + str(np.random.randint(hundred_minus_amount,100)) + ' 100,100 '  + str(np.random.randint(hundred_minus_amount,100)) + ',' + str(np.random.randint(hundred_minus_amount,100)) + '\''

# Randomly extracts 32x32 regions of an image and saves it to outdir
def create_random_crops(image_filename, num_crops, out_dir):
  im_height, im_width, channels = scipy.ndimage.imread(image_filename).shape

  for i in range(0, num_crops):
    # Randomly select first co-ordinate of square for cropping image
    x = random.randint(0,im_width - 32)
    y = random.randint(0,im_height - 32)
    outfile = uuid.uuid4().hex + '.jpg'
    command = "magick convert "+ image_filename + " -crop 32x32"+"+"+str(x)+"+"+str(y)+" " + os.path.join(out_dir, outfile)
    os.system(str(command))

# Generate crops for all files in file_list and store them in dirname
def generate_crops(file_list, dirname):
  if not os.path.isdir(dirname):
    os.mkdir(dirname)
    for f in file_list:
      create_random_crops(f, 10, dirname)

# Generate crops for all files in file_list and store them in dirname
def generate_image(generateType, NUM_IMAGES_PER_CLASS):
 
  if generateType == 'test':
    OUTPUT_DIR = 'data/test/'
  else:
    OUTPUT_DIR = 'data/train/'

  char_list = []
  for i in range(65, 65+26):
    char_list.append(chr(i))

  color_light = ['white','lime','gray','yellow','silver','aqua']
  color_dark = ['black','green','maroon','blue','purple','red']
  light_backgrounds = get_files_from_dir(PATH_TO_LIGHT_BACKGROUNDS)
  dark_backgrounds = get_files_from_dir(PATH_TO_DARK_BACKGROUNDS)
  list_files_fontt = get_files_from_dir(PATH_TO_FONT_FILES)
  light_backgrounds_crops_dir = 'data/light_backgrounds_crops/'
  dark_backgrounds_crops_dir = 'data/dark_backgrounds_crops/'

  generate_crops(light_backgrounds, light_backgrounds_crops_dir)
  generate_crops(dark_backgrounds, dark_backgrounds_crops_dir)

  light_backgrounds = get_files_from_dir(light_backgrounds_crops_dir)
  dark_backgrounds = get_files_from_dir(dark_backgrounds_crops_dir)

  all_backgrounds = [dark_backgrounds, light_backgrounds]

  for i in range(0,len(char_list)):
    char = char_list[i]
    char_output_dir = OUTPUT_DIR + str(char) + "/"
    
    if not os.path.exists(char_output_dir):
      os.makedirs(char_output_dir)

    # Generate images
    for j in range(0,NUM_IMAGES_PER_CLASS):
      
      path = random.choice(all_backgrounds)
      list_filernd = random.choice(path)
      list_rfo = random.choice(list_files_fontt)
      distort_arg = get_distort_arg()
      blur = random.randint(0,3)
      noise = random.randint(0,5)
      x = str(random.randint(-3,3))
      y = str(random.randint(-3,3))
      if path == all_backgrounds[0] :
        color = random.choice(color_light)
      else:
        color = random.choice(color_dark)

      command =  "magick convert " + str(list_filernd) + " -fill "+str(color)+" -font "+ \
              str(list_rfo) + " -weight 200 -pointsize 24 -distort Perspective "+str(distort_arg)+" "+"-gravity center" + " -blur 0x" + str(blur) \
      + " -evaluate Gaussian-noise " + str(noise) +  " " + " -annotate +" + x + "+" + y + " " +  str(char_list[i]) + " " + char_output_dir + "output_file"+str(i)+str(j)+".jpg"
      
      os.system(str(command))

if(len(sys.argv) != 3):
    print("Please refer read me and type argument")
    sys.exit(0)

gen_type = sys.argv[1]
numberOfData = int(sys.argv[2])
generate_image(gen_type,numberOfData)