from os import listdir
from os.path import isfile, join

train_list = [f for f in listdir("DSG/train_resize/") if isfile(join("DSG/train_resize/", f))]
test_list =  [f for f in listdir("DSG/test_resize/") if isfile(join("DSG/test_resize/", f))]
