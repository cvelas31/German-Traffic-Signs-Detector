# German-Traffic-Signs-Detector

## KIWI CHALLENGE DEEP LEARNING
### In the reports directory is more explicit information
### To execute the program:
python app.py [command]

Commands: <br />
  download --  Download images from the German Traffic Signs <br />
  infering -- Infer from model # from images in directory <br />
  test  --    Test Model # from images in directory <br />
  train  --   Train model # from images in directory <br />

#### python app.py download [Options]
Description: <br />
This command will download all data from the German Traffic Signs Dataset <br />
Default=(http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset). <br />
This will store the the data set inside the images folder. <br />
It will split the images between train and test, and will be saved in their correspondent folder <br />

Options: <br />
-u, --url  (OPTIONAL) URL where the image datset is going to be downloaded. <br />
--help <br />

#### python app.py train [Options]
Description: <br />
Train model # from images in directory. <br />
  INPUT: <br />
  model: Number of model or name to train <br />
      (0 - 'SK_LR' refers to SK logistic regression) <br />
      (1 - 'TF_LR' refers to  TF Logistic regression) <br />
      (2 - 'TF_LENET' refers to  TF LENET) <br />

  directory: Directory where the train images are placed in format .ppm <br />
  
-m, --model  ---   Number of model or name to train. See description above <br />
-d, --directory  --- Directory where the train images are placed in format .ppm <br />
--help <br />

#### python app.py test [Options]
Description: <br />
Test model # from images in directory. <br />
  INPUT: <br />
  model: Number of model or name to train <br />
      (0 - 'SK_LR' refers to SK logistic regression) <br />
      (1 - 'TF_LR' refers to  TF Logistic regression) <br />
      (2 - 'TF_LENET' refers to  TF LENET) <br />

  directory: Directory where the test images are placed in format .ppm <br />
  
-m, --model   ---  Number of model or name to train. See description above <br />
-d, --directory --- Directory where the test images are placed in format .ppm <br />
--help <br />

#### python app.py infering [Options]
Description: <br />
Infering from model # from images in directory. <br />
  INPUT: <br />
  model: Number of model or name to train <br />
      (0 - 'SK_LR' refers to SK logistic regression) <br />
      (1 - 'TF_LR' refers to  TF Logistic regression) <br />
      (2 - 'TF_LENET' refers to  TF LENET) <br />

  directory: Directory where the infer or user images are placed in format .ppm <br />
  
-m, --model    ---  Number of model or name to train. See description above <br />
-d, --directory --- Directory where the infer or user images are placed in format .ppm <br />
--help <br />




