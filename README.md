# SDL2 developer event "Walking on Thin Clouds"
Please, **carefully follow this README when working on this developer event!**

## How to get help during the event
Many of the organizers will be phyically at the AI Sweden office in Stockholm, so
for those present physically it's possible to just ask those who are there. You 
should also have been invited to our Slack forum for this event, where you can
chat with us. Please note that we have added some specific people for some
different parts of the README below.

## Where to send results etcetera
Whenever this README says that something should be sent to the organizers or
similar, please use the following subject line
```
[SDL2 developer event] Group X -- <optional info>
```
where you should raplace X with your group's number, and `<optional info>` should
be replaced with any other relevant info about the email. For example, it
could say "test results for Task 1", or similar, depending on the context
of course.

Then send to the following emails:
```
aleksis.pirinen@ri.se; thomas.ohlson.timoudas@ri.se; gyorgy.kovacs@ltu.se; nosheen.abid@ltu.se
```

## Where to run the challenge and how
There are several possibilities for you to run the challenge. You can start by experimenting on your local machine for instance, 
since the datasets and the model we provide are quite lightweight, computationally speaking. 
To do so, we recommend to create a conda environment with the required modules we listed below in the dedicated section.
We offer also the possibility to run your experiments on AWS, in case your models need more than just a regular laptop. 
This is particularly true if you decide to use more satellite data, like for instance the kappazeta dataset we linked in the 
task description document or datasets you can download from the Digital Earth Sweden platform.
If you do want to use AWS, please carefully follow the instructions in the dedicated section of this readme file.


## If you want to set up your own Conda environment
On a Ubuntu work station, the below should be sufficient for running this developer event.
Note that if you prefer, a virtual environment could be used instead of Conda.
```
conda create -n hackathon python=3.8
conda activate hackathon
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
pip install scipy
pip install matplotlib
pip install xarray
pip install scikit-image
pip install netCDF4
```
You could also use the `sdl2_env.yml` file and run the following command

```conda env create --file sdl2_env.yml```

This will create the hackathon environment which you will then activate with 

```conda activate hackathon```


## If you want to use AWS
_Contact persons for help with the AWS setup: Kim, Joakim, Chiara._

Your team will be provided with EC2 instance on AWS. Go to the private group channel (e.g. group 4 will get access to #group4 channel). In that channel you will find the key (.pem) file to access the instance with ssh and the instance ID of your group. Note that the key is an openssh key.
1. ```mkdir SDL2_AWS``` directory and ```cd``` into it.
2. download the your-team.pem file in the folder you have just created
3. ```chmod 400 <your-team.pem>```
4. run the command (make sure to substitute the right file name and instance ID in the following command):
   ```ssh -i "<your-team.pem>" ubuntu@<your-group-instanceID>.eu-north-1.compute.amazonaws.com   ```
5. Once you are inside the instance, run the following commands:
 
   ```sudo apt update && sudo apt upgrade -y && sudo reboot ```
   
   Note: by running this command you will be notified of a kernel update. Press ok. There will be another panel with options. Just press ok again.
   Note 2: it will reboot the instance, so you will need to reconnect. It might take a few seconds, so be patient.
   
   ```sudo apt install nvidia-driver-535 -y && sudo apt install unzip -y && sudo apt install pip -y && sudo pip install torch && pip install scipy && pip install matplotlib && pip install xarray && pip install scikit-image && pip install netCDF4 && sudo reboot```

   Note: it will reboot the instance, so you will need to reconnect.
   Now your environment is ready. 
   
6. It is time to get the repository set up into the instance. In order to run the code that we provide, we need to create a folder tree structure as the following:
   
   ```mkdir SDL2_group<X>```
   
   ```cd SDL2_group<X>```
   
   ```mkdir data && mkdir log_smhi  && cd data && wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wrlbcA4RO73eeZxkXK28ghpP3QrUROre' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wrlbcA4RO73eeZxkXK28ghpP3QrUROre" -O temp.zip && rm -rf /tmp/cookies.txt && unzip temp.zip```

   while still in the data folder (not in the SDL2_SMHI_data subfolder!), run also the following command to get the Skogsstyrelsen data as well:
   
   ```wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=19MBh9JIJTxYIPAeO7G5RML5_ddjJ1Cpa' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19MBh9JIJTxYIPAeO7G5RML5_ddjJ1Cpa" -O temp2.zip && rm -rf /tmp/cookies.txt && unzip temp2.zip```

   After these steps, you should have:

   ```
   SDL2_group<X>
   |
   └─── data
   |   | temp.zip
   |   | temp2.zip
   |   └─── SDL2_SMHI_data
   |   └─── skogsstyrelsen
   |
   └─── log_smhi
   ```
   Note: you might need to rename the folder ```skogsstyrelsen_data```into ```skogsstyrelsen```.
   
## Running in Docker
_Contact person for help with Docker: Joel Sundholm._

To run in Docker you should:

1. Run  ```./docker_build``` to build the image. It is tagged "hack"
2. Run ```./docker_run```to start the image, with port forward on port 8888 for jupyter-notebook. You will get an interactive terminal inside the image. The Git repository is mounted inside the container.

These steps should set you up with the conda environment, directory structure, and data you need to run the training scripts.

To start the included jupyter notebook, navigate to notebooks and run the jupyter-start script: ```./jupyter-start```

Note: If you want to run the jupyter-lab on AWS, make sure that you port forward port 8888 to the EC2 instance. E.g. ```ssh -L 8888:localhost:8888 ...```

## Things to do prior to working on any code

1. **One** person within each group sends her/his github user name to the organizers.
The organizers will then add that person as a collaborator of this private repository.

2. Log in into your github account and got to https://github.com/settings/tokens.
   
3. Create a token and save it somewhere safe because you will need it later too.
   
4. Move up of one folder: ```cd ..``` 
5. Clone this git repository via the command `git clone https://github.com/aleksispi/sdl2-hackathon.git`. You will be asked to input your username and password. You should use the token you have just created as your password.

6. Move into the git directory you cloned using the command `cd sdl2-hackathon`.

7. Create a git branch for your group. If you are group X, then run the command
`git branch groupX` (if X is 4 in your case, it should say `git branch group4`).

8. Check the set of branches via the command `git branch`. You should now see the
two branches `groupX` and `master`.

9. Switch to your group's branch via `git checkout groupX`. **Do all development
within this branch** (_not_ in `master`).

10. Whenever you want, you can push code to your branch via `git push`. **Please
ensure that you push your final code at the end of the developer event!**
Note that the first push may require the command `git push --set-upstream origin groupX`. At this stage you
will be asked again to use the token as your password.
Also note that you may not wish to git push the code prior to the end of the event,
unless you want other groups to also be able to see your code during the event.

After the above, you are ready to proceed working on the two tasks of the event (see below)!

For both tasks, you will have access to plenty of pre-written code, and even pretrained models,
so that you already get a baseline to which to compare your own ideas.

In all cases, it's perfectly fine (and even encouraged!) to add or remove code, and to try completely different things.
**The only constraint is that your final test set predictions should follow the pre-determined format(s)**, as will be
described further down.

## Task 1: Training and evaluating ML models for synthetic cloud optical thickness (COT) data provided by SMHI
_Contact persons for help with Task 1: Thomas Ohlson Timoudas, Nosheen Abid, Aleksis Pirinen, György Kovacs_

The main files of importance are:

- `cot_synth_train.py`
- `cot_synth_eval.py`
- `final_cot_synth_eval.py`

The workflow is oriented around a **(1) model training and development phase**
(using the files `cot_synth_train.py` and `cot_synth_eval.py`), followed by a **(2) final test set evaluation phase** (using the file `final_cot_synth_eval.py`).

In **phase (1)**, the workflow is oriented around first training (and at the end of training, saving) models using `cot_synth_train.py`, followed by evaluting said
models using `cot_synth_eval.py`. Note that at this stage, evaluation is only possible on the train and val splits. Once satisfied with phase (1), proceed to **phase (2)**
using the file `final_cot_synth_eval.py`, as will be described further down. Note: if you are working in AWS, step 1-3 are already done. Please double-check that your folder tree structure is like the one described in step 3.

1. Begin by creating a folder `../data`, and in this folder you should put the data folder `SDL2_SMHI_data` that you can download from
https://drive.google.com/drive/folders/16VBNSgT-ngsoH_ZZsDbOPbwpSB100k-1?usp=drive_link (you must also unzip the file).

2. Then create a folder `../log_smhi` (this folder should be side-by-side with the folder `../data`; not one inside the other).
3. After these steps, you should have:

   ```
   SDL2_group<X>
   |
   └─── data
   |   | SDL2_SMHI_data.zip
   |   └─── SDL2_SMHI_data
   |
   └─── log_smhi
   ```

4. After the above, to train a model simply run
```
python cot_synth_train.py
```
Model weights will then be automatically saved in a log-folder, e.g. `../log_smhi/data-stamp-of-folder/model_it_xxx`. By default, `cot_synth_train.py` trains models on the training set,
where each input data point is assumed to be a 12-dimensional vector corresponding to the 12 spectral bands in the synthetic dataset (all of the 13 standard bands, except for B1), and where 3% noise
is added to the inputs during training (see the flag `INPUT_NOISE_TRAIN`).

**NOTE:** To train models which also omit band B10, set `SKIP_BAND_10` to True instead of False. In particular,
note that models that you want to use in Task 2 (see below) will have to be trained with `SKIP_BAND_10 = True`
because the data in that task is Level 2A instead of Level 1C. However, you may still want to use
`SKIP_BAND_10 = False` for this task (Task 1), since it may lead to better results for this task.
In summary, when you are ready to move on from Task 1 to Task 2, you may be interested in re-training
the best model(s) that you got for Task 1, but this time with `SKIP_BAND_10 = True`, so that they 
can also be tried out for Task 2.

Note that it is allowed to modify the code in `cot_synth_train.py` so that model training is performed on the union of the train and val splits (could be done e.g. prior to final test set evaluation).

4. To evaluate model(s), update the flag `MODEL_LOAD_PATH` in `cot_synth_eval.py` so that it points to the model checkpoint file induced by running the above training command.

5. After that, simply run
```
python cot_synth_eval.py
```
in order to evaluate the model that you trained using `cot_synth_train.py`. **The aim of this task is to get as low RMSE as possible.**

By default, evaluation occurs for the training split (can be changed with the flag `SPLIT_TO_USE`). Also, the flag `INPUT_NOISE` is set as the list `[0.00, 0.01, 0.02, 0.03, 0.04, 0.05]`.
In this case, the evaluation script will show average results across different input noise levels. In the final test set evaluation (see more below), models will be evaluated based on
the average RMSE across these six noise levels (0% to 5%), which corresponds to 120,000 examples (original test set size is 20,000, but 6 different noise-variants results in 120,000 in total).

It is possible to evaluate ensemble models as well. To do this for an ensmble of N models:

1. First train N models by running the `cot_synth_train.py` script N times (using different `SEED` each time
so that the models do not become identical).

2. Then, when running `cot_synth_eval.py`, ensure `MODEL_LOAD_PATH` becomes a list where each list element is a model-log path. Examples of this type of
path specification are already available within `cot_synth_eval.py`.

Pretrained model weights are already available here: https://drive.google.com/drive/folders/1MkqcoxLBb9C1vAUwvHipq5cr6Z7bXIel?usp=sharing. 
Note that if you are working on AWS you would need to run the following command to get the pretrained model weights:

```wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1oJXawadtmw0FCyjIQQXWL8123TP0TeSd' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1oJXawadtmw0FCyjIQQXWL8123TP0TeSd" -O temp3.zip && rm -rf /tmp/cookies.txt && unzip temp3.zip```

Download the content of this folder (a zip-file), unzip it,
and ensure the resulting 10 folders land inside `../log_smhi/`. The model weights that are available are 10 five-layer MLPs that were trained using 3% additive noise, and where
each input data point is assumed to be a 12-dimensional vector corresponding to the 12 spectral bands in the synthetic dataset (all of the 13 standard bands, except for B1). The model architectures
are identical; they were trained with different random network initializations. They can be run in ensemble-mode as described in the previous paragraph.

Finally, once satisfied with model training and development (phase 1), proceed to the final evaluation step (phase 2) using the file `final_cot_synth_eval.py`, as will be described now.
The syntax is similar to `cot_synth_eval.py`, and the participant only has to point `MODEL_LOAD_PATH` to the desired model / models. Then one runs
```
python final_cot_synth_eval.py
```
which will produce test set COT predictions in a file called `smhi_preds.npy` (in the same folder as `final_cot_synth_eval.py` resides). Note that the shape has to be `(120000,)`.
There will be no result shown, as the ground truth is withheld from the participants in this developer event. Instead, the file `smhi_preds.npy` should be shared with the organizers,
who will then use a withheld script to compare these COT predictions with the ground truth COTs on this test set, and calculate a final RMSE (lower is better).

**Note:** Running `final_cot_synth_eval.py` assumes a model whose input features are exactly the 12 spectral bands (all except B1) in the
default order (this order is ensured automatically by default in all scripts by default, so there is nothing to worry about). In case you are working with only a subset of those bands,
change the code accordingly.  

## Task 2: Cloudy / clear image classification use-case by Skogsstyrelsen (the Swedish Forest Agency (SFA))
_Contact persons for help with Task 2: Thomas Ohlson Timoudas, Nosheen Abid, Aleksis Pirinen, György Kovacs_

The main files of importance here are:

- `swe_forest_agency_cls.py`
- `eval_swe_forest_agency_cls.py`

In this task, you may want to try out the model(s) that you worked with in Task 1
(see above), and/or you may be interested in trying out completely different ideas and directions (such as training image classification models directly on this dataset). Feel free to explore ideas!
If you work with models from Task 1, you need to retrain similar models but where you omit band B10, because that band is not within this SFA dataset (since it's Level 2A instead of Level 1C data).

1. If you haven't already, begin by creating a folder `../data`, and in this folder you should create a folder `skogsstyrelsen`.

2. Within `../data/skogsstyrelsen/`, put the data that you can download from
https://drive.google.com/drive/folders/1lRCIcQo9CqFRDhUd3aZRAA46k8nLL49J?usp=sharing.
After the download, unzip the downloaded file (`skogsstyrelsen-data.zip`), and then move
all of the content within the resulting folder (called `skogsstyrelsen-data`) into
the folder `../data/skogsstyrelsen/`. Note that the folder `skogsstyrelsen-data` should
be empty after this step (and it may optionally be removed). Thus at the end of this step,
make sure to have the following folder structure:

   ```
   SDL2_group<X>
   |
   └─── data
   |   | SDL2_SMHI_data.zip
   |   | skogsstyrelsen_data.zip
   |   └─── SDL2_SMHI_data
   |   └─── skogsstyrelsen
   |
   └─── log_smhi
   ```

4. Then, also create a folder `../log_skogs` (i.e. the `data` and `log_skogs` folders should be next to each other; not one of them within the other). Thus at the end of this step, the
folder structure should look like this:

   ```
   SDL2_group<X>
   |
   └─── data
   |   | SDL2_SMHI_data.zip
   |   | skogsstyrelsen_data.zip
   |   └─── SDL2_SMHI_data
   |   └─── skogsstyrelsen
   |
   └─── log_smhi
   └─── log_skogs
   ```

6. To evaluate model(s) on the Skogsstyrelsen cloudy / clear image binary classification setup, first ensure that `MODEL_LOAD_PATH` points to model / models that have been trained on the synthetic
data by SMHI (see "Task 1" above, but note that the models must have 11-dimensional and not 12-dimensional inputs, since B10 is missing, as explained earlier), AND/OR first download pretrained models
as described below (these pretrained models expect 11-dimensional inputs, as desired). 

7. Then run
```
python swe_forest_agency_cls.py
```
The above will by default run the model on the whole train-val set in the provided train-val-test split. You can change what split to run the model on using the flag `SPLIT_TO_USE` in
`swe_forest_agency_cls.py`.

6. After this, a file `skogs_preds.npy` should have been created in the same folder where `swe_forest_agency_cls.py` resides.

7. Once this file exists, set `SPLIT_TO_USE` within `eval_swe_forest_agency_cls.py` to the same thing that it was when you just ran `swe_forest_agency_cls.py`.

8. Then run
```
python eval_swe_forest_agency_cls.py
```
in order to get various results. **The aim of this task is to get as high `F1 score (macro)` score as possible.**

Note that running the script `eval_swe_forest_agency_cls.py` script can only be done with the splits `train`, `val` and `trainval` (since the test set ground truth
is withheld from the participants). For the test set, it is instead the organizers who will run the script `eval_swe_forest_agency_cls.py` with the `skogs_preds.npy` files provided by the
participants, where the participants have produced their `skogs_preds.npy` files by running `python swe_forest_agency_cls.py` with `SPLIT_TO_USE = 'test'`.

Pretrained model weights are already available here: https://drive.google.com/drive/folders/14xTbLHPxaPznemG7ShE0DMC9zJsNU_hr?usp=sharing. Note that if you are working in AWS you would need to run this command instead:
```wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1j0Np2xTsQXWroUC-1T5io4Fj_XQpD0-d' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1j0Np2xTsQXWroUC-1T5io4Fj_XQpD0-d" -O temp4.zip && rm -rf /tmp/cookies.txt && unzip temp4.zip```

Download the folder, unzip it, and ensure the resulting
10 folders land inside `../log_skogs/`. The model weights that are available are 10 five-layer MLPs that were trained using 3% additive noise, and where each input data point is assumed to be an
11-dimensional vector corresponding to 11 out of the 12 spectral bands in the synthetic dataset (all of the 13 standard bands, except for B1 and B10). The model architectures
are identical; they were trained with different random network initializations. They can be run in ensemble-mode as described in the previous paragraph.

Prior to evaluating and submitting results on the test split, a suggestion is to tune the `THRESHOLD_THICKNESS_IS_CLOUD` (and `THRESHOLD_THICKNESS_IS_THIN_CLOUD`, but it's recommended to keep
these two the same in this binary setup) on the whole train-val set, in such a way that you get as high Macro F1-score as possible on the whole train-val set.
