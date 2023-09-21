# SDL2 developer event "Walking on Thin Clouds"
Please, **carefully follow this README when working on this developer event!** Prior to
working on any code:

1. **One** person within each group sends her/his github user name to the organizers. The organizers will then
add that person as a collaborator of this private repository.

2. That person shall then clone this git repository via the command `git clone https://github.com/aleksispi/sdl2-hackathon.git`.

3. Move into the git directory you cloned using the command `cd`.

4. Create a git branch for your group. If you are group X, then run the command
`git branch groupX` (if X is 4 in your case, it should say `git branch group4`).

5. Check the set of branches via the command `git branch`. You should now see the
two branches `groupX` and `master`.

6. Switch to your group's branch via `git checkout groupX`. **Do all development
within this branch** (_not_ in `master`).

7. Whenever you want, you can push code to your branch via `git push`. **Please
ensure that you push your final code at the end of the developer event!**
Note that the first push may require the command `git push --set-upstream origin groupX`.
Also note that you may not wish to git push the code prior to the end of the event,
unless you want other groups to also be able to see your code during the event.

After the above, you are ready to proceed working on the two tasks of the event (see below)!

For both tasks, you will have access to plenty of pre-written code, and even pretrained models,
so that you already get a baseline to which to compare your own ideas.

In all cases, it's perfectly fine (and even encouraged!) to add or remove code, and to try completely different things.
**The only constraint is that your final test set predictions should follow the pre-determined format(s)**, as will be described further down.

## Task 1: Training and evaluating ML models for synthetic cloud optical thickness (COT) data provided by SMHI
The main files of importance are:

- `cot_synth_train.py`
- `cot_synth_eval.py`
- `final_cot_synth_eval.py`

The workflow is oriented around a **(1) model training and development phase**
(using the files `cot_synth_train.py` and `cot_synth_eval.py`), followed by a **(2) final test set evaluation phase** (using the file `final_cot_synth_eval.py`).

In **phase (1)**, the workflow is oriented around first training (and at the end of training, saving) models using `cot_synth_train.py`, followed by evaluting said
models using `cot_synth_eval.py`. Note that at this stage, evaluation is only possible on the train and val splits. Once satisfied with phase (1), proceed to **phase (2)**
using the file `final_cot_synth_eval.py`, as will be described further down.

1. Begin by creating a folder `../data`, and in this folder you should put the data folder `SDL2_SMHI_data` that you can download from
https://drive.google.com/drive/folders/19I4FvvBzYC1Plo1psrq_KoMIURCJwrjr?usp=sharing (you must also unzip the file).

2. Then create a folder `../log_smhi` (this folder should be side-by-side with the folder `../data`; not one inside the other).

3. After the above, to train a model simply run
```
python cot_synth_train.py
```
Model weights will then be automatically saved in a log-folder, e.g. `../log_smhi/data-stamp-of-folder/model_it_xxx`. By default, `cot_synth_train.py` trains models on the training set,
where each input data point is assumed to be a 12-dimensional vector corresponding to the 12 spectral bands in the synthetic dataset (all of the 13 standard bands, except for B1), and where 3% noise
is added to the inputs during training (see the flag `INPUT_NOISE_TRAIN`).

To train models which also omit band B10 (e.g. to do evaluations on Skogsstyrelsen data; see "Task 2" below), set `SKIP_BAND_10` to True instead of False.

Note that it is allowed to modify the code in `cot_synth_train.py` so that model training is performed on the union of the train and val splits (could be done e.g. prior to final test set evaluation).

4. To evaluate model(s), update the flag `MODEL_LOAD_PATH` in `cot_synth_eval.py` so that it points to the model checkpoint file induced by running the above training command.

5. After that, simply run
```
python cot_synth_eval.py
```
in order to evaluate the model that you trained using `cot_synth_train.py`. 

By default, evaluation occurs for the training split (can be changed with the flag `SPLIT_TO_USE`). Also, the flag `INPUT_NOISE` is set as the list `[0.00, 0.01, 0.02, 0.03, 0.04, 0.05]`.
In this case, the evaluation script will show average results across different input noise levels. In the final test set evaluation (see more below), models will be evaluated based on
the average RMSE across these six noise levels (0% to 5%), which corresponds to 120,000 examples (original test set size is 20,000, but 6 different noise-variants results in 120,000 in total).

It is possible to evaluate ensemble models as well. To do this for an ensmble of N models:

1. First train N models by running the `cot_synth_train.py` script N times (using different `SEED` each time
so that the models do not become identical).

2. Then, when running `cot_synth_eval.py`, ensure `MODEL_LOAD_PATH` becomes a list where each list element is a model-log path. Examples of this type of
path specification are already available within `cot_synth_eval.py`.

Pretrained model weights are already available here: https://drive.google.com/drive/folders/1MkqcoxLBb9C1vAUwvHipq5cr6Z7bXIel?usp=sharing. Download the content of this folder (a zip-file), unzip it,
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
The main files of importance here are:

- `swe_forest_agency_cls.py`
- `eval_swe_forest_agency_cls.py`

In this task, you may want to try out the model(s) that you worked with in Task 1
(see above), and/or you may be interested in trying out completely different ideas and directions (such as training image classification models directly on this dataset). Feel free to explore ideas!
If you work with models from Task 1, you need to retrain similar models but where you omit band B10, because that band is not within this SFA dataset (since it's Level 2A instead of Level 1C data).

1. If you haven't already, begin by creating a folder `../data`, and in this folder you should create a folder `skogsstyrelsen`.

2. Within `../data/skogsstyrelsen/`, put the data that you can download from
https://drive.google.com/drive/folders/1lRCIcQo9CqFRDhUd3aZRAA46k8nLL49J?usp=sharing.

3. Then, also create a folder `../log_skogs` (i.e. the `data` and `log_skogs` folders should be next to each other; not one of them within the other).

4. To evaluate model(s) on the Skogsstyrelsen cloudy / clear image binary classification setup, first ensure that `MODEL_LOAD_PATH` points to model / models that have been trained on the synthetic
data by SMHI (see "Task 1" above), AND/OR first download pretrained models as described below. 

5. Then run
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

Pretrained model weights are already available here: https://drive.google.com/drive/folders/14xTbLHPxaPznemG7ShE0DMC9zJsNU_hr?usp=sharing. Download the folder, unzip it, and ensure the resulting
10 folders land inside `../log_skogs/`. The model weights that are available are 10 five-layer MLPs that were trained using 3% additive noise, and where each input data point is assumed to be an
11-dimensional vector corresponding to 11 out of the 12 spectral bands in the synthetic dataset (all of the 13 standard bands, except for B1 and B10). The model architectures
are identical; they were trained with different random network initializations. They can be run in ensemble-mode as described in the previous paragraph.

Prior to evaluating and submitting results on the test split, a suggestion is to tune the `THRESHOLD_THICKNESS_IS_CLOUD` (and `THRESHOLD_THICKNESS_IS_THIN_CLOUD`, but it's recommended to keep
these two the same in this binary setup) on the whole train-val set, in such a way that you get as high Macro F1-score as possible on the whole train-val set.

## If you want to set up your own Conda environment
On a Ubuntu work station, the below should be sufficient for running this developer event.
```
conda create -n hackathon python=3.8
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
pip install scipy
pip install matplotlib
pip install xarray
pip install scikit-image
pip install netCDF4
```
