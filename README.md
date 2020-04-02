# SAL: Sign Agnostic Learning of Shapes From Raw Data
<p align="center">
  <img src="teaser2.png"/>
</p>

This repository contains an implementation to the CVPR 2020 paper SAL: Sign Agnostic Learning of Shapes From Raw Data.

SAL is a deep learning approach for learning implicit shape representations directly from raw, unsigned geometric data, such as point clouds and triangle soups.

The teaser above depicts an example where collectively learning a dataset of raw human scans using SAL overcomes many imperfections and artifacts in the data (left in every gray pair) and provides high quality surface reconstructions (right in every gray pair) and shape space (interpolations of latent representations are in gold).

For more details visit: https://arxiv.org/abs/1911.10414.

### Installation Requirmenets
The code is compatible with python 3.7 + pytorch 1.4. In addition, the following packages are required:  
pyhocon, plotly, scikit-image, trimesh, GPUtil, tqdm, CGAL.

### Usage
#### Learning shape space from the D-Faust dataset raw scans

##### Data
The raw scans can be downloaded from http://dfaust.is.tue.mpg.de/downloads.
In order to be able to run the training process, the raw scans need to be preprocessed using:

```
cd ./code
python preprocess/preprocess_dfaust.py 
```

##### Predicting Meshed surfaces with SAL trained network
```
cd ./code
python evaluate/evaluate.py --checkpoint 2000 --parallel --exp_name dfaust --conf ./confs/dfaust.conf --split ./confs/splits/dfaust/test_all_every5.json --exps_dir trained_models
```

##### Training
To train, run:
```
cd ./code
python training/exp_runner.py
```
##### Evaluation
To produce meshed surfaces of the learned implicit representations via the Marching Cubes algorithm, run:
```
cd ./code
python evaluate/evaluate.py
```

Notice that it also possible to compute the chamfer distance to registrations and input scan using the --compute_dist_to_gt flag.

#### Surface reconstruction
SAL can be used to reconstruct a single raw 3D data such as a point cloud or a triangle soup. Update the file ./confs/recon.conf to point the path of your inpur raw 3D data:
```
train
{
  ...
  dataset_path = your_path
  ...
}
```
Then, run training:
```
cd ./code
python evaluate/evaluate.py --batch_size 1 --conf ./confs/recon.conf --workers 1 
```
Finally, to produce the meshed surface, run:
```
cd ./code
python evaluate/evaluate.py --exp_name recon --conf ./confs/recon.conf --split none
```

### Citation
If you find our work useful in your research, please consider citing:

        @article{atzmon2019sal,
        title={Sal: Sign agnostic learning of shapes from raw data},
        author={Atzmon, Matan and Lipman, Yaron},
        journal={arXiv preprint arXiv:1911.10414},
        year={2019}
        }
	
