
**Note**: The current software works well with PyTorch 0.1-0.3. PyTorch 0.4 support will be added by the end of May.


## Prerequisites
- Linux or macOS
- Python 2 or 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
- Install Torch vision from the source.
```bash
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```
- Install python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate).
```bash
pip install visdom
pip install dominate
```
- Clone this repo:
```bash
git clone https://github.com/tsalex1992/Unsupervised-domain-adaptation-for-human-movement
cd Unsupervised-domain-adaptation-for-human-movement/CycleGAN
```

### SkeletonGAN train/test

- Train a model:
```bash
python train.py --dataroot ./datasets/<dataset_folder> --name <dataset_name> --no_dropout
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097. To see more intermediate results, check out `./checkpoints/<dataset_name>/web/index.html`
- Test the model:
```bash
python test.py --dataroot ./datasets/<dataset_folder> --name <dataset_name> --phase test --no_dropout
```
The test results will be saved to a html file here: `./results/<dataset_name>/latest_test/index.html`.


## Training/test Details
useful flag to control the skeleton loss factor is --lambda_S used in training.
- Flags: see `options/train_options.py` and `options/base_options.py` for all the training flags; see `options/test_options.py` and `options/base_options.py` for all the test flags.
- CPU/GPU (default `--gpu_ids 0`): set`--gpu_ids -1` to use CPU mode; set `--gpu_ids 0,1,2` for multi-GPU mode. You need a large batch size (e.g. `--batchSize 32`) to benefit from multiple GPUs.  
- Visualization: during training, the current results can be viewed using two methods. First, if you set `--display_id` > 0, the results and loss plot will appear on a local graphics web server launched by [visdom](https://github.com/facebookresearch/visdom). To do this, you should have `visdom` installed and a server running by the command `python -m visdom.server`. The default server URL is `http://localhost:8097`. `display_id` corresponds to the window ID that is displayed on the `visdom` server. The `visdom` display functionality is turned on by default. To avoid the extra overhead of communicating with `visdom` set `--display_id 0`. Second, the intermediate results are saved to `[opt.checkpoints_dir]/[opt.name]/web/` as an HTML file. To avoid this, set `--no_html`.
- Preprocessing: images can be resized and cropped in different ways using `--resize_or_crop` option. The default option `'resize_and_crop'` resizes the image to be of size `(opt.loadSize, opt.loadSize)` and does a random crop of size `(opt.fineSize, opt.fineSize)`. `'crop'` skips the resizing step and only performs random cropping. `'scale_width'` resizes the image to have width `opt.fineSize` while keeping the aspect ratio. `'scale_width_and_crop'` first resizes the image to have width `opt.loadSize` and then does random cropping of size `(opt.fineSize, opt.fineSize)`.
- Fine-tuning/Resume training: to fine-tune a pre-trained model, or resume the previous training, use the `--continue_train` flag. The program will then load the model based on `which_epoch`. By default, the program will initialize the epoch count as 1. Set `--epoch_count <int>` to specify a different starting epoch count.
- For Conda users, we include a script `./scripts/conda_deps.sh` to install PyTorch and other libraries.



## Citation
```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}

```


## Acknowledgments
Code is based on [CycleGAN](https://github.com/junyanz/CycleGAN): Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks  
