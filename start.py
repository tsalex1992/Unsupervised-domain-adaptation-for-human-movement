import os


if __name__ == "__main__":
    os.system('python CycleGAN/train.py --dataroot ./datasets/horse2zebra --name m_horse2zebra_cyclegan --model cycle_gan --no_dropout')