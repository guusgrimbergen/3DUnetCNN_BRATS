import os
import glob
import argparse

from unet3d.generator import get_training_and_validation_generators
from unet3d.data import write_data_to_file, open_data_file
from unet3d.utils.print_utils import print_processing, print_section, print_separator

cwd = os.path.realpath(__file__)
parent_dir = os.path.abspath(os.path.join(cwd, os.pardir))
os.chdir(parent_dir)

config = dict()


def fetch_training_data_files():
    training_data_files = list()
    for subject_dir in glob.glob(os.path.join(os.path.dirname(__file__), "data_train", "preprocessed", "*", "*")):
        subject_files = list()
        for modality in config["training_modalities"] + ["truth"]:
            subject_files.append(os.path.join(
                subject_dir, modality + ".nii.gz"))
        training_data_files.append(tuple(subject_files))
    return training_data_files


def main(overwrite=False):

    parser = argparse.ArgumentParser(description='Data preparation')
    parser.add_argument('-c', '--challenge', type=str,
                        choices=[2018], default=2018,
                        help="year of brats challenge")
    parser.add_argument('-d', '--dataset', type=str,
                        choices=["original", "preprocessed",
                                 "denoised_preprocessed", "denoised_original"],
                        default="preprocessed",
                        help="dataset type")
    parser.add_argument('-i', '--inms', type=bool,
                        default=True,
                        help="is normalize mean, standard deviation")

    args = parser.parse_args()
    dataset = args.dataset
    is_normalize_mean_std = args.inms
    challenge = args.challenge

    if is_normalize_mean_std:
        dataset = "brats{}_{}_normalize_mean_std".format(challenge, dataset)
    else:
        dataset = "brats{}_{}_normalize_minh".format(challenge, dataset)

    save_to_dir = os.path.join(parent_dir, "database", dataset)

    # make dir
    if not os.path.exists(save_to_dir):
        print_separator()
        print("making dir", save_to_dir)
        os.makedirs(save_to_dir)

    print_section("convert input images into an hdf5 file")

    data_file_path = os.path.join(save_to_dir, "brats_data.h5")
    if overwrite or not os.path.exists(data_file_path):
        training_files = fetch_training_data_files()
        write_data_to_file(training_files, data_file_path,
                           image_shape=(240, 240, 155))

if __name__ == "__main__":
    main(False)