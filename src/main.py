import argparse
from data_loader import DataLoader
from model import MyModel
from model_helper import ModelHelper, FilePaths


def main():
    # optional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train the NN', action='store_true')
    parser.add_argument('--validate', help='validate the NN', action='store_true')

    args = parser.parse_args()

    model_helper = ModelHelper()
    if args.train or args.validate:
        loader = DataLoader(FilePaths.fnTrain, MyModel.batch_size, (MyModel.img_width, MyModel.img_height),
                            MyModel.max_text_len)

        if args.train:
            model_helper.train(loader)
        elif args.validate:
            model_helper.validate(loader)
    else:
        model_helper.infer()


if __name__ == '__main__':
    main()
