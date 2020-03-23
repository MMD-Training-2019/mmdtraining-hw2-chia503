# -*- coding: utf-8 -*-


def main() -> None:
    """Main function."""
    parser = ArgumentParser()
    parser.add_argument('train', type=str, help="raw training data")
    parser.add_argument('test', type=str, help="raw testing data")
    parser.add_argument('pretrain', type=str, help="preprocessed training "
                                                   "feature")
    parser.add_argument('label', type=str, help="training label")
    parser.add_argument('pretest', type=str, help="preprocessed testing "
                                                  "feature")
    parser.add_argument('output', type=str, help="output path")
    parser.add_argument('--logistic', action='store_true', help="logistic mode")
    parser.add_argument('--generative', action='store_true', help="generative "
                                                                  "mode")
    args = parser.parse_args()

    # TODO
    # Use parameter as `args.train`, `args.test`, `args.pretrain` etc.
    # `args.logistic == True` if `--logistic` is passed.
    # `args.generative == True` if `--generative` is passed.


if __name__ == '__main__':
    main()
