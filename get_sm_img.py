import argparse
import sagemaker


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', type=str)
    parser.add_argument('--framework', type=str)
    args = parser.parse_args()
    

    image_uri = sagemaker.image_uris.retrieve(framework=args.framework
    ,region=args.region,version='0.23-1',
    image_scope='inference')

    print(image_uri)
