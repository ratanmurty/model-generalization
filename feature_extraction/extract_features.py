from feature_extractor import *
import argparse
import os

os.environ['CM_HOME'] = '/mindhive/nklab4/users/ighodgao/checkpoints'
os.environ['MT_HOME'] = '/mindhive/nklab4/users/ighodgao/checkpoints'
os.environ['HOME'] = '/mindhive/nklab4/users/ighodgao'

os.environ['CM_TSLIM_WEIGHTS_DIR'] = '/mindhive/nklab4/users/ighodgao/checkpoints'
os.environ['MT_IMAGENET_PATH'] = '/mindhive/nklab4/users/ighodgao'
os.environ['RESULTCACHING_HOME'] = '/mindhive/nklab4/users/ighodgao'

parser = argparse.ArgumentParser()
parser.add_argument('--images_path', type=str,
                    help='path to grab images from')
parser.add_argument('--save_dir', type=str,
                    help='where to store the feature maps')
parser.add_argument('--model', type=str, help='what models to extract features from')
parser.add_argument('--index', type=int, help='what model index to extract if given a list')
args = parser.parse_args()
model = args.model
index = args.index
images_path = args.images_path

fe = FeatureExtractor()

if model == 'base-models':
        model_ids = fe.base_models.keys()
        fe.extract_features(images_path=images_path, cnn_model=list(model_ids)[index], save_dir=args.save_dir)
elif model == 'cornets':
        model_ids = fe.cornet_models.keys()
        fe.extract_features(images_path=images_path, cnn_model=list(model_ids)[index], save_dir=args.save_dir)
elif model == 'pytorch':
        model_ids = fe.pytorch_models.keys()
        fe.extract_features(images_path=images_path, cnn_model=list(model_ids)[index], save_dir=args.save_dir, use_pytorch=True)
else:
	fe.extract_features(images_path=images_path, cnn_model=model, save_dir=args.save_dir)




