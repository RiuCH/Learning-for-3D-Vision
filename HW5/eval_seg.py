import numpy as np
import argparse
import os
import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg
from tqdm import tqdm
from pytorch3d.transforms import Rotate, euler_angles_to_matrix
from concat import concat_media_horizontally


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')
    parser.add_argument('--rotate', type=int, default=0)

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')


    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model(num_seg_classes=args.num_seg_class).to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy((np.load(args.test_label))[:,ind])

    # ------ TO DO: Make Prediction ------
    R = euler_angles_to_matrix(torch.tensor([0., 0., np.radians(args.rotate)]), "XYZ")
    TF = Rotate(R)
    test_data = TF.transform_points(test_data)

    pred_label = []
    for data in tqdm(test_data, desc="Evaluating"):
        data = data.unsqueeze(0)
        pred = model(data.to(args.device))
        pred_label.append(pred.argmax(dim=2).cpu())
    pred_label = torch.cat(pred_label, dim=0)
 
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
    print ("test accuracy: {}".format(test_accuracy))

    # num_good = 3
    # num_bad = 3
    # thres = 0.7
    paths = []
    ids = [0, 1, 2]
    output_dir = os.path.join(args.output_dir, f"{args.exp_name}_seg_{args.rotate}_{args.num_points}_{test_accuracy:.4f}")
    create_dir(output_dir)

    for i in range(test_label.size()[0]):
        acc = pred_label[i].eq(test_label[i].data).cpu().sum().item() / test_label[i].size()[0]
        # if num_good > 0 and acc >= thres:
        #     viz_seg(test_data[i], test_label[i], os.path.join(output_dir, "obj_{}_gt_{}.gif".format(i, acc)), args.device)
        #     viz_seg(test_data[i], pred_label[i], os.path.join(output_dir, "obj_{}_pred_{}.gif".format(i, acc)), args.device)
        #     num_good -= 1
        # elif num_bad > 0 and acc < thres:
        #     viz_seg(test_data[i], test_label[i], os.path.join(output_dir, "obj_{}_gt_{}.gif".format(i, acc)), args.device)
        #     viz_seg(test_data[i], pred_label[i], os.path.join(output_dir, "obj_{}_pred_{}.gif".format(i, acc)), args.device)
        #     num_bad -= 1
        # if num_good == 0 and num_bad == 0:
        #     break
        if i in ids:
            path = os.path.join(output_dir, 'obj_{}.gif'.format(i))
            viz_seg(test_data[i], pred_label[i], path, args.device)
            paths.append(path)

    concat_media_horizontally(paths, output_path=os.path.join(output_dir, 'out.gif'), captions=['Accuracy: {:.2f} %'.format((pred_label[i].eq(test_label[i].data).cpu().sum().item() / test_label[i].size()[0])*100) for i in ids])