import numpy as np
import argparse

import torch
from models import cls_model
from utils import create_dir, viz_cls
import os
from tqdm import tqdm
from pytorch3d.transforms import Rotate, euler_angles_to_matrix
from concat import concat_media_horizontally

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')
    parser.add_argument('--rotate', type=int, default=0)

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model(num_classes=args.num_cls_class).to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy(np.load(args.test_label))

    # ------ TO DO: Make Prediction ------
    R = euler_angles_to_matrix(torch.tensor([0., 0., np.radians(args.rotate)]), "XYZ")
    TF = Rotate(R)
    test_data = TF.transform_points(test_data)

    pred_label = []
    for data in tqdm(test_data, desc="Evaluating"):
        data = data.unsqueeze(0)
        pred = model(data.to(args.device))
        pred_label.append(pred.argmax(dim=1).cpu())
    
    pred_label = torch.tensor(pred_label)

    # Compute Accuracy
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    print ("test accuracy: {}".format(test_accuracy))

    # num_correct = {0,1,2}
    # num_wrong = {0,1,2}
    paths = []
    ids = [0, 617, 719]
    output_dir = os.path.join(args.output_dir, f"{args.exp_name}_cls_{args.rotate}_{args.num_points}_{test_accuracy:.4f}")
    create_dir(output_dir)

    for i in range(test_label.size()[0]):
        # if pred_label[i].item() == test_label[i].item() and test_label[i].item() in num_correct:
        #     viz_cls(test_data[i], os.path.join(output_dir, 'obj_{}_gt_{}_pred_{}.gif'.format(i, int(test_label[i].item()), pred_label[i].item())), args.device)
        #     num_correct.remove(test_label[i].item())
        # elif pred_label[i].item() != test_label[i].item() and test_label[i].item() in num_wrong:
        #     viz_cls(test_data[i], os.path.join(output_dir, 'obj_{}_gt_{}_pred_{}.gif'.format(i, int(test_label[i].item()), pred_label[i].item())), args.device)
        #     num_wrong.remove(test_label[i].item())
        # if len(num_correct) == 0 and len(num_wrong) == 0:
        #     break
        if i in ids:
            path = os.path.join(output_dir, 'obj_{}_gt_{}_pred_{}.gif'.format(i, int(test_label[i].item()), pred_label[i].item()))
            viz_cls(test_data[i], path, args.device)
            paths.append(path)

    concat_media_horizontally(paths, output_path=os.path.join(output_dir, 'out.gif'), captions=['GT: {}'.format(int(test_label[i].item())) + ', Pred: {}'.format(int(pred_label[i].item())) for i in ids])

