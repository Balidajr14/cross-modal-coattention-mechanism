import os
from argparse import ArgumentParser, Namespace
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from .model.integrator import CoAttention_Model
from skimage.segmentation import find_boundaries


def main(args):
    print('HEY')
    model = CoAttention_Model.load_from_checkpoint(args.checkpoint_path)
    model.cuda()
    model.prepare_data()
    model.freeze()
    model.eval()
    #model.train_dataset.dataset.dataset.transform = model.val_dataset.dataset.transform
    test_loader = DataLoader(model.test_dataset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             shuffle=False)

    with torch.no_grad():
        feats = defaultdict(list)
        print('HEY')
        b = 0
        total_positives = 0
        for batch in test_loader:
            # batch_size = 16
            # image: 16 x 1 x 30 x 60 x 60
            # emr  : 16 x 20
            out = model.get_weights((batch.image, batch.emr))
            # out : 128 x 484 x 484 . The batch of the output seems to be multiplied by 8, the number of units (not heads) of the attention code.
            out = torch.mean(out, 1)
            # out : 128 x 484
            out = out.view(args.batch_size*model.hparams.num_heads, 4, 11, 11)
            # out : 128 x 4 x 11 x 11
            out = torch.nn.functional.interpolate(out, size=(60, 60), mode='bilinear')
            # out : 128 x 4 x 60 x 60
            out = out.permute(0, 2, 3, 1)
            # out : 128 x 60 x 60 x 4
            out = torch.nn.functional.interpolate(out, size=(60, 30), mode='bilinear')
            # out : 128 x 60 x 60 x 30
            out = out.permute(0, 3, 1, 2)
            # out : 128 x 30 x 60 x 60

            (img, x_lores), emr = (batch.image, batch.emr)
            # img : 16 x 1 x 60 x 60
            img = img.view(args.batch_size, 30, 60, 60).cpu()
            mask = batch.mask.view(args.batch_size, 30, 60, 60).cpu()
            out = out.view(model.hparams.num_heads, args.batch_size, 30, 60, 60).cpu()


            pred = model.forward((batch.image, batch.emr))
            output = {
                "pred_surv": pred["surv"],
            }
            pred_surv = torch.cat([output["pred_surv"]]).detach().cpu()
            two_year_bin = np.digitize(24, model.time_bins)
            survival_fn = model.survival_at_times(pred_surv, np.pad(model.time_bins, (1, 0)), model.eval_times)
            pred_binary = 1 - model.survival(pred_surv)[:, two_year_bin]

            true_binary = batch.target_binary
            print(true_binary)
            total_positives += np.sum(np.asarray(pred_binary))
            print(total_positives)

            threshold = 0.166742

            for h in range(model.hparams.num_heads):
                head = out[h]
                for i in range(args.batch_size):
                    image = img[i]
                    feature_map = head[i]
                    emr_data = emr[i]
                    mask_i = mask[i]
                    pred_bin = pred_binary[i]
                    true_bin = true_binary[i]

                    if pred_bin > threshold:
                        pred_bin = 1
                    else:
                        pred_bin = 0

                    nonzero_slices = np.where(mask_i.sum((1,2))>0)[0]
                    contours = mask_i[nonzero_slices[len(nonzero_slices)//2]]
                    contours = np.asarray(contours, dtype=np.uint8)
                    contours = find_boundaries(contours, mode='outer')
                    contours = np.ma.masked_where(contours == 0, contours)
                    plt.imshow(image[nonzero_slices[len(nonzero_slices)//2]], cmap='bone')
                    plt.imshow(contours, cmap='Greens', vmin=0., vmax=1.7)
                    if true_bin == pred_bin:
                        plt.savefig('attention_maps_test/correct/' + str(b) + '_' + str(i) + '_' + str(h) + '_bone.png' )
                    else:
                        plt.savefig('attention_maps_test/incorrect/' + str(b) + '_' + str(i) + '_' + str(h) + '_bone.png' )
                    plt.imshow(image[nonzero_slices[len(nonzero_slices)//2]], cmap='bone')
                    plt.imshow(feature_map[nonzero_slices[len(nonzero_slices)//2]], cmap='coolwarm', alpha=.4, vmin=0., vmax=1.)
                    if true_bin == pred_bin:
                        plt.savefig('attention_maps_test/correct/' + str(b) + '_' + str(i) + '_' + str(h) + '_heatmap.png' )
                    else:
                        plt.savefig('attention_maps_test/incorrect/' + str(b) + '_' + str(i) + '_' + str(h) + '_heapmap.png' )
                    # plt.savefig('attention_maps_2/' + str(b) + '_' + str(h) + '_' + str(i) + '_heatmap.png')
                    # plt.imshow(image[image.shape[0]//2], cmap='bone')
                    # plt.savefig('attention_maps_2/' + str(b) + '_' + str(h) + '_' + str(i) + '_bone.png' )
                    # plt.imshow(feature_map[feature_map.shape[0]//2], cmap='Spectral', alpha=.5)
                    # plt.colorbar()
                    # plt.savefig('attention_maps_2/' + str(b) + '_' + str(h) + '_' + str(i) + '_heatmap.png')
            b += 1

        print('TOTAL POSITIVES:', total_positives)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('checkpoint_path', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    main(args)