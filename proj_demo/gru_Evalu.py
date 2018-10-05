#!/usr/bin/env python

from __future__ import print_function
from __future__ import division

import os
import numpy as np
from optparse import OptionParser

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

#import memodels as models
import gru_models as models
from dataset import VideoFeatDataset as dset
from tools.config_tools import Config
from tools import utils



parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="evaluation configuration",
                  default="./configs/test_config.yaml")

(opts, args) = parser.parse_args()
assert isinstance(opts, object)
opt = Config(opts.config)
print(opt)

if opt.checkpoint_folder is None:
    opt.checkpoint_folder = 'checkpoints'

test_video_dataset = dset(opt.data_dir, opt.video_flist, which_feat='vfeat')
test_audio_dataset = dset(opt.data_dir, opt.audio_flist, which_feat='afeat')

print('number of test samples is: {0}'.format(len(test_video_dataset)))
print('finished loading data')


if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with \"cuda: True\"")
else:
    if int(opt.ngpu) == 1:
        print('so we use gpu 1 for testing')
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
        cudnn.benchmark = True
        print('setting gpu on gpuid {0}'.format(opt.gpu_id))

# test function for metric learning
def test(video_loader, audio_loader, model, opt):
    """
    train for one epoch on the training set
    """
    # training mode
    model.eval()
    simi_mat = np.zeros((30,30))
    vcount = 0
    for _, vfeat in enumerate(video_loader):
        acount = 0
        sizev = vfeat.size()[0]
        for _, afeat in enumerate(audio_loader):
            sizea = afeat.size()[0]
            for k in range(sizea):
                afeat_cpy = afeat.numpy().copy()
                a_1 = afeat_cpy[-k+sizea:sizea]
                a_2 = afeat_cpy[0:sizea-k]
                afeat_cpy = np.concatenate([a_1,a_2])
                afeat_cpy = torch.from_numpy(afeat_cpy)
                vfeat_var = Variable(vfeat)
                afeat_var = Variable(afeat_cpy)
                if opt.cuda:
                    vfeat_var = vfeat_var.cuda()
                    afeat_var = afeat_var.cuda()
                similarity = model(vfeat_var,afeat_var)
                sim = similarity[:,0].data.cpu().numpy()
                for irange in range(len(sim)):
                    simi_mat[vcount+(k+irange)%sizea,acount+irange]=sim[irange]
                # simi_mat[it+k,jt:jt+sizeJ] = sim
            acount += sizev
        vcount +=sizea
    cnt = 0

    simi_cpy = simi_mat.copy()
    sortNo = np.argsort(-simi_cpy)
    for i in range(np.shape(simi_mat)[0]):
        s_view = sortNo[i][:opt.topk]
        print('most similar for ',i,' is among',s_view)
        if i in s_view:
            cnt += 1
    print('Testing accuracy (top{}): {:.3f}'.format(opt.topk, cnt / np.shape(simi_mat)[0]))

def main():
    global opt
    # test data loader
    test_video_loader = torch.utils.data.DataLoader(test_video_dataset, batch_size=opt.batchSize,
                                     shuffle=False, num_workers=int(opt.workers))
    test_audio_loader = torch.utils.data.DataLoader(test_audio_dataset, batch_size=opt.batchSize,
                                     shuffle=False, num_workers=int(opt.workers))

    # create model
    model = models.GRU_net()

    if opt.init_model != '':
        print('loading pretrained model from {0}'.format(opt.init_model))
        model.load_state_dict(torch.load(opt.init_model))

    if opt.cuda:
        print('shift model to GPU .. ')
        model = model.cuda()

    test(test_video_loader, test_audio_loader, model, opt)


if __name__ == '__main__':
    main()
