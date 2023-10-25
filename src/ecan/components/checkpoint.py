import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import argparse

parser = argparse.ArgumentParser(description="PyTorch")
parser.add_argument("--scale_factor", type=int, default=4, help="scale")
parser.add_argument("--test_dataset_dir", default='./data',
                    type=str, help="test_dataset dir")
parser.add_argument("--test_results_dir", default='./results',
                    type=str, help="test_results dir")
parser.add_argument("--model1", default='../logs/Extra/nFr7_batch64_K20L5M10_ecan_lab_cnn3_baseline/model4_epoch200.pth.tar', type=str, help="checkpoint")
parser.add_argument("--model2", default='../logs/Settings/nFr7_batch64_K20L5M10_ecan_lab_cnn3_no_dpw/model4_epoch200.pth.tar', type=str, help="checkpoint")
parser.add_argument("--model3", default='../logs/Extra/nFr7_batch64_K20L5M10_ecan_lab_dpw3_no_att/model4_epoch200.pth.tar', type=str, help="checkpoint")

parser.add_argument("--inType", type=str, default='y',
                    help="RGB input or y input")
parser.add_argument("--batchSize", type=int, default=1, help="Test batch size")
parser.add_argument("--gpu", type=int, default=1, help="used gpu")
parser.add_argument("--datasets", type=str, default=['vimeo', 'vid4', 'SPMC-11', ], help="datasets")
parser.add_argument('--seed', type=int, default=1, help='random seed to use. Default=0')
global opt, model
opt = parser.parse_args()
if not torch.cuda.is_available():
    raise Exception('No Gpu found, please run with gpu')
else:
    use_gpu = torch.cuda.is_available()
if use_gpu:
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(opt.seed)


def main():
    # model1 = torch.load(opt.model1)
    # model2 = torch.load(opt.model2)
    # model3 = torch.load(opt.model3)
    # NF3_psnr = model1['test_psnr']
    # NF5_psnr = model2['test_psnr']
    # NF7_psnr = model3['test_psnr']
    #
    # epoch = [i for i in range(len(NF3_psnr))]
    #
    # fig, ax = plt.subplots()
    #
    # ax.plot(epoch, NF3_psnr, linewidth=2.0, color='purple', label='3 Input Frames')
    # ax.plot(epoch, NF5_psnr, linewidth=2.0, color='orange',  label='5 Input Frames')
    # ax.plot(epoch, NF7_psnr, linewidth=2.0, color='red',  label='7 Input Frames')
    # ax.set(xlabel='Epoch', ylabel='PSNR')
    # ax.grid()
    # ax.legend()
    # plt.show()
    # plt.savefig("../logs/plots/length_graph")
    #
    model = torch.load(opt.model2)
    state_dict = model['state_dict']
    test_psnr = model['test_psnr']
    # valid_psnr = model['valid_psnr']
    loss = model['loss']
    epoch = [i for i in range(len(test_psnr))]

    print('state_dict', len(state_dict))  # plot
    fig, ax = plt.subplots()

    ax.plot(epoch, test_psnr, linewidth=2.0, color='tab:blue', label='Test')
    # ax.plot(epoch, valid_psnr, linewidth=2.0, color='tab:red',  label='Valid')
    ax.set(xlabel='Epoch', ylabel='PSNR', title='Training PSNR per Epoch', )
    ax.grid()
    ax.legend()
    plt.show()

    print('MAX TEST PSNR===>> ', max(test_psnr))
    # print('MAX VALID PSNR===>> ', max(valid_psnr))
    print(model['epoch'])
    for p in test_psnr:
        print('test_psnr', test_psnr.index(p), p)

    for p in valid_psnr:
        print('valid_psnr', valid_psnr.index(p), p)

if __name__ == '__main__':
    main()
    # print(torch.cuda.is_available())
    # if(torch.cuda.is_available()):
    #     print(torch.cuda.device_count())