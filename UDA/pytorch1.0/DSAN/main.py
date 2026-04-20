import torch
import torch.nn.functional as F
import math
import argparse
import numpy as np
import os
import json

from DSAN import DSAN
import data_loader


def save_checkpoint(model, checkpoint_path='model.pkl'):
    torch.save(model.state_dict(), checkpoint_path)


def load_checkpoint(model, checkpoint_path='model.pkl'):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
    except RuntimeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
        return model

    # Backward compatibility for legacy checkpoints serialized as whole models.
    # PyTorch 2.6+ defaults to weights_only=True, so explicit False is needed.
    return torch.load(checkpoint_path, weights_only=False)


def load_data(root_path, src_train, src_val, tar_train, tar_test, batch_size, num_workers=0, pin_memory=None):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
    loader_src_train = data_loader.load_training(root_path, src_train, batch_size, kwargs)
    loader_src_val = data_loader.load_testing(root_path, src_val, batch_size, kwargs)
    loader_tar_train = data_loader.load_training(root_path, tar_train, batch_size, kwargs)
    loader_tar_test = data_loader.load_testing(
        root_path, tar_test, batch_size, kwargs)
    return loader_src_train, loader_src_val, loader_tar_train, loader_tar_test


def train_epoch(epoch, model, dataloaders, optimizer):
    

    model.train()
    source_loader, _, target_train_loader, _ = dataloaders
    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len(source_loader)
    for i in range(1, num_iter):
        data_source, label_source = next(iter_source)
        data_target, _ = next(iter_target)
        if i % len(target_train_loader) == 0:
            iter_target = iter(target_train_loader)
        data_source, label_source = data_source.cuda(), label_source.cuda()
        data_target = data_target.cuda()

        optimizer.zero_grad()
        label_source_pred, loss_lmmd = model(
            data_source, data_target, label_source)
        loss_cls = F.nll_loss(F.log_softmax(
            label_source_pred, dim=1), label_source)
        lambd = 2 / (1 + math.exp(-10 * (epoch) / args.nepoch)) - 1
        loss = loss_cls + args.weight * lambd * loss_lmmd

        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print(f'Epoch: [{epoch:2d}], Loss: {loss.item():.4f}, cls_Loss: {loss_cls.item():.4f}, loss_lmmd: {loss_lmmd.item():.4f}')


def test(model, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.cuda(), target.cuda()
            pred = model.predict(data)
            # sum up batch loss
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()
            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(dataloader)
        print(
            f'Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(dataloader.dataset)} ({100. * correct / len(dataloader.dataset):.2f}%)')
    return correct.item()


def get_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='Root path for dataset',
                        default='/data/zhuyc/OFFICE31/')
    parser.add_argument('--src_train', type=str,
                        help='Source train domain', default='amazon')
    parser.add_argument('--src_val', type=str,
                        help='Source validation domain', default='amazon')
    parser.add_argument('--tar_train', type=str,
                        help='Target train domain', default='webcam')
    parser.add_argument('--tar_test', type=str,
                        help='Target test domain', default='webcam')
    parser.add_argument('--nclass', type=int,
                        help='Number of classes', default=31)
    parser.add_argument('--batch_size', type=int,
                        help='batch size', default=32)
    parser.add_argument('--nepoch', type=int,
                        help='Total epoch num', default=200)
    parser.add_argument('--lr', type=list, help='Learning rate', default=[0.001, 0.01, 0.01])
    parser.add_argument('--early_stop', type=int,
                        help='Early stoping number', default=30)
    parser.add_argument('--seed', type=int,
                        help='Seed', default=2021)
    parser.add_argument('--weight', type=float,
                        help='Weight for adaptation loss', default=0.5)
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    parser.add_argument('--decay', type=float,
                        help='L2 weight decay', default=5e-4)
    parser.add_argument('--bottleneck', type=str2bool,
                        nargs='?', const=True, default=True)
    parser.add_argument('--pretrained', type=str2bool,
                        nargs='?', const=True, default=True,
                        help='Use ImageNet pretrained ResNet-50 backbone')
    parser.add_argument('--smoke_test', type=str2bool,
                        nargs='?', const=True, default=False,
                        help='Run a single synthetic forward/backward pass and exit')
    parser.add_argument('--log_interval', type=int,
                        help='Log interval', default=10)
    parser.add_argument('--gpu', type=str,
                        help='GPU ID', default='0')
    parser.add_argument('--num_workers', type=int,
                        help='Number of dataloader worker processes (set 0 to avoid NFS multiprocessing cleanup issues)',
                        default=0)
    parser.add_argument('--result_json', type=str, default='',
                        help='Optional path to write machine-readable run metrics as JSON')
    args = parser.parse_args()
    return args


def run_smoke_test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running smoke test on {device}.')
    model = DSAN(num_classes=args.nclass, bottle_neck=args.bottleneck,
                 pretrained=False).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    data_source = torch.randn(args.batch_size, 3, 224, 224, device=device)
    data_target = torch.randn(args.batch_size, 3, 224, 224, device=device)
    label_source = torch.randint(
        0, args.nclass, (args.batch_size,), device=device)

    optimizer.zero_grad()
    label_source_pred, loss_lmmd = model(data_source, data_target, label_source)
    loss_cls = F.nll_loss(F.log_softmax(label_source_pred, dim=1), label_source)
    loss = loss_cls + args.weight * loss_lmmd
    loss.backward()
    optimizer.step()
    print(f'Smoke test passed. Loss: {loss.item():.4f}, cls: {loss_cls.item():.4f}, lmmd: {loss_lmmd.item():.4f}')


def get_checkpoint_path(args):
    checkpoint_dir = 'chekpoint'
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = f'source-{args.src_train}_target-{args.tar_train}_seed-{args.seed}_weight-{args.weight}.pkl'
    return os.path.join(checkpoint_dir, filename)


if __name__ == '__main__':
    args = get_args()
    print(vars(args))
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.smoke_test:
        run_smoke_test(args)
        raise SystemExit(0)

    dataloaders = load_data(args.root_path, args.src_train,
                            args.src_val, args.tar_train, args.tar_test,
                            args.batch_size, args.num_workers)
    model = DSAN(num_classes=args.nclass, bottle_neck=args.bottleneck,
                 pretrained=args.pretrained).cuda()
    checkpoint_path = get_checkpoint_path(args)
    
    best_source_val_correct = 0
    best_source_val_acc = 0.0
    stop = 0

    if args.bottleneck:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.bottle.parameters(), 'lr': args.lr[1]},
            {'params': model.cls_fc.parameters(), 'lr': args.lr[2]},
        ], lr=args.lr[0], momentum=args.momentum, weight_decay=args.decay)
    else:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.cls_fc.parameters(), 'lr': args.lr[1]},
        ], lr=args.lr[0], momentum=args.momentum, weight_decay=args.decay)

    for epoch in range(1, args.nepoch + 1):
        stop += 1
        for index, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = args.lr[index] / math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75)

        train_epoch(epoch, model, dataloaders, optimizer)
        source_val_correct = test(model, dataloaders[1])
        if source_val_correct > best_source_val_correct:
            best_source_val_correct = source_val_correct
            best_source_val_acc = 100. * best_source_val_correct / \
                len(dataloaders[1].dataset)
            stop = 0
            save_checkpoint(model, checkpoint_path)
        print(
            f'{args.src_train}-{args.tar_train}: max source_val correct: {best_source_val_correct} max source_val accuracy: {best_source_val_acc:.2f}%\n')

        if stop >= args.early_stop:
            break

    best_model = load_checkpoint(model, checkpoint_path)
    tar_test_correct = test(best_model, dataloaders[-1])
    tar_test_acc = 100. * tar_test_correct / len(dataloaders[-1].dataset)
    print(f'Best source_val acc: {best_source_val_acc:.2f}%')
    print(f'Final tar_test acc: {tar_test_acc:.2f}%')

    if args.result_json:
        result = {
            'src_train': args.src_train,
            'src_val': args.src_val,
            'tar_train': args.tar_train,
            'tar_test': args.tar_test,
            'seed': args.seed,
            'weight': args.weight,
            'best_source_val_acc': best_source_val_acc,
            'target_test_acc': tar_test_acc,
            'best_checkpoint': checkpoint_path,
        }
        result_dir = os.path.dirname(args.result_json)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
        with open(args.result_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
