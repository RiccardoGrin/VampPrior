import argparse
from utils import *

parser = argparse.ArgumentParser(description='PyTorch VAE Training')
parser.add_argument('--data', default='pokemon', help='Dataset name')
parser.add_argument('--epochs', '-e', default=5000, type=int, help='Total epochs to run (default: 5000)')
parser.add_argument('--batch_size', '-bs', default=256, type=int, help='Mini-batch size (default: 256)')
parser.add_argument('--learn_rate', '-lr', default=1e-3, type=int, help='Learning rate (default: 1e-3)')
parser.add_argument('--label', '-l', default='VAE', help='Experiment name', type=str)
parser.add_argument('--checkpoint', '-cp', default=None, type=str, help='Checkpoint name')


def train(model, optimizer, scheduler, dataloader, epoch, label, losses, bces, kls, max_epochs):
    step = 0
    for _ in range(max_epochs):
        for images in dataloader:
            optimizer.zero_grad()
            
            image_in = images.permute(0,3,1,2)
            x_in = Variable(image_in.float().cuda())
            
            loss, bce, kl = model.calculate_loss(x_in)
            
            loss.backward()
            scheduler.step()
            optimizer.step()
            losses.append(loss.item())
            bces.append(bce.item())
            kls.append(kl.item())
            
            step += 1
        epoch += 1
        
        clear_output(wait=True)
        print("Epoch:", epoch, '- Loss: {:3f}'.format(loss.item()))
        multi_plot(images, model)
        
        if epoch%10 == 0:
            save_file = "checkpoints/" + label + "_epoch_{:06d}".format(epoch) + '.pth'
            if not os.path.isfile(save_file):
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'losses' : losses,
                    'bces' : bces,
                    'kls' : kls,
                    'cs' : step
                }, save_file)
                print("Saved checkpoint")
            data_train(model, "/home/ubuntu/VAE/Pokemon/charizard.jpg", epoch)
    return losses, bces, kls


def main():
    args = parser.parse_args()

    net = Net()
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)
        net = net.cuda()
        
    try:
        net, epoch, losses, bces, kls, optimizer, scheduler = load_checkpoint("./checkpoints/" + args.checkpoint, net, args.learn_rate)
    except:
        epoch = 0
        losses = []
        bces = []
        kls = []
        optimizer = optim.Adam(net.parameters(), lr=args.learn_rate, amsgrad=True)
        scheduler = SGDRScheduler(optimizer, min_lr=1e-5, max_lr=args.learn_rate, cycle_length=500, current_step=0)
        print("Starting new training")

    gen_data_list()
    multiSet = MultiSet(args.data)
    dataloader = Utils.DataLoader(dataset=multiSet, shuffle=True, batch_size=args.batch_size)
    
    train_losses, bces, kls = train(net, optimizer, scheduler, dataloader, epoch, args.label, losses, bces, kls, args.epochs)
    generate_animation("data/", args.label)
    print("Training completed!")

    
if __name__ == '__main__':
    main()