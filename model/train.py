import torch
from tqdm import tqdm
from torch.autograd import Variable
from model.util import save_checkpoint, ReconstructionLoss, DiscriminatorLoss, \
    GeneratorLoss, CrossEntropyLabelSmooth


def train_model(args, model, train_dl, lr, epochs, classify_dim=1, save_path=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = ReconstructionLoss(args)
    criterion_cty = CrossEntropyLabelSmooth(num_classes=classify_dim).cuda()
    criterion_discriminator = DiscriminatorLoss(args)
    criterion_generator = GeneratorLoss(args)

    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        model = model.train()
        for i, batch_sample in enumerate(train_dl):
            optimizer.zero_grad()
            # lode data
            x = batch_sample['data']
            x = Variable(x)
            x = torch.reshape(x, (x.size(0), -1))
            train_label = batch_sample['label']
            train_label = Variable(train_label)
            x_prime, h, x_r, x_d, x_g, x_cluster, mu, var = model(x.to(device))
            # loss function
            loss1 = criterion(x_prime, x.to(device))  # reconstructionLoss loss
            loss2 = criterion_cty(x_cluster, train_label.to(device))  # classification loss
            loss3 = criterion_discriminator(x_d, x_r)  # discriminator loss
            loss4 = criterion_generator(x_g)  # generator loss
            loss = loss1 + 0.1 * loss2 + loss3 + loss4  # total loss
            # optimize
            loss.backward()
            optimizer.step()
        if epoch == epochs:
            save_checkpoint({'epoch': epoch,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             }, save_path)
    return model
