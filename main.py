import torch
import torch.nn as nn
import torch.utils
import torch.distributions as D
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.model_selection import train_test_split
from models.vae.vae import VariationalAutoencoder
from models.gan.label_generator import Generator
from models.gan.discriminator import Discriminator
from models.gan.wasserstein_distance import SinkhornDistance
from models.estimator.bnn import BNN
from itertools import chain
import pandas as pd
from config import Config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = Config.config


class GeneratorDataset(Dataset):
    def __init__(self, t, x, y):
        self.reg_term = np.concatenate((t, x, y), axis=1)

    def __getitem__(self, item):
        return self.reg_term[item]

    def __len__(self):
        return len(self.reg_term)


class EstimatorDataset(Dataset):
    def __init__(self, t, x, y):
        self.t = t
        self.x = x
        self.y = y

    def __getitem__(self, item):
        return self.t[item], self.x[item], self.y[item]

    def __len__(self):
        return len(self.t)


def load_data(file_path):
    data = pd.read_csv(file_path)
    t = np.array(data['X1']).reshape((-1, 1))
    s = np.array(data['S1']).reshape((-1, 1))
    y = np.array(data['Y']).reshape((-1, 1))
    x = np.array(data[['X2_' + str(i) for i in range(1, Config.confounds_num + 1)]])
    gt = np.array(data['GT']).reshape((-1, 1))
    t_train, t_test, x_train, x_test, y_train, y_test, s_train, s_test, gt_train, gt_test = train_test_split(
        t, x, y, s, gt, test_size=0.40)
    t_test, t_val, x_test, x_val, y_test, y_val, s_test, s_val, gt_test, gt_val = train_test_split(
        t_test, x_test, y_test, s_test, gt_test, test_size=0.50)
    return t_train, t_test, t_val, x_train, x_test, x_val, y_train, y_test, y_val, s_train, s_test, s_val, gt_train, gt_test, gt_val


def build_generator_dataset(t_train, t_test, t_val, x_train, x_test, x_val, y_train, y_test, y_val):
    generator_train_dataset = GeneratorDataset(t_train, x_train, y_train)
    generator_test_dataset = GeneratorDataset(t_test, x_test, y_test)
    generator_val_dataset = GeneratorDataset(t_val, x_val, y_val)
    return generator_train_dataset, generator_test_dataset, generator_val_dataset


def build_estimator_dataset(t_train, t_test, t_val, x_train, x_test, x_val, y_train, y_test, y_val):
    estimator_train_dataset = EstimatorDataset(t_train, x_train, y_train)
    estimator_test_dataset = EstimatorDataset(t_test, x_test, y_test)
    estimator_val_dataset = EstimatorDataset(t_val, x_val, y_val)
    return estimator_train_dataset, estimator_test_dataset, estimator_val_dataset


def build_cf_dataset(t_train, t_test, t_val, x_train, x_test, x_val, y_train, y_test, y_val):
    t_train_cf = np.where(t_train == 1, 0, 1).reshape(-1, 1)
    t_test_cf = np.where(t_test == 1, 0, 1).reshape(-1, 1)
    t_val_cf = np.where(t_val == 1, 0, 1).reshape(-1, 1)
    estimator_train_dataset = EstimatorDataset(t_train_cf, x_train, y_train)
    estimator_test_dataset = EstimatorDataset(t_test_cf, x_test, y_test)
    estimator_val_dataset = EstimatorDataset(t_val_cf, x_val, y_val)
    return estimator_train_dataset, estimator_test_dataset, estimator_val_dataset


def train_vae(autoencoder, dataloader, bin_feats):
    epochs = config.get('vae_epochs')
    lr = config.get('encoder_lr')
    wd = config.get('encoder_wd')
    opt = torch.optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=wd)
    mse_func = nn.MSELoss(reduction='sum')
    autoencoder.train()
    writer = SummaryWriter()
    for epoch in range(epochs):
        loss_sum = 0
        for t_x_y in dataloader:
            t_x_y = t_x_y.to(device)
            opt.zero_grad()
            x_hat = autoencoder(t_x_y, bin_feats)
            loss = mse_func(t_x_y.float(), x_hat) + autoencoder.encoder.kl
            loss_sum += loss
            loss.backward()
            opt.step()
        writer.add_scalar('VAE loss', loss_sum, epoch)
    writer.close()
    return autoencoder


def train_gan(label_generator, sample_generator, label_discriminator, sample_discriminator, obs_dataloader,
              rep_dataloader, bin_feats):
    epochs = config.get('gan_epochs')
    gen_lr = config.get('generator_lr')
    gen_wd = config.get('generator_wd')
    dis_lr = config.get('discriminator_lr')
    dis_wd = config.get('discriminator_wd')
    opt_gen = torch.optim.Adam(chain(label_generator.parameters(), sample_generator.parameters()), lr=gen_lr,
                               weight_decay=gen_wd)
    opt_dis = torch.optim.Adam(chain(label_discriminator.parameters(), sample_discriminator.parameters()), lr=dis_lr,
                               weight_decay=dis_wd)
    opt_sgen = torch.optim.Adam(sample_generator.parameters(), lr=gen_lr,
                                weight_decay=gen_wd)
    wasserstein_func = SinkhornDistance(0.1, 100, reduction='mean', device=device)
    writer = SummaryWriter()
    for epoch in range(epochs):
        dis_loss_sum = 0
        gen_loss_sum = 0
        distance_sum = 0
        label_generator.train()
        sample_generator.train()
        label_discriminator.train()
        sample_discriminator.train()
        rep_dataloader_iter = iter(rep_dataloader)
        for index, data in enumerate(obs_dataloader):
            try:
                rep_t_x_y = next(rep_dataloader_iter).to(device)
            except StopIteration:
                rep_dataloader_iter = iter(rep_dataloader)
                rep_t_x_y = next(rep_dataloader_iter).to(device)
            obs_t_x_y = data.to(device)
            normal_distribution = D.Normal(0, 1)
            normal_distribution.loc = normal_distribution.loc.to(device)
            normal_distribution.scale = normal_distribution.scale.to(device)
            rep_probability_gen = label_generator(rep_t_x_y)
            noise = normal_distribution.sample((len(obs_t_x_y), config.get('encoder_dim_latent')))
            unselected_t_x_y_gen = sample_generator(noise, bin_feats)
            loss3 = -torch.mean(torch.log(label_discriminator(obs_t_x_y) + 1e-4) + rep_probability_gen * torch.log(
                1 - label_discriminator(rep_t_x_y) + 1e-4))
            loss4 = -torch.mean((1 - rep_probability_gen) * torch.log(sample_discriminator(rep_t_x_y) + 1e-4) +
                                torch.log(1 - sample_discriminator(unselected_t_x_y_gen) + 1e-4))
            dis_loss = loss4 + loss3
            opt_dis.zero_grad()
            dis_loss.backward()
            opt_dis.step()
            dis_loss_sum += dis_loss
            rep_probability_gen = label_generator(rep_t_x_y)
            noise = normal_distribution.sample((len(obs_t_x_y), config.get('encoder_dim_latent')))
            unselected_t_x_y_gen = sample_generator(noise, bin_feats)
            unsel_probability_gen = label_generator(unselected_t_x_y_gen)
            loss1 = torch.mean((1 - unsel_probability_gen) * torch.log(
                sample_discriminator(unselected_t_x_y_gen) + 1e-4) + rep_probability_gen * torch.log(
                1 - label_discriminator(rep_t_x_y) + 1e-4) + unsel_probability_gen * torch.log(
                1 - label_discriminator(unselected_t_x_y_gen) + 1e-4) + (1 - rep_probability_gen) * torch.log(
                sample_discriminator(rep_t_x_y) + 1e-4))
            loss2 = torch.mean(
                unsel_probability_gen * torch.log(1 - label_discriminator(unselected_t_x_y_gen) + 1e-4) + torch.log(
                    1 - sample_discriminator(unselected_t_x_y_gen) + 1e-4) + (1 - unsel_probability_gen) * torch.log(
                    sample_discriminator(unselected_t_x_y_gen) + 1e-4))
            gen_loss = loss1 + loss2
            gen_loss_sum += gen_loss
            opt_gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()
            rep_probability_gen = label_generator(rep_t_x_y)
            labels = (rep_probability_gen >= 0.5)
            n_gen = 2 + (len(labels) - torch.sum(labels)) / (torch.sum(labels) + 1) * len(obs_t_x_y)
            noise = normal_distribution.sample(((n_gen.int()), config.get('encoder_dim_latent')))
            unselected_t_x_y_gen = sample_generator(noise, bin_feats)
            distance, _, _ = wasserstein_func(torch.cat((obs_t_x_y, unselected_t_x_y_gen), 0), rep_t_x_y)
            distance_sum += distance
            opt_sgen.zero_grad()
            distance.backward()
            opt_sgen.step()
        writer.add_scalar('Dis loss', dis_loss_sum, epoch)
        writer.add_scalar('Gen loss', gen_loss_sum, epoch)
        writer.add_scalar('Distance', distance_sum, epoch)
    writer.close()
    return label_generator, sample_generator, label_discriminator, sample_discriminator


def generate_labels(label_generator, data):
    label_generator.eval()
    with torch.no_grad():
        data = data.to(device)
        gen_label = label_generator(data)
    return gen_label


def train_estimator_bnn(estimator, train_dataloader, val_dataloader):
    epochs = config.get('est_epochs')
    lr = config.get('est_lr')
    weight_decay = config.get('est_wd')
    ipm_weight = config.get('IPM_weight')
    regression_loss_func = torch.nn.MSELoss(reduction='mean')
    wasserstein_func = SinkhornDistance(0.1, 100, reduction='mean', device=device)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=lr, weight_decay=weight_decay)
    writer = SummaryWriter()
    for epoch in range(epochs):
        loss_sum = 0
        estimator.train()
        for index, batch in enumerate(train_dataloader):
            t, x, ground_truth = batch
            t = t.to(device)
            x = x.to(device)
            ground_truth = ground_truth.to(device)
            y_pre, t_rep, c_rep = estimator(t, x)
            loss1 = regression_loss_func(y_pre.to(torch.float32), ground_truth.to(torch.float32))
            loss2, _, _ = wasserstein_func(t_rep, c_rep)
            loss = loss1 + ipm_weight * loss2
            loss_sum += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        writer.add_scalar('Estimator Train loss', loss_sum, epoch)
        loss_sum = 0
        estimator.eval()
        with torch.no_grad():
            for index, batch in enumerate(val_dataloader):
                t, x, ground_truth = batch
                t = t.to(device)
                x = x.to(device)
                ground_truth = ground_truth.to(device)
                y_pre, t_rep, c_rep = estimator(t, x)
                loss1 = regression_loss_func(y_pre.to(torch.float32), ground_truth.to(torch.float32))
                loss = loss1
                loss_sum += loss
        writer.add_scalar('Estimator Val loss', loss_sum, epoch)
    writer.close()
    return estimator


def test_estimator_bnn(estimator, t, x):
    estimator.eval()
    with torch.no_grad():
        t1 = torch.tensor(t, dtype=torch.float32).to(device)
        x1 = torch.tensor(x, dtype=torch.float32).to(device)
        y_pre, _, _ = estimator(t1, x1)
        y_pre = y_pre.detach().cpu().numpy()
    return y_pre.reshape((-1, 1))


data_path = Config.data
res_list = []
for i in range(Config.experiment_num):
    print("Data Preparation")
    obs_file_path = data_path + '_obs.csv'
    rep_file_path = data_path + '_rep.csv'
    unselected_file_path = data_path + '_unsel.csv'
    bin_feats = [0] + Config.bin_feats
    rep_t_train, rep_t_test, rep_t_val, rep_x_train, rep_x_test, rep_x_val, rep_y_train, rep_y_test, rep_y_val, rep_s_train, rep_s_test, rep_s_val, rep_gt_train, rep_gt_test, rep_gt_val = load_data(
        rep_file_path)
    obs_t_train, obs_t_test, obs_t_val, obs_x_train, obs_x_test, obs_x_val, obs_y_train, obs_y_test, obs_y_val, obs_s_train, obs_s_test, obs_s_val, obs_gt_train, obs_gt_test, obs_gt_val = load_data(
        obs_file_path)
    unselected_t_train, unselected_t_test, unselected_t_val, unselected_x_train, unselected_x_test, unselected_x_val, unselected_y_train, unselected_y_test, unselected_y_val, unselected_s_train, unselected_s_test, unselected_s_val, unselected_gt_train, unselected_gt_test, unselected_gt_val = load_data(
        unselected_file_path)
    rep_generator_train_dataset, rep_generator_test_dataset, rep_generator_val_dataset = build_generator_dataset(
        rep_t_train, rep_t_test, rep_t_val, rep_x_train, rep_x_test, rep_x_val, rep_y_train, rep_y_test, rep_y_val)
    obs_generator_train_dataset, obs_generator_test_dataset, obs_generator_val_dataset = build_generator_dataset(
        obs_t_train, obs_t_test, obs_t_val, obs_x_train, obs_x_test, obs_x_val, obs_y_train, obs_y_test, obs_y_val)
    unsel_generator_train_dataset, unsel_generator_test_dataset, unsel_generator_val_dataset = build_generator_dataset(
        unselected_t_train, unselected_t_test, unselected_t_val, unselected_x_train, unselected_x_test,
        unselected_x_val, unselected_y_train, unselected_y_test, unselected_y_val)
    print("Generate Samples")
    rep_gen_train_dataloader = DataLoader(rep_generator_train_dataset, batch_size=config.get('gen_batch_num'),
                                          shuffle=True, drop_last=False)
    obs_gen_train_dataloader = DataLoader(obs_generator_train_dataset, batch_size=config.get('gen_batch_num'),
                                          shuffle=True, drop_last=True)
    vae = VariationalAutoencoder(config).to(device)
    vae = train_vae(vae, rep_gen_train_dataloader, bin_feats)
    sample_generator = vae.decoder
    normal_distribution = D.Normal(0, 1)
    normal_distribution.loc = normal_distribution.loc.to(device)
    normal_distribution.scale = normal_distribution.scale.to(device)
    sample_discriminator = Discriminator(config).to(device)
    label_generator = Generator(config).to(device)
    label_discriminator = Discriminator(config).to(device)
    label_generator, sample_generator, label_discriminator, sample_discriminator = train_gan(label_generator,
                                                                                             sample_generator,
                                                                                             label_discriminator,
                                                                                             sample_discriminator,
                                                                                             obs_gen_train_dataloader,
                                                                                             rep_gen_train_dataloader,
                                                                                             bin_feats
                                                                                             )
    generated_labels = generate_labels(label_generator,
                                       torch.tensor(np.concatenate((rep_t_train, rep_x_train, rep_y_train), axis=1),
                                                    dtype=torch.float32))
    labels = (generated_labels >= 0.5)
    n_gen = 2 + (len(labels) - torch.sum(labels)) / (torch.sum(labels) + 1) * len(obs_t_train)
    noise = normal_distribution.sample((n_gen.int(), config.get('encoder_dim_latent')))
    generated_samples = sample_generator(noise, bin_feats)
    generated_samples = generated_samples.detach().cpu().numpy()
    rep_estimator_train_dataset, rep_estimator_test_dataset, rep_estimator_val_dataset = build_estimator_dataset(
        rep_t_train, rep_t_test, rep_t_val, rep_x_train, rep_x_test, rep_x_val, rep_y_train, rep_y_test, rep_y_val)
    obs_estimator_train_dataset, obs_estimator_test_dataset, obs_estimator_val_dataset = build_estimator_dataset(
        obs_t_train, obs_t_test, obs_t_val, obs_x_train, obs_x_test, obs_x_val, obs_y_train, obs_y_test, obs_y_val)
    obs_cf_train_dataset, obs_cf_test_dataset, obs_cf_val_dataset = build_cf_dataset(
        obs_t_train, obs_t_test, obs_t_val, obs_x_train, obs_x_test, obs_x_val, obs_y_train, obs_y_test, obs_y_val)
    gen_estimator_train_dataset, gen_estimator_test_dataset, gen_estimator_val_dataset = build_estimator_dataset(
        np.concatenate((obs_t_train, generated_samples[:, 0].reshape(-1, 1), rep_t_train), axis=0),
        obs_t_test, obs_t_val,
        np.concatenate(
            (obs_x_train, generated_samples[:, [i for i in range(1, Config.confounds_num + 1)]], rep_x_train), axis=0),
        obs_x_test, obs_x_val,
        np.concatenate((obs_y_train, generated_samples[:, Config.confounds_num + 1].reshape(-1, 1), rep_y_train),
                       axis=0),
        obs_y_test, obs_y_val
    )
    gen_est_train_dataloader = DataLoader(gen_estimator_train_dataset, batch_size=config.get('est_batch_num'),
                                          shuffle=True, drop_last=False)
    gen_est_val_dataloader = DataLoader(gen_estimator_val_dataset, batch_size=config.get('est_batch_num') // 3,
                                        shuffle=True, drop_last=False)
    print("Train Estimator")
    gan_estimator = BNN(config).to(device)
    gan_estimator = train_estimator_bnn(gan_estimator, gen_est_train_dataloader, gen_est_val_dataloader)
    gan_y_fact = test_estimator_bnn(gan_estimator, obs_t_test, obs_x_test)
    gan_y_cf = test_estimator_bnn(gan_estimator, 1 - obs_t_test, obs_x_test)
    unsel_gan_y_fact = test_estimator_bnn(gan_estimator, unselected_t_test, unselected_x_test)
    unsel_gan_y_cf = test_estimator_bnn(gan_estimator, 1 - unselected_t_test, unselected_x_test)
    ite_gan = np.where(obs_t_test == 1, gan_y_fact - gan_y_cf, gan_y_cf - gan_y_fact).reshape(-1, 1)
    unsel_ite_gan = np.where(unselected_t_test == 1, unsel_gan_y_fact - unsel_gan_y_cf,
                             unsel_gan_y_cf - unsel_gan_y_fact).reshape(-1, 1)
    res_list.append(np.sqrt(np.mean(np.square(ite_gan - obs_gt_test))))
    res_list.append(np.sqrt(np.mean(np.square(unsel_ite_gan - unselected_gt_test))))
    if device == 'cuda':
        torch.cuda.empty_cache()
res_list = np.array(res_list).reshape(-1, 2)
sd = np.std(res_list, axis=0).reshape(res_list.shape[1], 1)
mae = np.mean(np.abs(res_list), axis=0).reshape(res_list.shape[1], 1)
np.savetxt('res/result.txt', np.concatenate((mae, sd), 0))
