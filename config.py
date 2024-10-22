class Config:
    data = 'data/ihdp'
    experiment_num = 50
    confounds_num = 25
    bin_feats = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    config = {
        'encoder_dim_in': confounds_num + 2,
        'encoder_dim_out': 10,
        'encoder_dim_latent': 2,
        'encoder_layer_num': 2,
        'encoder_lr': 0.003,
        'encoder_wd': 0.0003,
        'vae_epochs': 100,
        'gan_epochs': 200,
        'est_epochs': 100,
        'gen_batch_num': 60,
        'discriminator_dim_in': confounds_num + 2,
        'discriminator_dim_out': 10,
        'discriminator_dim_latent': 1,
        'discriminator_layer_num': 2,
        'discriminator_lr': 0.003,
        'discriminator_wd': 0.0003,
        'generator_dim_in': confounds_num + 2,
        'generator_dim_out': 10,
        'generator_dim_latent': 1,
        'generator_layer_num': 2,
        'generator_lr': 0.005,
        'generator_wd': 0.0005,
        'estimator_dim_in': confounds_num,
        'estimator_dim_out': 10,
        'estimator_dim_latent': 1,
        'estimator_layer_num': 2,
        'est_lr': 0.01,
        'est_wd': 0.001,
        'est_batch_num': 300,
        'IPM_weight': 0.01,
    }
