import argparse
import copy

import torch
import torch.nn.functional as F

from src.config import (
    CKPT_DIR, FIG_DIR, RESULT_DIR, SEED, Z_DIM_AE,
)
from src.data import get_loaders, from_minus1_1
from src.utils import set_seed, get_device, ensure_dirs, save_checkpoint, save_json
from src.models import (
    MLPClassifier, FrozenLatentRecon, VanillaAE, SparseAE, VAE,
    VanillaGAN_G, VanillaGAN_D, DCGAN_G, DCGAN_D, dcgan_init,
    UNet,
)
from src.training import fit_recon, fit_generic, train_gan, train_ddpm
from src.evaluation import evaluate_recon, train_mlp, measure_latency
from src.visualization import plot_sample_grids, curves


def make_sparse_loss(lam):
    def loss(model, images):
        latent = model.encode(images)
        reconstructed = model.dec(latent)
        flat_images = images.reshape(images.size(0), -1)
        recon = F.mse_loss(reconstructed, flat_images)
        l1 = lam * latent.abs().sum(1).mean()
        return recon + l1, {"recon": recon.item(), "l1": l1.item()}
    return loss


def make_vae_loss(model, images):
    reconstructed, mu, logvar = model(images)
    flat_images = images.reshape(images.size(0), -1)
    recon = F.binary_cross_entropy(
        reconstructed, flat_images, reduction="sum"
    ) / images.size(0)
    kl = (
        -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
    ).mean()
    return recon + kl, {"recon": recon.item(), "kl": kl.item()}


def train_pipeline(args):
    set_seed(SEED)
    device = get_device()
    ensure_dirs(CKPT_DIR, FIG_DIR, RESULT_DIR)
    loaders = get_loaders(batch_size=args.batch_size)
    print("Device:", device)
    print("Dataset: FashionMNIST | Seed: 42 | Batch:", args.batch_size)

    set_seed(SEED)
    mlp = MLPClassifier().to(device)
    mlp_history, mlp_test_acc = train_mlp(
        mlp, loaders, device, epochs=args.mlp_epochs
    )
    save_checkpoint(
        CKPT_DIR / "mlp_classifier.pt",
        mlp, torch.optim.Adam(mlp.parameters(), lr=1e-3),
        args.mlp_epochs, mlp_history,
    )
    save_json(
        mlp_history,
        RESULT_DIR / "mlp_classifier_history.json",
    )
    print("MLP test accuracy:", mlp_test_acc)

    for parameter in mlp.parameters():
        parameter.requires_grad = False
    mlp.eval()

    set_seed(SEED)
    frozen = FrozenLatentRecon(mlp)
    frozen_history = fit_recon(
        frozen, loaders, device, epochs=args.ae_epochs,
        ckpt_name="frozen_latent_decoder",
    )

    set_seed(SEED)
    vanilla_ae = VanillaAE()
    vanilla_history = fit_recon(
        vanilla_ae, loaders, device, epochs=args.ae_epochs,
        ckpt_name="vanilla_ae_z64",
    )

    set_seed(SEED)
    sparse_ae = SparseAE()
    sparse_history = fit_generic(
        sparse_ae, make_sparse_loss(args.sparse_lambda),
        loaders, device, epochs=args.ae_epochs,
        ckpt_name="sparse_ae_z64",
    )

    set_seed(SEED)
    vae = VAE()
    vae_history = fit_generic(
        vae, make_vae_loss, loaders, device,
        epochs=args.ae_epochs, ckpt_name="vae_z64",
    )

    frozen.eval()
    vanilla_ae.eval()
    sparse_ae.eval()
    vae.eval()
    recon_results = {
        "vanilla_ae": evaluate_recon(
            lambda images: vanilla_ae(images), loaders["test"], device
        ),
        "sparse_ae": evaluate_recon(
            lambda images: sparse_ae(images), loaders["test"], device
        ),
        "vae": evaluate_recon(
            lambda images: vae(images)[0], loaders["test"], device
        ),
        "frozen_latent_decoder": evaluate_recon(
            lambda images: frozen(images), loaders["test"], device
        ),
    }
    save_json(recon_results, RESULT_DIR / "ae_family_metrics.json")

    set_seed(SEED)
    vanilla_generator = VanillaGAN_G()
    vanilla_discriminator = VanillaGAN_D()
    vanilla_history = train_gan(
        vanilla_generator, vanilla_discriminator,
        "vanilla_gan", loaders, device,
        epochs=args.gan_epochs, z_dim=100,
    )

    set_seed(SEED)
    dcgan_generator = DCGAN_G()
    dcgan_discriminator = DCGAN_D()
    dcgan_generator.apply(dcgan_init)
    dcgan_discriminator.apply(dcgan_init)
    dcgan_history = train_gan(
        dcgan_generator, dcgan_discriminator,
        "dcgan", loaders, device,
        epochs=args.gan_epochs, z_dim=100,
    )

    set_seed(SEED)
    ddpm_model = UNet().to(device)
    ema_model = copy.deepcopy(ddpm_model).eval()
    ddpm_history = train_ddpm(
        ddpm_model, ema_model, loaders, device,
        epochs=args.ddpm_epochs,
    )

    samplers = {
        "VAE": lambda n: vae.decode(
            torch.randn(n, vae.z_dim, device=device)
        ).view(-1, 1, 28, 28),
        "Vanilla GAN": lambda n: from_minus1_1(
            vanilla_generator(torch.randn(n, 100, device=device))
        ),
        "DCGAN": lambda n: from_minus1_1(
            dcgan_generator(torch.randn(n, 100, device=device))
        ),
    }
    plot_sample_grids(samplers, ["VAE", "Vanilla GAN", "DCGAN"])

    latency = {
        name: measure_latency(
            lambda fn=sampler: fn(64),
            warmup=1 if args.quick else 10,
            runs=3 if args.quick else 50,
            device=device,
        )
        for name, sampler in samplers.items()
    }
    print("Latency inferensi per batch 64 sampel (ms):")
    for name, value in latency.items():
        print(f"  {name:<12} {value:.2f}")

    history_map = {
        "Vanilla AE": vanilla_history,
        "Sparse AE": sparse_history,
        "VAE": vae_history,
        "Vanilla GAN": vanilla_history,
        "DCGAN": dcgan_history,
        "DDPM": ddpm_history,
    }
    save_json({
        "mlp_test_acc": mlp_test_acc,
        "recon_metrics": recon_results,
        "latency_ms": latency,
    }, RESULT_DIR / "final_results.json")

    curves(
        history_map,
        "Training loss comparision",
        [
            ("Vanilla GAN", "loss_G"),
            ("DCGAN", "loss_G"),
        ],
        ylabel="Generator loss",
        filename="gan_training_curves.png",
    )


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Modular implementation of the generative models assignment"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ae-epochs", type=int, default=30)
    parser.add_argument("--gan-epochs", type=int, default=50)
    parser.add_argument("--ddpm-epochs", type=int, default=40)
    parser.add_argument("--mlp-epochs", type=int, default=30)
    parser.add_argument("--sparse-lambda", type=float, default=1e-4)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(args)


if __name__ == "__main__":
    arguments = parse_args()
    train_pipeline(arguments)
