import time
import copy

import torch
import torch.nn.functional as F

from .config import CKPT_DIR, RESULT_DIR, T_DIFF
from .utils import save_checkpoint, save_json
from .models.ddpm import make_diffusion_schedule, q_sample, ema_update


def fit_recon(model, loaders, device, epochs=30, lr=1e-3,
              ckpt_name=None, in_dim=28 * 28):
    model.to(device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=lr,
    )
    history = {"train_loss": [], "val_loss": [], "epoch_time": []}
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        train_total = train_count = 0
        for images, _ in loaders["train"]:
            images = images.to(device)
            flat_images = images.reshape(images.size(0), -1)
            loss = F.mse_loss(model(images), flat_images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total += loss.item() * images.size(0)
            train_count += images.size(0)

        model.eval()
        val_total = val_count = 0
        with torch.no_grad():
            for images, _ in loaders["val"]:
                images = images.to(device)
                flat_images = images.reshape(images.size(0), -1)
                val_total += F.mse_loss(
                    model(images), flat_images, reduction="sum"
                ).item() / in_dim
                val_count += images.size(0)

        if device.type == "cuda":
            torch.cuda.synchronize()
        train_loss = train_total / train_count
        val_loss = val_total / val_count
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epoch_time"].append(time.time() - start)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                key: value.clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"Epoch {epoch:02d} | train {train_loss:.5f} "
            f"| val {val_loss:.5f} | {history['epoch_time'][-1]:.1f}s"
        )

    model.load_state_dict(best_state)
    if ckpt_name:
        save_checkpoint(
            CKPT_DIR / f"{ckpt_name}.pt",
            model, optimizer, epochs, history,
        )
        save_json(history, RESULT_DIR / f"{ckpt_name}_history.json")
    return history


def fit_generic(model, loss_fn, loaders, device, epochs=30, lr=1e-3,
                ckpt_name=None, verbose=True):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "epoch_time": []}
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        train_total = train_count = 0
        components = {}

        for images, _ in loaders["train"]:
            images = images.to(device)
            loss, values = loss_fn(model, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total += loss.item() * images.size(0)
            train_count += images.size(0)
            for name, value in values.items():
                components[name] = components.get(name, 0.0) + value * images.size(0)

        model.eval()
        val_total = val_count = 0
        with torch.no_grad():
            for images, _ in loaders["val"]:
                images = images.to(device)
                loss, _ = loss_fn(model, images)
                val_total += loss.item() * images.size(0)
                val_count += images.size(0)

        if device.type == "cuda":
            torch.cuda.synchronize()
        train_loss = train_total / train_count
        val_loss = val_total / val_count
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epoch_time"].append(time.time() - start)

        for name, value in components.items():
            history.setdefault(f"train_{name}", []).append(value / train_count)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                key: value.clone()
                for key, value in model.state_dict().items()
            }

        if verbose:
            component_text = " ".join(
                f"{name}={value / train_count:.4f}"
                for name, value in components.items()
            )
            print(
                f"Epoch {epoch:02d} | train {train_loss:.5f} "
                f"| val {val_loss:.5f} | {component_text} "
                f"| {history['epoch_time'][-1]:.1f}s"
            )

    model.load_state_dict(best_state)
    if ckpt_name:
        save_checkpoint(
            CKPT_DIR / f"{ckpt_name}.pt",
            model, optimizer, epochs, history,
        )
        save_json(history, RESULT_DIR / f"{ckpt_name}_history.json")
    return history


def train_gan(generator, discriminator, name, loaders, device,
              epochs=50, lr=2e-4, z_dim=100, save_every=5):
    generator.to(device)
    discriminator.to(device)
    opt_generator = torch.optim.Adam(
        generator.parameters(), lr=lr, betas=(0.5, 0.999)
    )
    opt_discriminator = torch.optim.Adam(
        discriminator.parameters(), lr=lr, betas=(0.5, 0.999)
    )
    history = {
        "loss_D": [], "loss_G": [], "D_real": [], "D_fake": [],
        "epoch_time": [],
    }

    checkpoint_path = CKPT_DIR / f"{name}.pt"
    start_epoch = 0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint["G"])
        discriminator.load_state_dict(checkpoint["D"])
        opt_generator.load_state_dict(checkpoint["optG"])
        opt_discriminator.load_state_dict(checkpoint["optD"])
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"]
        print(f"[resume] {name} dari epoch {start_epoch}")

    bce = F.binary_cross_entropy_with_logits
    for epoch in range(start_epoch + 1, epochs + 1):
        start = time.time()
        generator.train()
        discriminator.train()
        discriminator_loss = generator_loss = real_score = fake_score = count = 0

        for images, _ in loaders["train"]:
            images = images * 2.0 - 1.0
            images = images.to(device)
            batch_size = images.size(0)
            ones = torch.ones(batch_size, device=device)
            zeros = torch.zeros(batch_size, device=device)

            fake = generator(torch.randn(batch_size, z_dim, device=device))
            real_output = discriminator(images)
            fake_output = discriminator(fake.detach())
            loss_discriminator = (
                bce(real_output, ones) + bce(fake_output, zeros)
            )
            opt_discriminator.zero_grad()
            loss_discriminator.backward()
            opt_discriminator.step()

            loss_generator = bce(discriminator(fake), ones)
            opt_generator.zero_grad()
            loss_generator.backward()
            opt_generator.step()

            discriminator_loss += loss_discriminator.item() * batch_size
            generator_loss += loss_generator.item() * batch_size
            real_score += torch.sigmoid(real_output).mean().item() * batch_size
            fake_score += torch.sigmoid(fake_output).mean().item() * batch_size
            count += batch_size

        if device.type == "cuda":
            torch.cuda.synchronize()
        history["loss_D"].append(discriminator_loss / count)
        history["loss_G"].append(generator_loss / count)
        history["D_real"].append(real_score / count)
        history["D_fake"].append(fake_score / count)
        history["epoch_time"].append(time.time() - start)
        print(
            f"Epoch {epoch:02d} | D {discriminator_loss/count:.3f} "
            f"| G {generator_loss/count:.3f} "
            f"| D(x) {real_score/count:.2f} "
            f"| D(G(z)) {fake_score/count:.2f} "
            f"| {history['epoch_time'][-1]:.1f}s"
        )

        if epoch % save_every == 0 or epoch == epochs:
            torch.save({
                "G": generator.state_dict(),
                "D": discriminator.state_dict(),
                "optG": opt_generator.state_dict(),
                "optD": opt_discriminator.state_dict(),
                "epoch": epoch,
                "history": history,
            }, checkpoint_path)
            save_json(history, RESULT_DIR / f"{name}_history.json")

    return history


def train_ddpm(model, ema_model, loaders, device, epochs=40, lr=2e-4,
               name="ddpm", save_every=5):
    model.to(device)
    ema_model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    schedule = make_diffusion_schedule(device)
    history = {"train_loss": [], "epoch_time": []}
    checkpoint_path = CKPT_DIR / f"{name}.pt"
    start_epoch = 0

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"]
        print(f"[resume] {name} dari epoch {start_epoch}")

    for epoch in range(start_epoch + 1, epochs + 1):
        start = time.time()
        model.train()
        total = count = 0
        for images, _ in loaders["train"]:
            images = images.to(device) * 2.0 - 1.0
            batch_size = images.size(0)
            timesteps = torch.randint(
                0, T_DIFF, (batch_size,), device=device
            )
            noise = torch.randn_like(images)
            noisy = q_sample(images, timesteps, noise, schedule)
            predicted = model(noisy, timesteps)
            loss = F.mse_loss(predicted, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_update(ema_model, model)
            total += loss.item() * batch_size
            count += batch_size

        if device.type == "cuda":
            torch.cuda.synchronize()
        history["train_loss"].append(total / count)
        history["epoch_time"].append(time.time() - start)
        print(
            f"Epoch {epoch:02d} | train {total/count:.5f} "
            f"| {history['epoch_time'][-1]:.1f}s"
        )

        if epoch % save_every == 0 or epoch == epochs:
            torch.save({
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "history": history,
            }, checkpoint_path)
            save_json(history, RESULT_DIR / f"{name}_history.json")

    return history
